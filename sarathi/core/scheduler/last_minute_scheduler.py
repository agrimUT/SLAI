import time
from typing import List, Tuple
import heapq 
import numpy as np

from sarathi.config import CacheConfig, LastMinuteSchedulerConfig
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.core.datatypes.sequence_status import SequenceStatus  
from sarathi.logger import init_logger

logger = init_logger(__name__)


class LastMinuteScheduler(BaseScheduler):

    def __init__(
        self,
        scheduler_config: LastMinuteSchedulerConfig,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__(scheduler_config, cache_config)

        self.prompt_limit = self.scheduler_config.max_model_len
        self.token_budget = self.scheduler_config.token_budget
        self.offset = self.scheduler_config.offset * 1
        self.next_batch_run_time_for_decode = float("inf")
        self.minimum_time_between_tokens = float("inf")
        self.last_batch_start_time = 0
        self.last_batch_end_time = 0
        self.time_between_tokens = self.scheduler_config.time_between_tokens
        self.decode_queue: List[Tuple[float, int, Sequence]] = [] # Store LST, ID, and Sequence obj

    def get_block_space_manager_class(self):
        return SarathiBlockSpaceManager

    def _get_seq_next_num_prefill_tokens(
        self, seq: Sequence, num_batched_tokens: int
    ) -> int:
        assert not seq.is_finished()

        next_num_tokens = min(
            seq.get_prompt_len() - seq.get_num_prompt_tokens_processed(),
            self.token_budget - num_batched_tokens,
        )

        return next_num_tokens

    def set_last_batch_times(
        self,
        batch_start_time: float,
        batch_end_time: float,
    ) -> None:
        # Update the last batch start and end time
        self.last_batch_start_time = batch_start_time
        self.last_batch_end_time = batch_end_time
        
    def _post_batch_processing(self) -> None:
        if(len(self.running) == 0):
            return 
        # check if there was a decode job in the batch 
        logger.debug(f"Running _post_batch_processing based on previous self.running (size {len(self.running)})")
        decode_job_in_batch = False
        for seq in self.running:
            if seq.prompt_processing_finished:
                decode_job_in_batch = True
                break
        self.next_batch_run_time_for_decode = float("inf")
        # if there was then set the next time a decode batch is going to run 
        if decode_job_in_batch:
            self.minimum_time_between_tokens = float("inf")
            for seq in self.running:
                if seq.prompt_processing_finished:
                    # calculate the time between tokens for this sequence
                    self.minimum_time_between_tokens = min(self.minimum_time_between_tokens, self.time_between_tokens)
            self.next_batch_run_time_for_decode = self.last_batch_end_time + self.minimum_time_between_tokens - self.offset * (self.last_batch_end_time - self.last_batch_start_time)
        
        for seq in self.running:
            if not seq.prompt_processing_finished:
                # check whether it has completed all prefill tokens 
                if seq.get_num_prompt_tokens_processed() == seq.get_prompt_len():
                    if self.next_batch_run_time_for_decode == float("inf"):
                        self.next_batch_run_time_for_decode = self.last_batch_end_time + self.minimum_time_between_tokens - self.offset * (self.last_batch_end_time - self.last_batch_start_time)
                    seq.last_schedulable_time = min(self.next_batch_run_time_for_decode, self.last_batch_end_time + self.minimum_time_between_tokens - self.offset * (self.last_batch_end_time - self.last_batch_start_time))
                    heapq.heappush(self.decode_queue, (seq.last_schedulable_time, seq.seq_id, seq)) 
                    logger.debug(f"Seq {seq.seq_id}: Calculated LST = {seq.last_schedulable_time}")
                    logger.debug(f"Seq {seq.seq_id}: Pushing ({seq.last_schedulable_time}, seq_id={seq.seq_id}) to heap.")
                else: 
                    self.waiting.insert(0, seq)
            else:
                seq.last_schedulable_time = self.next_batch_run_time_for_decode
                heapq.heappush(self.decode_queue, (seq.last_schedulable_time, seq.seq_id, seq)) 

    
    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        running: List[Sequence] = [] # what I am going to include in the batch
        ignored_seq_ids: List[int] = []
        preempted_seq_ids: List[int] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []
        num_batched_tokens: int = 0

        logger.debug(f"--- Schedule Step {self._iteration_id} ---")
        logger.debug(f"Time: {now}")
        logger.debug(f"State BEFORE post_batch: Running={len(self.running)}, Waiting={len(self.waiting)}, DecodeHeap={len(self.decode_queue)}")
        # Log top few heap items if not empty
        if self.decode_queue:
            top_items = heapq.nsmallest(5, self.decode_queue)
            logger.debug(f"DecodeHeap Top 5 (LST, SeqID): {[(lst, seq_id, s) for lst, seq_id, s in top_items]}")

        self._post_batch_processing() # Call the state update logic

        logger.debug(f"State AFTER post_batch: Running={len(self.running)}, Waiting={len(self.waiting)}, DecodeHeap={len(self.decode_queue)}")
        if self.decode_queue:
            top_items = heapq.nsmallest(5, self.decode_queue)
            logger.debug(f"DecodeHeap Top 5 (LST, SeqID) after update: {[(lst, seq_id, s) for lst, seq_id, s in top_items]}")

        # Log free blocks
        free_blocks = self.block_manager.get_num_free_gpu_blocks()
        logger.debug(f"Free Blocks: {free_blocks}")
        # Remove things from decode queue that have finished everything? Depends on when is the status of different jobs set        
        while self.decode_queue and num_batched_tokens < self.token_budget: 
            seq = self.decode_queue[0][2]
            if now >= seq.last_schedulable_time: 
                if self.block_manager.can_append_slot():
                    seq = heapq.heappop(self.decode_queue)[2]
                    seq.status = SequenceStatus.RUNNING
                    self._append_slot(seq)
                    running.append(seq)
                    num_batched_tokens += 1
                    scheduled_seq_metadata_list.append(
                        SequenceScheduleMetadata.from_sequence(seq)
                    )
                else:
                    logger.debug(f"No slot for critical decode seq {seq.seq_id}. Preemption needed?")
                    break 
            else:
                break
       
        # process the jobs from the waiting queue 
        while self.waiting and num_batched_tokens < self.token_budget:
            seq = self.waiting[0]
            if seq.arrival_time > now:
                break
            
            if not self._check_request_prompt_length(seq): # model can not handle this big of a prompt
                ignored_seq_ids.append(seq.seq_id)
                continue
                
            # If the sequence group cannot be allocated, stop.
            if not self.block_manager.can_allocate(seq):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            if len(running) >= self.scheduler_config.max_num_seqs:
                break

            # check if we can fit the prefill in the batch
            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )

            if next_num_prefill_tokens == 0:
                break

            seq = self.waiting.pop(0)
            self._allocate(seq)
            seq.status = SequenceStatus.RUNNING
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)
        # now go through the remaining decode queue and add them to batch if there is space 
        while self.decode_queue and num_batched_tokens < self.token_budget:
            seq = self.decode_queue[0][2]
            if self.block_manager.can_append_slot():
                seq = heapq.heappop(self.decode_queue)[2]
                seq.status = SequenceStatus.RUNNING
                self._append_slot(seq)
                running.append(seq)
                num_batched_tokens += 1
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )
            else:
                break 
            
        
        self.running = running
        logger.debug(f"State AFTER scheduling: Running={len(self.running)}, Waiting={len(self.waiting)}, DecodeHeap={len(self.decode_queue)}")
        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
