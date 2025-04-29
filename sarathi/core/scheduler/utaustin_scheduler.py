import time
from typing import List, Tuple
import heapq 
import numpy as np

from sarathi.config import CacheConfig, LastMinuteSchedulerConfig
from sarathi.core.block_space_manager.sarathi_block_space_manager import (
    SarathiBlockSpaceManager,
)
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata, SequenceMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
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
        self.token_budget = self.scheduler_config.chunk_size
        self.offset = self.scheduler_config.offset
        self.next_batch_run_time_for_decode = float("inf")
        self.minimum_time_between_tokens = float("inf")
        self.last_batch_start_time = 0
        self.last_batch_end_time = 0
        self.time_between_tokens = self.scheduler_config.time_between_tokens
        self.decode_queue: List[Tuple[float, int, Sequence]] = []

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
        if(self.running.size() == 0):
            return 
        # check if there was a decode job in the batch 
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
                    heapq.heappush(self.decode_queue, (seq.last_schedulable_time, seq)) 
                else: 
                    self.waiting.insert(0, seq)
            else:
                seq.last_schedulable_time = self.next_batch_run_time_for_decode
                heapq.heappush(self.decode_queue, (seq.last_schedulable_time, seq)) 
    
    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        running: List[Sequence] = [] # what I am going to include in the batch
        ignored_seq_ids: List[int] = []
        preempted_seq_ids: List[int] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []
        num_batched_tokens: int = 0

        self._post_batch_processing()
        
        # Remove things from decode queue that have finished everything? Depends on when is the status of different jobs set        
        while self.decode_queue and num_batched_tokens < self.token_budget: 
            seq = self.decode_queue[0][1]
            if now >= seq.last_schedulable_time: 
                if self.block_manager.can_append_slot():
                    seq = heapq.heappop(self.decode_queue)[1]
                    self._append_slot(seq)
                    running.append(seq)
                    num_batched_tokens += 1
                    scheduled_seq_metadata_list.append(
                        SequenceScheduleMetadata.from_sequence(seq)
                    )
                else:
                    logger.warning(f"No slot for critical decode seq {seq.seq_id}. Preemption needed?")
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
            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq)
        # now go through the remaining decode queue and add them to batch if there is space 
        while self.decode_queue and num_batched_tokens < self.token_budget:
            seq = self.decode_queue[0][1]
            if self.block_manager.can_append_slot():
                seq = heapq.heappop(self.decode_queue)[1]
                self._append_slot(seq)
                running.append(seq)
                num_batched_tokens += 1
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )
            else:
                break 
            
        
        self.running = running

        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
        )
