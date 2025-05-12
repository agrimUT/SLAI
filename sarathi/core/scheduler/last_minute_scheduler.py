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
        self.decode_queue: List[Tuple[float, float, Sequence]] = [] # Store LST, Arrival time, and Sequence obj, it represents decode sequences that have run at least once and thus have a last schedulable time
        self.paused_prefills: List[Sequence] = [] # Store Sequence obj, it represents the set of sequences that are in prefill phase and have already been scheduled at least once
        self.process_smallest_prefill = self.scheduler_config.process_smallest_prefill # if true then process the smallest prefill first
        self.prefill_waiting = 0 
    def get_queue_sizes(self) -> tuple[int, int]:
        """Return (#prefill_waiting, #decode_waiting)."""
        return (self.prefill_waiting + len(self.paused_prefills)), (len(self.decode_queue))
    
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
        # self.running here represents the previous batch that was processed - we are going to construct our decode queue and paused prefill jobs from this
        if(len(self.running) == 0):
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
            # eithe
            if not seq.prompt_processing_finished:
                if seq not in self.paused_prefills:
                     self.paused_prefills.append(seq)
            elif seq.prompt_processing_finished and not seq.is_finished(): 
                if self.next_batch_run_time_for_decode == float("inf"):
                    self.next_batch_run_time_for_decode = self.last_batch_end_time + self.minimum_time_between_tokens - self.offset * (self.last_batch_end_time - self.last_batch_start_time)
                seq.last_schedulable_time = min(self.next_batch_run_time_for_decode, self.last_batch_end_time + self.minimum_time_between_tokens - self.offset * (self.last_batch_end_time - self.last_batch_start_time))
                heapq.heappush(self.decode_queue, (seq.last_schedulable_time, seq.arrival_time, seq)) 

    
    def _schedule(self) -> SchedulerOutputs:
        # Fix the current time.
        now = time.monotonic()

        running: List[Sequence] = [] # what I am going to include in the batch
        ignored_seq_ids: List[int] = []
        preempted_seq_ids: List[int] = []
        scheduled_seq_metadata_list: List[SequenceScheduleMetadata] = []
        num_batched_tokens: int = 0
        critical_decodes = 0
        noncritical_decodes = 0
        preempted_seq_prefill = 0 
        preempted_seq_decode = 0
        prefill_waiting = 0

        self._post_batch_processing() # Add sequences to decode queue and paused prefill 
        self.paused_prefills = self.policy.sort_by_priority(now, self.paused_prefills) # sort paused prefill sequences based on FCFS
                
        while self.decode_queue and num_batched_tokens < self.token_budget: 
            seq = self.decode_queue[0][2]
            if now >= seq.last_schedulable_time: 
                seq = heapq.heappop(self.decode_queue)[2]
                critical_decodes += 1
                if not seq.is_paused():
                    running.append(seq)
                    continue
                while not self.block_manager.can_append_slot():
                    if self.paused_prefills:
                        # Preempt the paused prefill sequence that too whichever arrived last 
                        victim_seq = self.paused_prefills.pop(-1)
                        self._preempt(victim_seq)
                        preempted_seq_ids.append(victim_seq.seq_id)
                        preempted_seq_prefill += 1
                    else:
                        critical_decodes -= 1
                        self._preempt(seq)
                        preempted_seq_ids.append(seq.seq_id)
                        preempted_seq_decode += 1
                        break 
                else:
                    # Append new slots to the sequence group.
                    self._append_slot(seq)
                    running.append(seq)
                    num_batched_tokens += 1
                    scheduled_seq_metadata_list.append(
                        SequenceScheduleMetadata.from_sequence(seq)
                    )
            else:
                break
        if self.process_smallest_prefill: 
            # processing paused prefills with smallest number of tokens left in the prefill phase
            self.paused_prefills = sorted(self.paused_prefills, key=lambda seq: (seq.get_prompt_len() - seq.get_num_prompt_tokens_processed(), seq.arrival_time))
        # process the paused prefill jobs
        while self.paused_prefills and num_batched_tokens < self.token_budget:
            seq = self.paused_prefills.pop(0) 
            assert not seq.prompt_processing_finished
            assert self.block_manager.is_allocated(seq) # ensure that this sequence was already allocated GPU blocks

            next_num_prefill_tokens = self._get_seq_next_num_prefill_tokens(
                seq, num_batched_tokens
            )
            assert next_num_prefill_tokens > 0, f"Prefill tokens should be > 0, got {next_num_prefill_tokens} for seq {seq.seq_id}"

            num_batched_tokens += next_num_prefill_tokens
            scheduled_seq_metadata_list.append(
                SequenceScheduleMetadata.from_sequence(
                    seq, prompt_chunk_len=next_num_prefill_tokens
                )
            )
            running.append(seq) 
        if self.process_smallest_prefill:
            # reorder ONLY the arrivals that are <= now
            k = 0
            while k < len(self.waiting) and self.waiting[k].arrival_time <= now:
                k += 1
            self.waiting[:k] = sorted(self.waiting[:k], key=lambda seq: (seq.get_prompt_len() - seq.get_num_prompt_tokens_processed(), seq.arrival_time))
        # process the jobs from the waiting queue 
        while self.waiting and num_batched_tokens < self.token_budget:
            seq = self.waiting[0]
            if seq.arrival_time > now:
                break
            prefill_waiting += 1
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
            prefill_waiting -= 1
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
            seq = heapq.heappop(self.decode_queue)[2]
            noncritical_decodes += 1
            if not seq.is_paused():
                running.append(seq)
                continue
            while not self.block_manager.can_append_slot():
                if self.paused_prefills:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq = self.paused_prefills.pop(-1)
                    self._preempt(victim_seq)
                    preempted_seq_ids.append(victim_seq.seq_id)
                    preempted_seq_prefill += 1
                else:
                    noncritical_decodes -= 1
                    self._preempt(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    preempted_seq_decode += 1
                    break 
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq)
                running.append(seq)
                num_batched_tokens += 1
                scheduled_seq_metadata_list.append(
                    SequenceScheduleMetadata.from_sequence(seq)
                )
            
        
        self.running = running
        self.prefill_waiting = prefill_waiting
        return SchedulerOutputs(
            id=self._iteration_id,
            ignored_seq_ids=ignored_seq_ids,
            preempted_seq_ids=preempted_seq_ids,
            scheduled_seq_metadata_list=scheduled_seq_metadata_list,
            num_time_critical_decodes=critical_decodes,
            num_noncritical_decodes=noncritical_decodes,
            preempted_seq_prefill=preempted_seq_prefill,
            preempted_seq_decode=preempted_seq_decode,
        )
