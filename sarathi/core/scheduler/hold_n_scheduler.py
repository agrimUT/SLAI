# sarathi/core/scheduler/hold_n_scheduler.py
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: MIT
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List

from sarathi.config import BaseSchedulerConfig, CacheConfig, SchedulerType
from sarathi.core.block_space_manager.vllm_block_space_manager import VLLMBlockSpaceManager
from sarathi.core.datatypes.scheduler_output import SchedulerOutputs
from sarathi.core.datatypes.sequence import Sequence, SequenceScheduleMetadata
from sarathi.core.scheduler.base_scheduler import BaseScheduler
from sarathi.core.datatypes.sequence_status import SequenceStatus
from sarathi.logger import init_logger
logger = init_logger(__name__)


# --------------------------------------------------------------------- #
# 1. Config                                                             #
# --------------------------------------------------------------------- #
class Hold_NSchedulerConfig(BaseSchedulerConfig):
    def __init__(
        self,
        max_num_seqs: int,
        max_model_len: int,
        num_pipeline_stages: int,
        hold_n: int,
        token_budget: int,
        prefill_factor: int,
    ):
        super().__init__(max_num_seqs, max_model_len, num_pipeline_stages)
        self.hold_n = hold_n
        self.token_budget = token_budget
        self.prefill_factor = prefill_factor

    @property
    def max_num_batched_tokens(self):
        return self.token_budget

    @property
    def type(self):
        return SchedulerType.HOLD_N


# --------------------------------------------------------------------- #
# 2. Scheduler                                                          #
# --------------------------------------------------------------------- #
class Hold_NScheduler(BaseScheduler):
    """Wait until *hold_n* decode-ready sequences exist, then run them
    together with one filler prefill request **once**."""

    def __init__(self, scheduler_config: Hold_NSchedulerConfig, cache_config: CacheConfig):
        super().__init__(scheduler_config, cache_config)
        self.hold_n        = scheduler_config.hold_n
        self.token_budget  = scheduler_config.token_budget
        self.prefill_factor = scheduler_config.prefill_factor

        self.decode_buf: List[Sequence] = []
        self.has_fired = False           # set after the timed batch runs

        self.filler_seq: Sequence | None = None
        self.filler_seq_found: bool = False

    def get_block_space_manager_class(self):
        return VLLMBlockSpaceManager

    def on_step_completed(self) -> None:
        """
        Called by BaseLLMEngine after every batch.
        We use it to mark the prompt-only filler as finished once its
        prompt has completed.
        """
        super().on_step_completed()
        if not self.has_fired:
            # We haven’t emitted the timed batch yet → nothing to do.
            return

        # Any sequence that has finished its prompt and has max_tokens == 0
        # is our filler; mark it finished so the runner can exit.
        for seq in list(self.running):
            if seq.prompt_processing_finished and seq.sampling_params.max_tokens == 0:
                seq.set_status(SequenceStatus.FINISHED_LENGTH_CAPPED)
                self.running.remove(seq)


    # ------------------------------------------------------------------ main
    def _schedule(self) -> SchedulerOutputs:
        if self.has_fired:
            return SchedulerOutputs(self._iteration_id, [], [], [])
        if not self.filler_seq_found:
            for i, seq in enumerate(self.waiting):
                if seq.get_prompt_len() == self.token_budget - self.hold_n:
                    self.filler_seq = self.waiting.pop(i)
                    break 
            self.filler_seq_found = True

        now = time.monotonic()
        ignored, preempted, meta = [], [], []

        # put any newly-finished prefills into decode_buf -----------------
        finished = [seq for seq in self.running if seq.prompt_processing_finished]
        for seq in finished:
            self.running.remove(seq)
            if seq not in self.decode_buf:
                self.decode_buf.append(seq)

        # ---------------- phase 1: finish prefills -----------------------
        if len(self.decode_buf) < self.hold_n and self.waiting:
            seq = self.waiting.pop(0)
            if not self._check_request_prompt_length(seq):
                ignored.append(seq.seq_id)
            self._allocate(seq)
            self.running.append(seq)

        # schedule prompt chunks for all running sequences
        for seq in list(self.running):                       # iterate over a copy
            remaining = seq.get_prompt_len() - seq.get_num_prompt_tokens_processed()
            if remaining > 0:
                meta.append(SequenceScheduleMetadata.from_sequence(seq, prompt_chunk_len=remaining))

        if meta or ignored:
            return SchedulerOutputs(self._iteration_id, ignored, preempted, meta)

        # ---------------- phase 2: timed batch ---------------------------
        if len(self.decode_buf) >= self.hold_n:
            assert self.filler_seq is not None, "Filler sequence should have been found by now."
            filler_seq = self.filler_seq
            filler_seq.sampling_params.max_tokens = 0       # prompt-only
            self._allocate(filler_seq)
            self.running.append(filler_seq)                 

            filler_chunk = (filler_seq.get_prompt_len() - filler_seq.get_num_prompt_tokens_processed())
            meta.append(
                SequenceScheduleMetadata.from_sequence(
                    filler_seq, prompt_chunk_len=filler_chunk
                )
            )

            for seq in self.decode_buf[: self.hold_n]:
                self._append_slot(seq)
                meta.append(SequenceScheduleMetadata.from_sequence(seq))

            self.has_fired = True
            return SchedulerOutputs(self._iteration_id, [], [], meta)

        # nothing schedulable this iteration
        return SchedulerOutputs(self._iteration_id, [], [], [])