from sarathi.config import SchedulerType
from sarathi.core.scheduler.faster_transformer_scheduler import (
    FasterTransformerScheduler,
)
from sarathi.core.scheduler.orca_scheduler import OrcaScheduler
from sarathi.core.scheduler.sarathi_scheduler import SarathiScheduler
from sarathi.core.scheduler.simple_chunking_scheduler import SimpleChunkingScheduler
from sarathi.core.scheduler.vllm_scheduler import VLLMScheduler
from sarathi.core.scheduler.hold_n_scheduler import Hold_NScheduler
from sarathi.core.scheduler.slai_scheduler import SLAIScheduler
from sarathi.utils.base_registry import BaseRegistry


class SchedulerRegistry(BaseRegistry):

    @classmethod
    def get_key_from_str(cls, key_str: str) -> SchedulerType:
        return SchedulerType.from_str(key_str)


SchedulerRegistry.register(SchedulerType.VLLM, VLLMScheduler)
SchedulerRegistry.register(SchedulerType.ORCA, OrcaScheduler)
SchedulerRegistry.register(SchedulerType.FASTER_TRANSFORMER, FasterTransformerScheduler)
SchedulerRegistry.register(SchedulerType.SARATHI, SarathiScheduler)
SchedulerRegistry.register(SchedulerType.SIMPLE_CHUNKING, SimpleChunkingScheduler)
SchedulerRegistry.register(SchedulerType.HOLD_N, Hold_NScheduler)
SchedulerRegistry.register(SchedulerType.SLAI_SCHEDULER, SLAIScheduler)
