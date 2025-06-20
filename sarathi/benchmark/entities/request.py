import logging
from typing import Tuple

from sarathi.benchmark.entities.base_entity import BaseEntity

logger = logging.getLogger(__name__)


class Request(BaseEntity):

    def __init__(
        self,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        time_between_tokens: float | None = None, 
        is_strict_tbt: bool = None
    ):
        self._id = Request.generate_id()
        self._arrived_at = arrived_at
        self._num_prefill_tokens = num_prefill_tokens
        self._num_decode_tokens = num_decode_tokens
        self._time_between_tokens = time_between_tokens
        self._is_strict_tbt = is_strict_tbt
        assert num_prefill_tokens > 0
        #assert num_decode_tokens > 0

    @property
    def size(self) -> Tuple[int, int]:
        return (self._num_prefill_tokens, self._num_decode_tokens)

    @property
    def arrived_at(self) -> float:
        return self._arrived_at

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens

    @property
    def pd_ratio(self) -> float:
        return self._num_prefill_tokens / self._num_decode_tokens

    @property
    def total_tokens(self) -> int:
        return self._num_prefill_tokens + self._num_decode_tokens
    @property
    def time_between_tokens(self) -> float | None:
        return self._time_between_tokens
    @property
    def is_strict_tbt(self) -> bool | None:
        return self._is_strict_tbt
    
    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "arrived_at": self._arrived_at,
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
            "time_between_tokens": self._time_between_tokens,
            "is_strict_tbt": self._is_strict_tbt,
        }
