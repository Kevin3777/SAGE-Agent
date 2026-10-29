"""Injectable clocks make retention and lifecycle behavior deterministic."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class SystemClock:
    def now(self) -> float:
        return time.time()


@dataclass
class FakeClock:
    current: float = 0.0

    def now(self) -> float:
        return float(self.current)

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("FakeClock cannot move backwards")
        self.current += float(seconds)
        return self.current
