"""Numerical form of SAGE Eq. 6: R(tau, S) = exp(-tau / S)."""
from __future__ import annotations

import math


class ForgettingCurve:
    SECONDS_PER_UNIT = {
        "seconds": 1.0,
        "minutes": 60.0,
        "hours": 3600.0,
        "days": 86400.0,
    }

    def __init__(self, time_unit: str = "seconds") -> None:
        if time_unit not in self.SECONDS_PER_UNIT:
            raise ValueError(f"time_unit must be one of {sorted(self.SECONDS_PER_UNIT)}")
        self.time_unit = time_unit
        self.seconds_per_unit = self.SECONDS_PER_UNIT[time_unit]

    @staticmethod
    def retention(elapsed: float, strength: float) -> float:
        elapsed = float(elapsed)
        strength = float(strength)
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed tau must be finite and >= 0")
        if not math.isfinite(strength) or strength <= 0:
            raise ValueError("strength S must be finite and > 0")
        return min(1.0, max(0.0, math.exp(-elapsed / strength)))

    def since(self, last_timestamp: float, now: float, strength: float) -> float:
        # Clock timestamps use Unix seconds. Convert to the selected unit before
        # applying Eq. 6 so strength and elapsed time use matching units.
        elapsed = (float(now) - float(last_timestamp)) / self.seconds_per_unit
        return self.retention(elapsed, strength)
