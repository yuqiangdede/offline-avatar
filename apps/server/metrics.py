# File: offline-avatar/apps/server/metrics.py
from __future__ import annotations

import time


class Metrics:
    @staticmethod
    def now() -> float:
        return time.perf_counter()

    @staticmethod
    def elapsed_ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)
