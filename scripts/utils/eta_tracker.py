"""ETA tracking utilities."""
from __future__ import annotations

from collections import deque
from typing import Deque


class EtaTracker:
    """Track average batch duration to estimate remaining time."""

    def __init__(self, start_batch: int = 20, window: int = 10):
        self.start_batch = start_batch
        self.window = window
        self.times: Deque[float] = deque(maxlen=window)

    def update(self, batch_idx: int, batch_duration_s: float, total_batches: int | None = None) -> float | None:
        if batch_idx < self.start_batch:
            return None
        self.times.append(batch_duration_s)
        if not self.times or total_batches is None:
            return None
        remaining = total_batches - (batch_idx + 1)
        avg = sum(self.times) / len(self.times)
        return remaining * avg
