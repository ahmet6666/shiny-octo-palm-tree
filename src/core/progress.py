from __future__ import annotations

"""Batch progress tracking with ETA."""

from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Deque


@dataclass
class BatchProgress:
    total_batches: int
    window: int = 20

    def __post_init__(self) -> None:
        self.done = 0
        self.start = monotonic()
        self.times: Deque[float] = deque(maxlen=self.window)

    def update(self, n: int = 1, batch_time: float | None = None) -> None:
        self.done += n
        if batch_time is None:
            batch_time = monotonic() - self.start
        self.times.append(batch_time)
        self.start = monotonic()

    @property
    def percent(self) -> float:
        return 100.0 * self.done / self.total_batches if self.total_batches else 0.0

    @property
    def eta(self) -> float | None:
        if self.done < self.window or not self.times:
            avg = sum(self.times) / max(len(self.times), 1)
        else:
            avg = sum(self.times) / len(self.times)
        remaining = self.total_batches - self.done
        return remaining * avg if remaining > 0 else 0.0
