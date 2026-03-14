"""utils/fps_counter.py — Simple rolling-window FPS counter."""

from __future__ import annotations
import time
from collections import deque


class FPSCounter:
    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self):
        self._times.append(time.monotonic())

    def get(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        return (len(self._times) - 1) / span

    @property
    def fps(self) -> float:
        return self.get()
