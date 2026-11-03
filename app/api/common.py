from __future__ import annotations

from typing import Callable, TypeVar

import anyio
from anyio import CapacityLimiter

T = TypeVar("T")


class Overloaded(Exception):
    pass


class InferenceGate:
    def __init__(self, workers: int, max_queue: int) -> None:
        self.limiter = CapacityLimiter(workers)
        self.max_queue = max_queue
        self.waiting = 0

    async def run(self, fn: Callable[..., T], *args) -> T:
        if self.waiting >= self.max_queue:
            raise Overloaded()
        self.waiting += 1
        try:
            return await anyio.to_thread.run_sync(fn, *args, limiter=self.limiter)
        finally:
            self.waiting -= 1
