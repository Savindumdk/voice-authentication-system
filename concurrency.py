"""
Concurrency helpers (Phase 2 — async/threadpool offload).

The endpoints are `async def` but the real work (audio decode, model inference,
synchronous pymongo calls) is blocking. Running it inline blocks the event loop,
so requests serialize and health checks stall under load. These helpers push the
blocking work onto a threadpool so the event loop stays responsive.

* run_blocking  — for I/O and DB calls; uses the shared threadpool (overlap OK).
* run_inference — for GPU/CPU model work; additionally bounded by a semaphore so
                  concurrent requests don't oversubscribe a single GPU.

Per-worker note: the semaphore lives in each worker's event loop, so the GPU
concurrency cap is MAX_CONCURRENT_INFERENCE per worker. Run one worker per GPU.
"""

import asyncio
from typing import Awaitable, Callable, TypeVar

from starlette.concurrency import run_in_threadpool

from config import settings

T = TypeVar("T")

_inference_sem: "asyncio.Semaphore | None" = None


def _get_inference_semaphore() -> asyncio.Semaphore:
    """Lazily create the semaphore so it binds to the running event loop."""
    global _inference_sem
    if _inference_sem is None:
        limit = max(1, settings.MAX_CONCURRENT_INFERENCE)
        _inference_sem = asyncio.Semaphore(limit)
    return _inference_sem


async def run_blocking(func: Callable[..., T], *args, **kwargs) -> T:
    """Run a blocking I/O or DB function off the event loop."""
    return await run_in_threadpool(func, *args, **kwargs)


async def run_inference(func: Callable[..., T], *args, **kwargs) -> T:
    """Run blocking GPU/CPU inference off the event loop, bounded by a semaphore."""
    async with _get_inference_semaphore():
        return await run_in_threadpool(func, *args, **kwargs)
