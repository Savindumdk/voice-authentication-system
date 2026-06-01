"""Tests for the threadpool/inference offload helpers."""

import asyncio
import threading
import time

import concurrency


def _reset_semaphore():
    concurrency._inference_sem = None


async def test_run_blocking_returns_result():
    _reset_semaphore()
    assert await concurrency.run_blocking(lambda x: x * 2, 21) == 42


async def test_run_inference_returns_result():
    _reset_semaphore()
    assert await concurrency.run_inference(lambda a, b: a + b, 2, 3) == 5


async def test_run_inference_propagates_exceptions():
    _reset_semaphore()

    def boom():
        raise ValueError("nope")

    try:
        await concurrency.run_inference(boom)
        assert False, "exception should propagate"
    except ValueError:
        pass


async def test_run_inference_bounds_concurrency(monkeypatch):
    monkeypatch.setattr(concurrency.settings, "MAX_CONCURRENT_INFERENCE", 2)
    _reset_semaphore()

    lock = threading.Lock()
    state = {"current": 0, "max": 0}

    def work():
        with lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        time.sleep(0.05)
        with lock:
            state["current"] -= 1
        return True

    await asyncio.gather(*[concurrency.run_inference(work) for _ in range(8)])
    assert state["max"] <= 2, f"observed {state['max']} concurrent (limit 2)"
