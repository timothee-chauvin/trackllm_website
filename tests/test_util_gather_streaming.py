import asyncio

import pytest

from trackllm_website.util import gather_with_concurrency_streaming

TIMEOUT = 5


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=TIMEOUT))


async def _ok(i, delay=0):
    await asyncio.sleep(delay)
    return i


async def _boom():
    raise ValueError("boom")


def test_streams_all_results_when_nothing_raises():
    async def run():
        coros = [_ok(i, delay=0.001 * (5 - i)) for i in range(5)]
        return [r async for r in gather_with_concurrency_streaming(2, *coros)]

    results = _run(run())
    assert sorted(results) == list(range(5))


def test_worker_exception_raises_instead_of_hanging():
    # Before the fix, a raising coroutine killed its worker before queue.put,
    # so the consumer loop waited on queue.get() forever.
    async def run():
        gen = gather_with_concurrency_streaming(2, _ok(1), _boom(), _ok(2))
        with pytest.raises(ValueError, match="boom"):
            async for _ in gen:
                pass

    _run(run())


def test_results_before_exception_still_stream():
    async def run():
        seen = []
        gen = gather_with_concurrency_streaming(1, _ok(0), _boom(), _ok(2))
        with pytest.raises(ValueError, match="boom"):
            async for r in gen:
                seen.append(r)
        return seen

    assert _run(run()) == [0]
