import asyncio
import time

from trackllm_website.api import retry_with_exponential_backoff
from trackllm_website.config import config


def test_onboarding_safety_knobs_present():
    assert config.bi.phase_1.max_retries == 3  # 4 attempts (onboarding)
    assert config.bi.phase_1.abandon_after_timeouts == 20
    assert config.bi.reinit.onboard_timeout_seconds == 10800  # 3h
    assert config.bi.reinit.onboard_concurrency == 40


def test_no_backoff_on_timeout_when_disabled():
    calls = {"n": 0}

    async def always_timeout():
        calls["n"] += 1
        raise asyncio.TimeoutError()

    async def run():
        t0 = time.monotonic()
        try:
            await retry_with_exponential_backoff(
                always_timeout, max_retries=3, backoff_on_timeout=False
            )
        except asyncio.TimeoutError:
            pass
        return time.monotonic() - t0, calls["n"]

    elapsed, n = asyncio.run(run())
    assert n == 4  # 1 + 3 retries
    assert elapsed < 0.5  # zero backoff sleeps
