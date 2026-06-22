import asyncio
import time
from datetime import datetime, timezone

from aiolimiter import AsyncLimiter

from trackllm_website.api import OpenRouterClient, retry_with_exponential_backoff
from trackllm_website.bi.common import EndpointState, query_single
from trackllm_website.config import Endpoint, config
from trackllm_website.storage import Response, ResponseError


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


class _AllTimeout(OpenRouterClient):
    def __init__(self):
        self.calls = 0

    async def query(self, endpoint, prompt, **kw):
        self.calls += 1
        return Response(
            date=datetime.now(timezone.utc),
            endpoint=endpoint,
            prompt=prompt,
            cost=0,
            error=ResponseError(http_code=0, message="Timeout after 15s"),
        )


def test_abandon_after_all_timeouts(tmp_path):
    ep = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))
    state = EndpointState(
        endpoint=ep,
        input_tokens=["a"],
        temperatures=[0.0],
        base_dir=tmp_path,
        rate_limiter=AsyncLimiter(1000, 1),
        concurrency_semaphore=asyncio.Semaphore(1),
        pending_before_new_semaphore=asyncio.Semaphore(1),
        queries_per_token=1,
        abandon_after_timeouts=3,
    )
    client = _AllTimeout()

    async def run():
        for _ in range(5):
            await query_single(client, state, "a", 0.0)

    asyncio.run(run())
    assert state.unresponsive is True
    assert client.calls == 3  # 3 timeouts trip the abandon; calls 4-5 hit the top gate


class _Sequenced(OpenRouterClient):
    """Returns preset responses by call index (the last one repeats)."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    async def query(self, endpoint, prompt, **kw):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return r


def _abandon_state(tmp_path):
    return EndpointState(
        endpoint=Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1)),
        input_tokens=["a"],
        temperatures=[0.0],
        base_dir=tmp_path,
        rate_limiter=AsyncLimiter(1000, 1),
        concurrency_semaphore=asyncio.Semaphore(1),
        pending_before_new_semaphore=asyncio.Semaphore(1),
        queries_per_token=1,
        abandon_after_timeouts=3,
    )


def _timeout(ep):
    return Response(
        date=datetime.now(timezone.utc),
        endpoint=ep,
        prompt="a",
        cost=0,
        error=ResponseError(http_code=0, message="Timeout after 15s"),
    )


def test_stray_error_does_not_disarm_abandon(tmp_path):
    """A non-timeout error mixed in must NOT keep a dead endpoint alive: with zero
    successes and >= N timeouts, the abandon still fires (the robust gate)."""
    state = _abandon_state(tmp_path)
    ep = state.endpoint
    http_500 = Response(
        date=datetime.now(timezone.utc),
        endpoint=ep,
        prompt="a",
        cost=0,
        error=ResponseError(http_code=500, message="server error"),
    )
    client = _Sequenced(
        [http_500, _timeout(ep), _timeout(ep), _timeout(ep), _timeout(ep)]
    )

    async def run():
        for _ in range(6):
            await query_single(client, state, "a", 0.0)

    asyncio.run(run())
    assert (
        state.unresponsive is True
    )  # 3 timeouts + 0 successes => abandon despite the 500
    assert client.calls == 4  # 500, t, t, t(abandon); calls 5-6 hit the top gate


def test_success_prevents_abandon(tmp_path):
    """Any successful query means the endpoint produced something — never abandon,
    however many timeouts follow."""
    state = _abandon_state(tmp_path)
    ep = state.endpoint
    ok = Response(
        date=datetime.now(timezone.utc),
        endpoint=ep,
        prompt="a",
        cost=0.0,
        content="x",
        input_tokens=1,
        output_tokens=1,
    )
    client = _Sequenced([ok, _timeout(ep), _timeout(ep), _timeout(ep), _timeout(ep)])

    async def run():
        for _ in range(6):
            await query_single(client, state, "a", 0.0)

    asyncio.run(run())
    assert state.successful_queries == 1
    assert state.unresponsive is False  # one success disables the abandon gate


def _state_kwargs(tmp_path):
    return dict(
        endpoint=Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1)),
        input_tokens=["a"],
        temperatures=[0.0],
        base_dir=tmp_path,
        rate_limiter=AsyncLimiter(1000, 1),
        concurrency_semaphore=asyncio.Semaphore(1),
        pending_before_new_semaphore=asyncio.Semaphore(1),
        queries_per_token=1,
    )


def test_phase1_state_uses_onboarding_safety_knobs(tmp_path):
    from trackllm_website.bi.phase_1 import Phase1EndpointState

    state = Phase1EndpointState(
        **_state_kwargs(tmp_path),
        max_retries=config.bi.phase_1.max_retries,
        backoff_on_timeout=False,
        abandon_after_timeouts=config.bi.phase_1.abandon_after_timeouts,
    )
    assert state.max_retries == 3
    assert state.backoff_on_timeout is False
    assert state.abandon_after_timeouts == 20


def test_bare_endpoint_state_keeps_safe_defaults(tmp_path):
    state = EndpointState(**_state_kwargs(tmp_path))
    assert state.backoff_on_timeout is True
    assert state.abandon_after_timeouts is None
