import asyncio
from datetime import datetime, timezone

from trackllm_website.config import Endpoint, config
from trackllm_website.logprob_discovery import (
    LOGPROB_LADDER,
    query_discovering_max_logprobs,
    query_endpoint,
)
from trackllm_website.storage import Response, ResponseError


def _ep(provider: str = "someprovider") -> Endpoint:
    return Endpoint(api="openrouter", model="m/x", provider=provider, cost=(1, 1))


def _response(endpoint: Endpoint, prompt: str, error: ResponseError | None) -> Response:
    return Response(
        date=datetime.now(tz=timezone.utc),
        endpoint=endpoint,
        prompt=prompt,
        logprobs=None,
        cost=0.0,
        error=error,
    )


class _CappedApi:
    """Fake query fn: rejects top_logprobs above `cap` with a 400, like Novita."""

    def __init__(self, cap: int, http_code: int = 400):
        self.cap = cap
        self.http_code = http_code
        self.calls: list[tuple[str, int]] = []

    async def query(self, endpoint: Endpoint, prompt: str) -> Response:
        n = endpoint.get_max_logprobs(cfg=config)
        self.calls.append((prompt, n))
        if n > self.cap:
            return _response(
                endpoint,
                prompt,
                ResponseError(http_code=self.http_code, message="nope"),
            )
        return _response(endpoint, prompt, None)


def test_success_at_configured_depth_probes_once():
    api = _CappedApi(cap=20)
    ep = _ep()
    r = asyncio.run(query_discovering_max_logprobs(api.query, ep, "x"))
    assert r.error is None
    assert [n for _, n in api.calls] == [20]
    assert ep.max_logprobs is None


def test_400_walks_ladder_down_to_working_cap():
    api = _CappedApi(cap=5)
    ep = _ep()
    r = asyncio.run(query_discovering_max_logprobs(api.query, ep, "x"))
    assert r.error is None
    assert [n for _, n in api.calls] == [20, 10, 8, 5]
    assert ep.max_logprobs == 5


def test_ladder_starts_below_provider_configured_cap():
    # fireworks is capped at 5 in config.toml: only lower rungs are probed.
    api = _CappedApi(cap=2)
    ep = _ep(provider="fireworks")
    r = asyncio.run(query_discovering_max_logprobs(api.query, ep, "x"))
    assert r.error is None
    assert [n for _, n in api.calls] == [5, 3, 2]
    assert ep.max_logprobs == 2


def test_unrelated_400_exhausts_ladder_and_restores_depth():
    api = _CappedApi(cap=0)
    ep = _ep()
    r = asyncio.run(query_discovering_max_logprobs(api.query, ep, "x"))
    assert r.error is not None and r.error.http_code == 400
    assert [n for _, n in api.calls] == [20, *LOGPROB_LADDER[1:]]
    assert ep.max_logprobs is None


def test_non_400_error_does_not_trigger_discovery():
    api = _CappedApi(cap=0, http_code=429)
    ep = _ep()
    r = asyncio.run(query_discovering_max_logprobs(api.query, ep, "x"))
    assert r.error is not None and r.error.http_code == 429
    assert [n for _, n in api.calls] == [20]
    assert ep.max_logprobs is None


def test_query_endpoint_serializes_discovery_before_remaining_prompts():
    api = _CappedApi(cap=5)
    ep = _ep()
    responses = asyncio.run(query_endpoint(api.query, ep, ["x", "Hi"]))
    assert len(responses) == 2
    assert all(r.error is None for r in responses)
    # "Hi" must run after discovery, at the discovered cap.
    assert api.calls == [("x", 20), ("x", 10), ("x", 8), ("x", 5), ("Hi", 5)]
