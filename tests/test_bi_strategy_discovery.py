"""Strategy discovery vs reasoning-by-default endpoints: a plain probe that
succeeds but bills a reasoning trace must not win outright; effort=none is
tried next and preferred when it truly disables reasoning."""

import asyncio
from datetime import datetime, timezone

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.common import (
    PlainStrategy,
    ReasoningDisabledStrategy,
    discover_strategy,
)
from trackllm_website.config import Endpoint
from trackllm_website.storage import Response, ResponseError

NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


def ep():
    return Endpoint(api="openrouter", model="m/x", provider="prov", cost=(1, 1))


def _resp(endpoint, content="x", reasoning_content=None, reasoning_tokens=0, error=None):
    return Response(
        date=NOW,
        endpoint=endpoint,
        prompt="p",
        cost=1e-6,
        content=content,
        reasoning_content=reasoning_content,
        reasoning_tokens=reasoning_tokens,
        error=error,
    )


class _Scripted(OpenRouterClient):
    """Like test_bi_cost_guard's _Scripted, but records each call's kwargs."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.call_kwargs = []

    async def query(self, endpoint, prompt, **kwargs):
        r = self._responses[min(len(self.call_kwargs), len(self._responses) - 1)]
        self.call_kwargs.append(kwargs)
        return r


def test_plain_reasoning_trace_prefers_effort_none():
    e = ep()
    client = _Scripted(
        _resp(e, content="Paris.", reasoning_content="thinking", reasoning_tokens=178),
        _resp(e, content="The"),
    )
    strategy, errors = asyncio.run(discover_strategy(client, e))
    assert isinstance(strategy, ReasoningDisabledStrategy)
    assert len(client.call_kwargs) == 2
    assert client.call_kwargs[1]["reasoning"] == {"effort": "none"}


def test_plain_reasoning_trace_falls_back_when_effort_none_errors():
    e = ep()
    client = _Scripted(
        _resp(e, content="Paris.", reasoning_tokens=178),
        _resp(e, content=None, error=ResponseError(http_code=400, message="unsupported")),
    )
    strategy, _ = asyncio.run(discover_strategy(client, e))
    assert isinstance(strategy, PlainStrategy)
    assert len(client.call_kwargs) == 2


def test_plain_reasoning_trace_falls_back_when_effort_none_still_reasons():
    e = ep()
    client = _Scripted(
        _resp(e, content="Paris.", reasoning_tokens=178),
        _resp(e, content="The", reasoning_tokens=150),
    )
    strategy, _ = asyncio.run(discover_strategy(client, e))
    assert isinstance(strategy, PlainStrategy)


def test_plain_without_reasoning_stays_single_probe():
    e = ep()
    client = _Scripted(_resp(e, content="The"))
    strategy, _ = asyncio.run(discover_strategy(client, e))
    assert isinstance(strategy, PlainStrategy)
    assert len(client.call_kwargs) == 1
