"""Cost guard: short-circuit strategy discovery when a probe proves an endpoint
will be too expensive (2x its measured price exceeds the per-request ceiling)."""

import asyncio
from datetime import datetime, timezone

from trackllm_website.bi.common import (
    TOO_EXPENSIVE,
    PlainStrategy,
    discover_strategy,
    resolve_strategies,
)
from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.selection import Rule, SelectionPolicy
from trackllm_website.config import Endpoint
from trackllm_website.storage import Response
from trackllm_website.update_endpoints import cache_too_expensive_from_probe
from trackllm_website.bi.vetting import EndpointCache

NOW = datetime(2026, 6, 21, tzinfo=timezone.utc)

# per-request ceiling = max_endpoint_cost / samples_per_month; with the 2x buffer a
# probe is rejected once 2*cost*samples_per_month > max_endpoint_cost.
POLICY = SelectionPolicy(
    budget_per_month=10.0,
    max_endpoint_cost=0.50,
    exclude=[],
    rules=[Rule(name="flagships", kind="models", patterns=["flag/*"], flagship=True)],
)


def _resp(endpoint, cost, content="x"):
    return Response(date=NOW, endpoint=endpoint, prompt="p", cost=cost, content=content)


class _Scripted(OpenRouterClient):
    """Returns a preset response per call (or always the last one). Subclasses the
    real client (without its network setup) to satisfy beartype's type check."""

    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls = 0

    async def query(self, endpoint, prompt, **kwargs):
        r = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return r


def ep(model="m/x", provider="prov"):
    return Endpoint(api="openrouter", model=model, provider=provider, cost=(1, 1))


def test_expensive_probe_short_circuits_on_first_query():
    e = ep()
    client = _Scripted(_resp(e, cost=1.0))  # 2 * 1.0 * 6000 >> 0.50
    strategy, errors = asyncio.run(discover_strategy(client, e, policy=POLICY))
    assert strategy is None
    assert errors[0] == TOO_EXPENSIVE
    assert client.calls == 1  # bailed immediately, no escalation


def test_flagship_is_exempt():
    e = ep(model="flag/big")
    client = _Scripted(_resp(e, cost=1.0))  # expensive, but flagship => exempt
    strategy, errors = asyncio.run(discover_strategy(client, e, policy=POLICY))
    assert isinstance(strategy, PlainStrategy)


def test_cheap_probe_proceeds():
    e = ep()
    client = _Scripted(_resp(e, cost=1e-6))  # 2 * 1e-6 * 6000 = 0.012 < 0.50
    strategy, _ = asyncio.run(discover_strategy(client, e, policy=POLICY))
    assert isinstance(strategy, PlainStrategy)


def test_no_policy_disables_cost_check():
    e = ep()
    client = _Scripted(_resp(e, cost=1.0))  # expensive but policy=None => no check
    strategy, _ = asyncio.run(discover_strategy(client, e))
    assert isinstance(strategy, PlainStrategy)


def test_short_circuits_during_budget_escalation():
    e = ep()
    no_token = _resp(e, cost=1e-6, content=None)  # plain + effort=none yield no token
    expensive_budget = _resp(e, cost=1.0)  # first budget query is expensive
    client = _Scripted(no_token, no_token, expensive_budget)
    strategy, errors = asyncio.run(discover_strategy(client, e, policy=POLICY))
    assert strategy is None
    assert errors[0] == TOO_EXPENSIVE
    assert client.calls == 3  # plain, effort=none, budget=1 -> bail


def test_resolve_strategies_surfaces_too_expensive(monkeypatch):
    # The probe fails (no strategy), so save_strategies is never reached; only the
    # cache read needs stubbing to keep the test off disk.
    monkeypatch.setattr("trackllm_website.bi.common.load_strategies", lambda: {})
    e = ep()
    client = _Scripted(_resp(e, cost=1.0))
    strategies, failed = asyncio.run(resolve_strategies(client, [e], policy=POLICY))
    assert str(e) not in strategies
    assert failed[str(e)][0] == TOO_EXPENSIVE


def test_cache_too_expensive_from_probe_routes_only_marked():
    a, b = ep(model="a/x"), ep(model="b/x")
    failed = {str(a): [TOO_EXPENSIVE, "detail"], str(b): ["plain: 500 boom"]}
    cache = EndpointCache(liars=[], too_expensive=[], bad_temperature=[])
    cache_too_expensive_from_probe(failed, [a, b], cache)
    assert cache.bucket_of(a) == "too_expensive"
    assert cache.bucket_of(b) is None
