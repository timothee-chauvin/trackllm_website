"""Universal per-query cost guard: any successful query billed above
api.max_cost_per_query raises QueryTooExpensive, and every activity retires the
endpoint immediately (August 2026: reasoning-token leaks billed 10-300x normal)."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import orjson
import pytest

from trackllm_website.api import OpenRouterClient, QueryTooExpensive
from trackllm_website.bi import monitor as monitor_mod
from trackllm_website.bi.common import TOO_EXPENSIVE, resolve_strategies
from trackllm_website.bi.migrate_state import migrate_endpoint
from trackllm_website.bi.monitor import run_endpoint
from trackllm_website.bi.vetting import EndpointCache, VetResult, vet_endpoint
from trackllm_website.bi.common import PlainStrategy
from trackllm_website.config import Endpoint, config
from trackllm_website.logprob_discovery import query_endpoint
from trackllm_website.spend import cumulative_by_kind, record_query, track
from trackllm_website.storage import Response, ResponseError
from trackllm_website.update_endpoints import (
    route_vet_result,
    update_endpoints_bi_lifecycle,
)

NOW = datetime(2026, 2, 15, tzinfo=timezone.utc)
GUARD = 1e-4
FIXTURES = Path("tests/fixtures/phase_2")


def ep(model="m/x", provider="p"):
    return Endpoint(api="openrouter", model=model, provider=provider, cost=(1, 1))


def resp(cost, error=None, content="y"):
    return Response(
        date=NOW,
        endpoint=ep(),
        prompt="x",
        content=content,
        logprobs=None,
        cost=cost,
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        reasoning_content=None,
        generation_id="g",
        error=error,
    )


@pytest.fixture
def guard(monkeypatch):
    monkeypatch.setattr(config.api, "max_cost_per_query", GUARD)


# --- api.query ---


def _run_query(monkeypatch, response):
    async def fake_make_request(self, *a, **k):
        return response

    monkeypatch.setattr(OpenRouterClient, "_make_request", fake_make_request)

    async def go():
        client = OpenRouterClient()
        try:
            return await client.query(ep(), "x")
        finally:
            await client.close()

    return asyncio.run(go())


def test_query_raises_above_guard_after_recording_spend(monkeypatch, guard):
    async def fake_make_request(self, *a, **k):
        return resp(0.002)

    monkeypatch.setattr(OpenRouterClient, "_make_request", fake_make_request)

    async def go():
        client = OpenRouterClient()
        try:
            with track() as s:
                with pytest.raises(QueryTooExpensive) as exc:
                    await client.query(ep(), "x")
            return s, exc.value
        finally:
            await client.close()

    s, e = asyncio.run(go())
    assert s.cost == 0.002 and s.n_queries == 1  # ledger sees the money first
    assert e.cost == 0.002
    assert e.endpoint == ep()


def test_query_passes_at_or_below_guard(monkeypatch, guard):
    r = _run_query(monkeypatch, resp(GUARD))
    assert r.error is None


def test_query_never_raises_on_error_responses(monkeypatch, guard):
    r = _run_query(
        monkeypatch, resp(0.0, error=ResponseError(http_code=500, message="boom"))
    )
    assert r.error is not None


# --- BI monitor ---


def _state_from_fixture(slug):
    results = orjson.loads((FIXTURES / slug / "data.json").read_bytes())
    state = migrate_endpoint(ep(), results)
    state.status = "monitoring"
    state.retired = None
    epoch = state.epochs[0]
    epoch.end = None
    epoch.end_reason = None
    return state, results


def _wire_monitor(monkeypatch, tmp_path, results):
    monkeypatch.setattr(type(config), "spend_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        monitor_mod, "get_output_path", lambda e, ym: tmp_path / "m.json"
    )
    monkeypatch.setattr(monitor_mod, "load_phase2_results", lambda d: results)


def test_monitor_sampling_guard_retires_immediately(monkeypatch, tmp_path, guard):
    state, results = _state_from_fixture("openai2fgpt-4o-mini23azure")

    async def exploding_sample(*a, **k):
        record_query(0.3, False)
        raise QueryTooExpensive(ep(), 0.002, resp(0.002))

    _wire_monitor(monkeypatch, tmp_path, results)
    monkeypatch.setattr(monitor_mod, "sample_prompts", exploding_sample)

    rows = []

    async def go():
        async with OpenRouterClient() as client:
            await run_endpoint(client, PlainStrategy(), state, NOW, event_rows=rows)

    asyncio.run(go())

    assert state.status == "retired"
    assert state.retired.reason == "too_expensive"
    assert state.epochs[-1].end == NOW
    assert state.epochs[-1].end_reason == "too_expensive"
    assert cumulative_by_kind(tmp_path)["monitor"] == 0.3  # spend line still written
    assert [r.event for r in rows] == ["retired_too_expensive"]
    assert rows[0].spent == 0.3


def test_monitor_reinit_guard_retires_immediately(monkeypatch, tmp_path, guard):
    state, results = _state_from_fixture(
        "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8"
    )
    epoch = state.current_epoch
    daily = {bi: [(NOW.isoformat(), "tok")] for bi in epoch.border_inputs}

    async def fake_sample(*a, **k):
        record_query(0.01, False)
        return daily, 0

    async def exploding_reinit(*a, **k):
        record_query(0.5, False)
        raise QueryTooExpensive(ep(), 0.002, resp(0.002))

    _wire_monitor(monkeypatch, tmp_path, results)
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample)
    monkeypatch.setattr(monitor_mod, "reinit", exploding_reinit)

    rows = []

    async def go():
        async with OpenRouterClient() as client:
            await run_endpoint(client, PlainStrategy(), state, NOW, event_rows=rows)

    asyncio.run(go())

    assert state.status == "retired"
    assert state.retired.reason == "too_expensive"
    assert state.epochs[-1].end_reason == "too_expensive"
    assert cumulative_by_kind(tmp_path)["reinit"] == 0.5
    assert "retired_too_expensive" in [r.event for r in rows]


# --- BI onboarding ---


class _FakeClient(OpenRouterClient):
    def __init__(self, *a, **k):
        pass

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


def _patch_lifecycle(monkeypatch, tmp_path, reinit):
    monkeypatch.setattr(config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        type(config), "spend_dir", property(lambda self: tmp_path / "spend")
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.OpenRouterClient", _FakeClient
    )

    async def fake_resolve(client, endpoints, policy=None, probe_spend=None):
        return {str(e): None for e in endpoints}, {}

    monkeypatch.setattr(
        "trackllm_website.update_endpoints.resolve_strategies", fake_resolve
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.select_monitoring_targets",
        lambda candidates, policy, popular: (candidates, {e: "r" for e in candidates}),
    )
    monkeypatch.setattr("trackllm_website.update_endpoints.reinit", reinit)
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.fetch_popular_models_safe", lambda n: []
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.ENDPOINTS_CACHE_BI_PATH",
        tmp_path / "endpoints_cache_bi.yaml",
    )


def test_onboarding_guard_retires_and_caches(monkeypatch, tmp_path, guard):
    e = ep("m/pricey")

    async def exploding_reinit(client, strategy, endpoint, old_bis, now):
        raise QueryTooExpensive(endpoint, 0.002, resp(0.002))

    _patch_lifecycle(monkeypatch, tmp_path, exploding_reinit)
    report = asyncio.run(update_endpoints_bi_lifecycle([e]))

    assert [r.outcome for r in report.rows] == ["too_expensive"]
    cache = EndpointCache.load(tmp_path / "endpoints_cache_bi.yaml")
    assert cache.bucket_of(e) == "too_expensive"
    from trackllm_website.bi.state import load_all_states

    states = load_all_states(config.bi.state_dir)
    (state,) = states.values()
    assert state.status == "retired"
    assert state.retired.reason == "too_expensive"


# --- vetting ---


class _VetClient(OpenRouterClient):
    def __init__(self, response=None, raises=None, actual=None):
        self._response, self._raises, self._actual = response, raises, actual
        self.session = None

    async def query(self, *a, **k):
        if self._raises:
            raise self._raises
        return self._response

    async def get_generation_cost(self, generation_id, session=None):
        return self._actual


def test_vet_guard_exception_buckets_too_expensive(guard):
    client = _VetClient(raises=QueryTooExpensive(ep(), 0.002, resp(0.002)))
    res = asyncio.run(vet_endpoint(client, ep(), PlainStrategy()))
    assert res.bucket == "too_expensive"


def test_vet_billed_cost_above_guard_buckets_too_expensive(guard):
    # expected 0 skips the liar check (advertised price stale/zero); the billed
    # backstop must still catch it
    client = _VetClient(response=resp(0.0), actual=0.002)
    res = asyncio.run(vet_endpoint(client, ep(), PlainStrategy()))
    assert res.bucket == "too_expensive"


def test_route_vet_result_caches_too_expensive(guard):
    from trackllm_website.bi.selection import SelectionPolicy

    cache = EndpointCache(liars=[], too_expensive=[], bad_temperature=[])
    policy = SelectionPolicy(
        budget_per_month=10.0, max_endpoint_cost=0.5, exclude=[], rules=[]
    )
    out = route_vet_result(
        VetResult(bucket="too_expensive"),
        ep(),
        cache,
        policy,
        threshold=3,
        prior_good=False,
    )
    assert out is None
    assert cache.bucket_of(ep()) == "too_expensive"


# --- strategy probing ---


def test_probe_guard_reports_too_expensive_failure(monkeypatch, guard):
    async def exploding_discover(client, e, policy=None):
        raise QueryTooExpensive(e, 0.002, resp(0.002))

    monkeypatch.setattr(
        "trackllm_website.bi.common.discover_strategy", exploding_discover
    )
    monkeypatch.setattr("trackllm_website.bi.common.load_strategies", lambda: {})
    monkeypatch.setattr("trackllm_website.bi.common.save_strategies", lambda d: None)

    client = _VetClient()
    strategies, failed = asyncio.run(resolve_strategies(client, [ep()]))
    assert strategies == {}
    assert failed[str(ep())][0] == TOO_EXPENSIVE


# --- review findings: spend accounting on the trip path ---


def test_guard_exception_carries_the_response(monkeypatch, guard):
    async def fake_make_request(self, *a, **k):
        return resp(0.002)

    monkeypatch.setattr(OpenRouterClient, "_make_request", fake_make_request)

    async def go():
        client = OpenRouterClient()
        try:
            with pytest.raises(QueryTooExpensive) as exc:
                await client.query(ep(), "x")
            return exc.value
        finally:
            await client.close()

    e = asyncio.run(go())
    assert e.response.cost == 0.002


def test_query_endpoint_keeps_tripped_and_completed_responses(guard):
    # The tripped query's Response must survive: LT's ledger is built from the
    # returned responses, and the daily prune reads that ledger.
    async def query(endpoint, prompt):
        if prompt == "b":
            raise QueryTooExpensive(endpoint, 0.002, resp(0.002))
        return resp(1e-5)

    out = asyncio.run(query_endpoint(query, ep(), ["a", "b", "c"]))
    assert sorted(r.cost for r in out) == [1e-5, 1e-5, 0.002]


def test_sampling_stops_issuing_queries_after_guard_trip(monkeypatch, guard):
    from trackllm_website.bi.sampling import sample_prompts

    monkeypatch.setattr(config.bi.phase_2, "max_concurrent_requests_per_endpoint", 1)
    monkeypatch.setattr(config.bi.phase_2, "request_delay_seconds", 0.0)
    calls = []

    class _Trip(OpenRouterClient):
        def __init__(self):
            pass

        async def query(self, endpoint, prompt, **kwargs):
            calls.append(prompt)
            if len(calls) == 2:
                raise QueryTooExpensive(endpoint, 0.002, resp(0.002))
            return resp(1e-5)

    with pytest.raises(QueryTooExpensive):
        asyncio.run(
            sample_prompts(
                _Trip(), ep(), PlainStrategy(), ["a", "b"], 3, temperature=0.0
            )
        )
    assert len(calls) == 2  # no further queries after the trip


def test_probe_spend_recorded_when_probe_trips(monkeypatch, guard):
    async def exploding_discover(client, e, policy=None):
        record_query(0.4, False)
        raise QueryTooExpensive(e, 0.002, resp(0.002))

    monkeypatch.setattr(
        "trackllm_website.bi.common.discover_strategy", exploding_discover
    )
    monkeypatch.setattr("trackllm_website.bi.common.load_strategies", lambda: {})
    monkeypatch.setattr("trackllm_website.bi.common.save_strategies", lambda d: None)

    probe_spend = {}
    client = _VetClient()
    _, failed = asyncio.run(resolve_strategies(client, [ep()], probe_spend=probe_spend))
    assert failed[str(ep())][0] == TOO_EXPENSIVE
    assert probe_spend[str(ep())].cost == 0.4


def test_guard_trips_on_billed_error_responses(monkeypatch, guard):
    # "No logprobs returned" responses are billed (tokens generated) but carry a
    # synthetic error; real money over the guard must still trip it.
    async def fake_make_request(self, *a, **k):
        return resp(
            0.002, error=ResponseError(http_code=200, message="No logprobs returned")
        )

    monkeypatch.setattr(OpenRouterClient, "_make_request", fake_make_request)

    async def go():
        client = OpenRouterClient()
        try:
            with pytest.raises(QueryTooExpensive):
                await client.query(ep(), "x")
        finally:
            await client.close()

    asyncio.run(go())


def test_query_endpoint_stops_after_first_prompt_trip(guard):
    calls = []

    async def query(endpoint, prompt):
        calls.append(prompt)
        raise QueryTooExpensive(endpoint, 0.002, resp(0.002))

    out = asyncio.run(query_endpoint(query, ep(), ["a", "b", "c"]))
    assert calls == ["a"]  # rest never issued once the guard tripped
    assert [r.cost for r in out] == [0.002]


def test_monitor_probe_trip_retires_endpoint(monkeypatch, tmp_path, guard):
    from trackllm_website.bi.monitor import monitor
    from trackllm_website.bi.state import EndpointBIState, Epoch
    from trackllm_website.spend import Spend

    state = EndpointBIState(
        endpoint=ep(),
        status="monitoring",
        epochs=[Epoch(start=NOW, border_inputs=["a"], reference={"a": []})],
    )
    monkeypatch.setattr(type(config), "spend_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(monitor_mod, "load_all_states", lambda d: {state.slug: state})
    monkeypatch.setattr(monitor_mod, "OpenRouterClient", _FakeClient)

    async def fake_resolve(client, endpoints, policy=None, probe_spend=None):
        probe_spend[str(ep())] = Spend(cost=0.002, n_queries=1, n_errors=0)
        return {}, {str(ep()): [TOO_EXPENSIVE, "probe billed $0.002000/query"]}

    monkeypatch.setattr(monitor_mod, "resolve_strategies", fake_resolve)

    report = asyncio.run(monitor())

    assert state.status == "retired"
    assert state.retired.reason == "too_expensive"
    assert cumulative_by_kind(tmp_path)["monitor"] == 0.002
    assert [r.event for r in report.rows] == ["retired_too_expensive"]
