"""LT reuse of the BI reasoning-strategy cache: reasoning models get effort=none
instead of billing a trace per 1-token query, priority provider variants are
excluded as price-doubled duplicates, and the ledger prune uses a trailing
window so rescued endpoints return within days."""

import asyncio
from datetime import datetime, timezone

import orjson
import pytest

from trackllm_website.bi.common import (
    PlainStrategy,
    ReasoningBudgetStrategy,
    ReasoningDisabledStrategy,
    cached_lt_query_args,
    lt_query_args,
)
from trackllm_website.config import Endpoint, config
from trackllm_website.main import lt_query_fn
from trackllm_website.update_endpoints import (
    exclude_provider_segments,
    lt_cost_per_query,
    test_endpoints_logprobs as vet_endpoints_logprobs,
)

NOW = datetime(2026, 2, 15, tzinfo=timezone.utc)


def ep(model="m/x", provider="p"):
    return Endpoint(api="openrouter", model=model, provider=provider, cost=(1, 1))


# --- strategy -> LT query args ---


def test_lt_query_args_keeps_only_reasoning_disabled():
    strategies = {
        "a": PlainStrategy(),
        "b": ReasoningDisabledStrategy(),
        "c": ReasoningBudgetStrategy(budget=4),
    }
    assert lt_query_args(strategies) == {"b": {"reasoning": {"effort": "none"}}}


def test_cached_lt_query_args_reads_committed_cache(tmp_path, monkeypatch):
    cache = {
        "openrouter#m/plain#p": None,
        "openrouter#m/effort#p": {"effort": "none"},
        "openrouter#m/budget#p": {"max_tokens": 4},
        "openrouter#m/skip#p": {"skip": "hidden reasoning"},
    }
    path = tmp_path / "strategies.json"
    path.write_bytes(orjson.dumps(cache))
    monkeypatch.setattr(config.bi.probe, "strategies_path", path)
    assert cached_lt_query_args() == {
        "openrouter#m/effort#p": {"reasoning": {"effort": "none"}}
    }


# --- main.py applies args per endpoint ---


def _capture_query():
    calls = []

    async def query(endpoint, prompt, **kwargs):
        calls.append((endpoint, prompt, kwargs))
        return "resp"

    return query, calls


def test_lt_query_fn_applies_cached_args():
    query, calls = _capture_query()
    e = ep("m/effort")
    fn = lt_query_fn(query, e, {str(e): {"reasoning": {"effort": "none"}}})
    asyncio.run(fn(e, "Hi"))
    assert calls == [(e, "Hi", {"reasoning": {"effort": "none"}})]


def test_lt_query_fn_passthrough_without_entry():
    query, calls = _capture_query()
    e = ep("m/plain")
    fn = lt_query_fn(query, e, {})
    assert fn is query
    asyncio.run(fn(e, "Hi"))
    assert calls == [(e, "Hi", {})]


# --- priority variant exclusion ---


def test_exclude_provider_segments():
    fleet = [
        ep(provider="xai"),
        ep(provider="xai/priority"),
        ep(provider="xai/zdr"),
        ep(provider="xai/zdr/priority"),
        ep(provider="alibaba/fp8"),
    ]
    kept = exclude_provider_segments(fleet, ["priority"])
    assert [e.provider for e in kept] == ["xai", "xai/zdr", "alibaba/fp8"]


def test_exclude_provider_segments_matches_segments_not_substrings():
    fleet = [ep(provider="priorityai"), ep(provider="a/priority-fast")]
    assert exclude_provider_segments(fleet, ["priority"]) == fleet


def test_lt_exclude_provider_segments_configured():
    assert config.api.lt_exclude_provider_segments == ["priority"]


# --- LT vetting under the same query shape ---


class _FakeClient:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, endpoint, prompt, **kwargs):
        from trackllm_website.storage import Response, ResponseError

        _FakeClient.calls.append((endpoint, kwargs))
        return Response(
            date=NOW,
            endpoint=endpoint,
            prompt=prompt,
            cost=1e-6,
            error=ResponseError(http_code=400, message="nope"),
        )


def test_lt_vetting_applies_strategy_args(monkeypatch):
    import trackllm_website.update_endpoints as ue

    monkeypatch.setattr(ue, "OpenRouterClient", _FakeClient)
    _FakeClient.calls = []
    e_effort, e_plain = ep("m/effort"), ep("m/plain")
    args = {str(e_effort): {"reasoning": {"effort": "none"}}}
    valid, failures = asyncio.run(vet_endpoints_logprobs([e_effort, e_plain], args))
    assert valid == []
    assert set(failures) == {e_effort, e_plain}
    by_endpoint = dict(_FakeClient.calls)
    assert by_endpoint[e_effort] == {"reasoning": {"effort": "none"}}
    assert by_endpoint[e_plain] == {}


# --- trailing prune window ---


def _write_ledger(spend_dir, slug, month, entries):
    d = spend_dir / slug
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{month}.jsonl", "ab") as f:
        for ts, kind, cost, nq, ne in entries:
            f.write(
                orjson.dumps(
                    {
                        "timestamp": ts,
                        "kind": kind,
                        "cost": cost,
                        "n_queries": nq,
                        "n_errors": ne,
                    }
                )
                + b"\n"
            )


def test_lt_prune_window_days_configured():
    assert config.api.lt_prune_window_days == 7


def test_lt_cost_ignores_entries_outside_window(tmp_path):
    # Expensive history outside the 7-day window, cheap inside: not pruned.
    _write_ledger(
        tmp_path,
        "slug",
        "2026-02",
        [
            ("2026-02-01T00:00:00Z", "lt", 5.0, 50, 0),
            ("2026-02-12T00:00:00Z", "lt", 0.001, 50, 0),
        ],
    )
    assert lt_cost_per_query(tmp_path, "slug", NOW) == pytest.approx(0.001 / 50)


def test_lt_cost_catches_recent_price_jump(tmp_path):
    # Cheap history outside the window, expensive inside: pruned fast.
    _write_ledger(
        tmp_path,
        "slug",
        "2026-01",
        [("2026-01-20T00:00:00Z", "lt", 0.001, 500, 0)],
    )
    _write_ledger(
        tmp_path,
        "slug",
        "2026-02",
        [("2026-02-12T00:00:00Z", "lt", 5.0, 50, 0)],
    )
    assert lt_cost_per_query(tmp_path, "slug", NOW) == pytest.approx(0.1)


def test_lt_cost_window_spans_month_boundary(tmp_path):
    early_march = datetime(2026, 3, 2, tzinfo=timezone.utc)
    _write_ledger(
        tmp_path,
        "slug",
        "2026-02",
        [("2026-02-27T00:00:00Z", "lt", 1.0, 25, 0)],
    )
    _write_ledger(
        tmp_path,
        "slug",
        "2026-03",
        [("2026-03-01T00:00:00Z", "lt", 1.0, 25, 0)],
    )
    assert lt_cost_per_query(tmp_path, "slug", early_march) == pytest.approx(0.04)


def test_lt_cost_needs_enough_queries_in_window(tmp_path):
    _write_ledger(
        tmp_path,
        "slug",
        "2026-02",
        [
            ("2026-02-01T00:00:00Z", "lt", 0.5, 100, 0),
            ("2026-02-12T00:00:00Z", "lt", 0.5, 19, 0),
        ],
    )
    assert lt_cost_per_query(tmp_path, "slug", NOW) is None


# --- update_endpoints_lt wiring ---


def test_update_endpoints_lt_resolves_strategies_and_filters_priority(
    tmp_path, monkeypatch
):
    """The daily LT update excludes priority variants (kept endpoints included),
    resolves strategies for the whole fleet, and vets new endpoints under their
    strategy's query shape."""
    import yaml

    import trackllm_website.update_endpoints as ue

    kept = ep("m/kept", "prov")
    kept_priority = ep("m/kept", "prov/priority")
    new = ep("m/new", "prov")
    new_priority = ep("m/new", "prov/priority")

    monkeypatch.setattr(config, "endpoints_lt", [kept, kept_priority])
    monkeypatch.setattr(config, "endpoints_yaml_path_lt", tmp_path / "lt.yaml")
    monkeypatch.setattr(ue, "ENDPOINTS_CACHE_LT_PATH", tmp_path / "cache.yaml")
    monkeypatch.setattr(config, "data_dir", tmp_path)

    class _Storage:
        def __init__(self, *a, **k):
            pass

        def is_stalled(self, e):
            return False

    monkeypatch.setattr(ue, "ResultsStorage", _Storage)

    async def fake_get_endpoints(**kwargs):
        return [kept, kept_priority, new, new_priority]

    monkeypatch.setattr(ue, "get_endpoints", fake_get_endpoints)

    resolved_fleets = []

    async def fake_resolve_strategies(client, endpoints):
        resolved_fleets.append(list(endpoints))
        return {str(kept): ReasoningDisabledStrategy(), str(new): PlainStrategy()}, {}

    monkeypatch.setattr(ue, "resolve_strategies", fake_resolve_strategies)
    monkeypatch.setattr(ue, "OpenRouterClient", _FakeClient)

    vetted = []

    async def fake_test_endpoints_logprobs(endpoints, query_args):
        vetted.append((sorted(endpoints, key=str), query_args))
        return list(endpoints), {}

    monkeypatch.setattr(ue, "test_endpoints_logprobs", fake_test_endpoints_logprobs)

    asyncio.run(ue.update_endpoints_lt())

    # Priority variants never reach strategy resolution, vetting, or the yaml.
    assert resolved_fleets == [[kept, new]]
    assert vetted == [([new], {str(kept): {"reasoning": {"effort": "none"}}})]
    with open(tmp_path / "lt.yaml") as f:
        written = yaml.safe_load(f)
    assert [(e["model"], e["provider"]) for e in written["endpoints_lt"]] == [
        ("m/kept", "prov"),
        ("m/new", "prov"),
    ]
