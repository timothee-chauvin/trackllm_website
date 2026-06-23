"""Tests: per-endpoint strategy-probe spend is folded into the activity ledger line."""

import asyncio
import orjson
from datetime import datetime, timezone
from pathlib import Path

import trackllm_website.update_endpoints as ue
from trackllm_website.bi import monitor as monitor_mod
from trackllm_website.bi.common import PlainStrategy
from trackllm_website.bi.migrate_state import migrate_endpoint
from trackllm_website.bi.monitor import run_endpoint
from trackllm_website.bi.reinit import ReinitResult
from trackllm_website.bi.state import Epoch
from trackllm_website.bi.vetting import VetResult
from trackllm_website.config import Endpoint, config
from trackllm_website.spend import Spend, cumulative_by_kind, record_query

NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)
FIXTURES = Path("tests/fixtures/phase_2")


def ep(model):
    return Endpoint(api="openrouter", model=model, provider="p", cost=(1, 1))


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# --- Unit test ---


def test_spend_merge():
    import pytest

    a = Spend(cost=0.10, n_queries=2, n_errors=0)
    b = Spend(cost=0.05, n_queries=3, n_errors=1)
    a.merge(b)
    assert a.cost == pytest.approx(0.15)
    assert a.n_queries == 5
    assert a.n_errors == 1


def test_spend_merge_default():
    a = Spend(cost=0.10, n_queries=2, n_errors=0)
    a.merge(Spend())
    assert a.cost == 0.10
    assert a.n_queries == 2
    assert a.n_errors == 0


# --- Integration: vetting folds probe cost ---


def test_vetting_folds_probe_cost(monkeypatch, tmp_path):
    """Vetting ledger line cost == probe cost + vet cost."""
    endpoint = ep("m/vet_probe")
    probe_cost = 0.02
    vet_cost = 0.07

    async def fake_vet_endpoint(client, ep_arg, strategy):
        record_query(vet_cost, is_error=False)
        return VetResult(bucket="candidate", cost_per_request=vet_cost)

    # resolve_strategies stub: populates probe_spend AND returns the strategy
    async def fake_resolve_strategies(client, endpoints, policy=None, probe_spend=None):
        if probe_spend is not None:
            for e in endpoints:
                probe_spend[str(e)] = Spend(cost=probe_cost, n_queries=2)
        return {str(e): PlainStrategy() for e in endpoints}, {}

    monkeypatch.setattr(
        type(config), "spend_dir", property(lambda self: tmp_path / "spend")
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.OpenRouterClient", _FakeClient
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.resolve_strategies", fake_resolve_strategies
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.vet_endpoint", fake_vet_endpoint
    )

    async def fake_get_endpoints(**_):
        return [endpoint]

    monkeypatch.setattr(
        "trackllm_website.update_endpoints.get_endpoints", fake_get_endpoints
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.save_endpoints_bi", lambda endpoints: None
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.ENDPOINTS_CACHE_BI_PATH",
        tmp_path / "endpoints_cache_bi.yaml",
    )
    monkeypatch.setattr(config, "endpoints_bi", [])
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.load_policy", lambda path: None
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.exceeds_ceiling",
        lambda cost, model, provider, policy: False,
    )

    asyncio.run(ue.update_endpoints_bi())

    spend_dir = tmp_path / "spend"
    lines = _read_jsonl(spend_dir)
    vetting_lines = [e for e in lines if e["kind"] == "vetting"]
    assert vetting_lines, "Expected a vetting ledger line"
    assert abs(vetting_lines[0]["cost"] - (probe_cost + vet_cost)) < 1e-9


# --- Integration: onboard folds probe cost ---


def test_onboard_folds_probe_cost(monkeypatch, tmp_path):
    """Onboard ledger line cost == probe cost + reinit cost."""
    endpoint = ep("m/onboard_probe")
    probe_cost = 0.03
    reinit_cost = 0.09

    async def fake_reinit(client, strategy, ep_arg, old_bis, now):
        record_query(reinit_cost, is_error=False)
        return ReinitResult(
            epoch=Epoch(start=now, border_inputs=["a"] * 10, reference={}),
            reason="ok",
        )

    async def fake_resolve_strategies(client, endpoints, policy=None, probe_spend=None):
        if probe_spend is not None:
            for e in endpoints:
                probe_spend[str(e)] = Spend(cost=probe_cost, n_queries=1)
        return {str(e): PlainStrategy() for e in endpoints}, {}

    monkeypatch.setattr(config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        type(config), "spend_dir", property(lambda self: tmp_path / "spend")
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.OpenRouterClient", _FakeClient
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.resolve_strategies", fake_resolve_strategies
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.select_monitoring_targets",
        lambda candidates, policy, popular_models: (list(candidates), {}),
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.fetch_popular_models_safe", lambda top_n: []
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.ENDPOINTS_CACHE_BI_PATH",
        tmp_path / "endpoints_cache_bi.yaml",
    )
    monkeypatch.setattr("trackllm_website.update_endpoints.reinit", fake_reinit)

    asyncio.run(ue.update_endpoints_bi_lifecycle([endpoint]))

    lines = _read_jsonl(tmp_path / "spend")
    onboard_lines = [e for e in lines if e["kind"] == "onboard"]
    assert onboard_lines, "Expected an onboard ledger line"
    assert abs(onboard_lines[0]["cost"] - (probe_cost + reinit_cost)) < 1e-9


# --- Integration: strategy-unresolved onboard includes probe cost ---


def test_onboard_unresolved_strategy_folds_probe_cost(monkeypatch, tmp_path):
    """Even when strategy resolution fails, the probe cost is recorded."""
    endpoint = ep("m/unresolved_probe")
    probe_cost = 0.01

    async def fake_resolve_strategies(client, endpoints, policy=None, probe_spend=None):
        if probe_spend is not None:
            for e in endpoints:
                probe_spend[str(e)] = Spend(cost=probe_cost, n_queries=1, n_errors=1)
        return {}, {}  # empty: no strategy resolved

    monkeypatch.setattr(config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        type(config), "spend_dir", property(lambda self: tmp_path / "spend")
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.OpenRouterClient", _FakeClient
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.resolve_strategies", fake_resolve_strategies
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.select_monitoring_targets",
        lambda candidates, policy, popular_models: (list(candidates), {}),
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.fetch_popular_models_safe", lambda top_n: []
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.ENDPOINTS_CACHE_BI_PATH",
        tmp_path / "endpoints_cache_bi.yaml",
    )

    asyncio.run(ue.update_endpoints_bi_lifecycle([endpoint]))

    lines = _read_jsonl(tmp_path / "spend")
    onboard_lines = [e for e in lines if e["kind"] == "onboard"]
    assert onboard_lines, "Expected an onboard line even with unresolved strategy"
    assert abs(onboard_lines[0]["cost"] - probe_cost) < 1e-9


# --- Integration: monitor folds probe cost ---


def test_monitor_folds_probe_cost(monkeypatch, tmp_path):
    """Monitor ledger line cost == probe cost + sampling cost."""
    probe_cost = 0.04
    sampling_cost = 0.06

    state, results = _state_from_fixture("openai2fgpt-4o-mini23azure")
    epoch = state.current_epoch
    monthly_path = tmp_path / "monthly.json"
    daily_batch = {bi: [(NOW.isoformat(), "tok")] for bi in epoch.border_inputs}

    async def fake_sample_prompts(*args, **kwargs):
        record_query(sampling_cost, False)
        return daily_batch, 0

    monkeypatch.setattr(type(config), "spend_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(monitor_mod, "get_output_path", lambda ep, ym: monthly_path)
    monkeypatch.setattr(monitor_mod, "load_phase2_results", lambda d: results)
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)

    probe_spend = {str(state.endpoint): Spend(cost=probe_cost, n_queries=2)}

    async def go():
        from trackllm_website.api import OpenRouterClient

        async with OpenRouterClient() as client:
            await run_endpoint(client, PlainStrategy(), state, NOW, probe_spend)

    asyncio.run(go())

    by_kind = cumulative_by_kind(tmp_path)
    assert abs(by_kind.get("monitor", 0) - (probe_cost + sampling_cost)) < 1e-9


# --- Helpers ---


def _read_jsonl(spend_dir: Path) -> list[dict]:
    lines = []
    if not spend_dir.exists():
        return lines
    for f in spend_dir.glob("*/*.jsonl"):
        for line in f.read_bytes().splitlines():
            if line.strip():
                lines.append(orjson.loads(line))
    return lines


def _state_from_fixture(slug: str):
    endpoint = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))
    results = orjson.loads((FIXTURES / slug / "data.json").read_bytes())
    state = migrate_endpoint(endpoint, results)
    state.status = "monitoring"
    state.retired = None
    epoch = state.epochs[0]
    epoch.end = None
    epoch.end_reason = None
    return state, results
