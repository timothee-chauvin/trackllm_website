import asyncio
from datetime import datetime, timedelta, timezone

import trackllm_website.update_endpoints as ue
from trackllm_website.bi.reinit import ReinitResult
from trackllm_website.bi.state import EndpointBIState, Epoch, RetiredInfo
from trackllm_website.config import Endpoint, config

NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)


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


def _patch_deps(monkeypatch, tmp_path, *, reinit=None):
    monkeypatch.setattr(config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        type(config), "spend_dir", property(lambda self: tmp_path / "spend")
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.OpenRouterClient", _FakeClient
    )

    async def fake_resolve_strategies(client, endpoints, policy=None, probe_spend=None):
        return {str(e): None for e in endpoints}, []

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
    if reinit is not None:
        monkeypatch.setattr("trackllm_website.update_endpoints.reinit", reinit)


def test_onboarded_outcome(monkeypatch, tmp_path):
    async def fake_reinit(client, strategy, endpoint, old_bis, now):
        return ReinitResult(
            epoch=Epoch(start=now, border_inputs=["x"] * 12, reference={}),
            reason="ok",
        )

    _patch_deps(monkeypatch, tmp_path, reinit=fake_reinit)
    report = asyncio.run(ue.update_endpoints_bi_lifecycle([ep("m/new")]))

    assert len(report.rows) == 1
    row = report.rows[0]
    assert row.outcome == "onboarded"
    assert row.n_bis == 12
    assert row.model == "m/new"
    assert row.provider == "p"


def test_no_bis_outcome(monkeypatch, tmp_path):
    async def fake_reinit(client, strategy, endpoint, old_bis, now):
        return ReinitResult(epoch=None, reason="no_bis")

    _patch_deps(monkeypatch, tmp_path, reinit=fake_reinit)
    report = asyncio.run(ue.update_endpoints_bi_lifecycle([ep("m/nobis")]))

    assert len(report.rows) == 1
    row = report.rows[0]
    assert row.outcome == "no_bis"
    assert row.n_bis is None


def test_recheck_resurrected_outcome(monkeypatch, tmp_path):
    endpoint = ep("m/old")
    state = EndpointBIState(
        endpoint=endpoint,
        status="retired",
        retired=RetiredInfo(
            reason="stalled",
            since=NOW - timedelta(days=60),
            last_recheck=NOW - timedelta(days=20),
        ),
        epochs=[],
    )

    async def fake_reinit(client, strategy, endpoint, old_bis, now):
        return ReinitResult(
            epoch=Epoch(start=now, border_inputs=["y"] * 7, reference={}),
            reason="ok",
        )

    _patch_deps(monkeypatch, tmp_path, reinit=fake_reinit)

    state_dir = config.bi.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state.save(state_dir)

    report = asyncio.run(ue.update_endpoints_bi_lifecycle([endpoint]))

    recheck_rows = [r for r in report.rows if r.outcome == "recheck_resurrected"]
    assert len(recheck_rows) == 1
    assert recheck_rows[0].n_bis == 7


def test_recheck_still_no_bis_outcome(monkeypatch, tmp_path):
    endpoint = ep("m/old2")
    state = EndpointBIState(
        endpoint=endpoint,
        status="retired",
        retired=RetiredInfo(
            reason="stalled",
            since=NOW - timedelta(days=60),
            last_recheck=NOW - timedelta(days=20),
        ),
        epochs=[],
    )

    async def fake_reinit(client, strategy, endpoint, old_bis, now):
        return ReinitResult(epoch=None, reason="no_bis")

    _patch_deps(monkeypatch, tmp_path, reinit=fake_reinit)

    state_dir = config.bi.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state.save(state_dir)

    report = asyncio.run(ue.update_endpoints_bi_lifecycle([endpoint]))

    recheck_rows = [r for r in report.rows if r.outcome == "recheck_still_no_bis"]
    assert len(recheck_rows) == 1
    assert recheck_rows[0].n_bis is None


def test_empty_report_when_no_to_init(monkeypatch, tmp_path):
    _patch_deps(monkeypatch, tmp_path)
    # No candidates → no onboards, no rechecks → early return with empty report
    report = asyncio.run(ue.update_endpoints_bi_lifecycle([]))

    assert report.rows == []
    assert report.date != ""
