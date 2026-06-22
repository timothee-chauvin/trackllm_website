import asyncio
from datetime import datetime, timezone

import trackllm_website.update_endpoints as ue
from trackllm_website.bi.reinit import ReinitResult
from trackllm_website.bi.state import EndpointBIState, Epoch, RetiredInfo
from trackllm_website.config import Endpoint, config
from trackllm_website.spend import cumulative_by_kind, record_query

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


def _patch_deps(monkeypatch, tmp_path, *, select, reinit=None, vet_endpoint=None):
    monkeypatch.setattr(config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        type(config), "spend_dir", property(lambda self: tmp_path / "spend")
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.OpenRouterClient", _FakeClient
    )

    async def fake_resolve_strategies(client, endpoints, policy=None):
        return {str(e): None for e in endpoints}, []

    monkeypatch.setattr(
        "trackllm_website.update_endpoints.resolve_strategies", fake_resolve_strategies
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.select_monitoring_targets", select
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
    if vet_endpoint is not None:
        monkeypatch.setattr(
            "trackllm_website.update_endpoints.vet_endpoint", vet_endpoint
        )


def _select_all(candidates, policy, popular_models):
    return list(candidates), {e: "test" for e in candidates}


def test_onboard_writes_onboard_spend(monkeypatch, tmp_path):
    async def fake_reinit(client, strategy, endpoint, old_bis, now):
        record_query(0.03, is_error=False)
        return ReinitResult(
            epoch=Epoch(start=now, border_inputs=["a"] * 10, reference={}),
            reason="ok",
        )

    _patch_deps(monkeypatch, tmp_path, select=_select_all, reinit=fake_reinit)
    asyncio.run(ue.update_endpoints_bi_lifecycle([ep("m/new")]))

    cum = cumulative_by_kind(tmp_path / "spend")
    assert cum.get("onboard", 0) > 0


def test_recheck_writes_recheck_spend(monkeypatch, tmp_path):
    from datetime import timedelta

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
        record_query(0.05, is_error=False)
        return ReinitResult(
            epoch=Epoch(start=now, border_inputs=["b"] * 10, reference={}),
            reason="ok",
        )

    _patch_deps(monkeypatch, tmp_path, select=_select_all, reinit=fake_reinit)

    state_dir = config.bi.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    state.save(state_dir)

    asyncio.run(ue.update_endpoints_bi_lifecycle([endpoint]))

    cum = cumulative_by_kind(tmp_path / "spend")
    assert cum.get("recheck", 0) > 0


def test_timeout_still_writes_spend(monkeypatch, tmp_path):
    async def slow_reinit(client, strategy, endpoint, old_bis, now):
        record_query(0.01, is_error=False)
        await asyncio.sleep(1)

    _patch_deps(monkeypatch, tmp_path, select=_select_all, reinit=slow_reinit)
    monkeypatch.setattr(config.bi.reinit, "onboard_timeout_seconds", 0.05)

    asyncio.run(ue.update_endpoints_bi_lifecycle([ep("m/slow")]))

    # Even with a timeout, the partial spend must be recorded.
    cum = cumulative_by_kind(tmp_path / "spend")
    assert cum.get("onboard", 0) > 0


def test_vetting_writes_vetting_spend(monkeypatch, tmp_path):
    from trackllm_website.bi.common import PlainStrategy
    from trackllm_website.bi.vetting import VetResult

    async def fake_vet_endpoint(client, endpoint, strategy):
        record_query(0.02, is_error=False)
        return VetResult(bucket="candidate", cost_per_request=0.02)

    async def fake_resolve_strategies(client, endpoints, policy=None):
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
        return [ep("m/vet")]

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
        "trackllm_website.update_endpoints.load_policy",
        lambda path: None,
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.exceeds_ceiling",
        lambda cost, model, provider, policy: False,
    )

    asyncio.run(ue.update_endpoints_bi())

    cum = cumulative_by_kind(tmp_path / "spend")
    assert cum.get("vetting", 0) > 0
