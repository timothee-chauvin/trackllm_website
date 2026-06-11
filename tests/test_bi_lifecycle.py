import asyncio
from datetime import datetime, timedelta, timezone

from trackllm_website.bi.state import EndpointBIState, Epoch, RetiredInfo
from trackllm_website.config import Endpoint, config
from trackllm_website.update_endpoints import (
    select_lifecycle_actions,
    update_endpoints_bi_lifecycle,
)

NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)


def ep(model):
    return Endpoint(api="openrouter", model=model, provider="p", cost=(1, 1))


def retired_state(model, reason, last_recheck):
    return EndpointBIState(
        endpoint=ep(model),
        status="retired",
        retired=RetiredInfo(
            reason=reason,
            since=NOW - timedelta(days=60),
            last_recheck=last_recheck,
        ),
        epochs=[],
    )


def test_all_new_candidates_onboarded():
    candidates = [ep(f"m/{i}") for i in range(20)]
    actions = select_lifecycle_actions(candidates, {}, NOW)
    assert len(actions.onboard) == 20


def test_recheck_due_only_after_interval():
    states = {
        "due": retired_state("m/due", "stalled", NOW - timedelta(days=20)),
        "recent": retired_state("m/recent", "stalled", NOW - timedelta(days=2)),
    }
    candidates = [ep("m/due"), ep("m/recent")]
    actions = select_lifecycle_actions(candidates, states, NOW)
    assert [s.endpoint.model for s in actions.recheck] == ["m/due"]


def test_delisted_when_absent_from_catalog():
    states = {
        "gone": EndpointBIState(endpoint=ep("m/gone"), status="monitoring", epochs=[])
    }
    actions = select_lifecycle_actions([], states, NOW)
    assert [s.endpoint.model for s in actions.delist] == ["m/gone"]


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def test_onboarding_failure_does_not_abort_others(monkeypatch, tmp_path):
    good = ep("m/good")
    bad = ep("m/bad")

    monkeypatch.setattr(config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(config, "endpoints_bi", [good, bad])
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.OpenRouterClient", _FakeClient
    )

    async def fake_resolve_strategies(client, endpoints):
        return {str(e): None for e in endpoints}, []

    monkeypatch.setattr(
        "trackllm_website.update_endpoints.resolve_strategies",
        fake_resolve_strategies,
    )

    async def fake_reinit(client, strategy, endpoint, old_bis, now):
        if endpoint == bad:
            raise RuntimeError("onboarding blew up")
        return Epoch(
            start=now, border_inputs=["x"], reference={"x": [(now.isoformat(), "t")]}
        )

    monkeypatch.setattr("trackllm_website.update_endpoints.reinit", fake_reinit)

    asyncio.run(update_endpoints_bi_lifecycle())

    state_dir = config.bi.state_dir
    good_state = EndpointBIState.load(state_dir / f"{slugify_eq(good)}.json")
    assert good_state.status == "monitoring"
    assert len(good_state.epochs) == 1
    # The failing endpoint must not have produced a state file.
    assert not (state_dir / f"{slugify_eq(bad)}.json").exists()


def slugify_eq(endpoint):
    from trackllm_website.util import slugify

    return slugify(f"{endpoint.model}#{endpoint.provider}")
