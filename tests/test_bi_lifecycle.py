import asyncio
from datetime import datetime, timedelta, timezone

from trackllm_website.bi.reinit import ReinitResult
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


def test_delisted_only_after_grace_period():
    grace = config.bi.reinit.deselection_grace_days
    states = {
        # Just fell out this run: still within grace, not delisted yet.
        "fresh": EndpointBIState(
            endpoint=ep("m/fresh"),
            status="monitoring",
            epochs=[],
            deselected_since=NOW,
        ),
        # Dropped out long ago: past grace, delisted.
        "stale": EndpointBIState(
            endpoint=ep("m/stale"),
            status="monitoring",
            epochs=[],
            deselected_since=NOW - timedelta(days=grace + 1),
        ),
    }
    actions = select_lifecycle_actions([], states, NOW)
    assert [s.endpoint.model for s in actions.delist] == ["m/stale"]


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def _patch_lifecycle_deps(monkeypatch, tmp_path, *, select, reinit):
    """Wire the non-networked lifecycle dependencies for an in-process run."""
    monkeypatch.setattr(config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.OpenRouterClient", _FakeClient
    )

    async def fake_resolve_strategies(client, endpoints):
        return {str(e): None for e in endpoints}, []

    monkeypatch.setattr(
        "trackllm_website.update_endpoints.resolve_strategies",
        fake_resolve_strategies,
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.select_monitoring_targets", select
    )
    monkeypatch.setattr("trackllm_website.update_endpoints.reinit", reinit)
    # Don't hit the live rankings API in lifecycle tests.
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.fetch_popular_models_safe", lambda top_n: []
    )
    monkeypatch.setattr(
        "trackllm_website.update_endpoints.ENDPOINTS_CACHE_BI_PATH",
        tmp_path / "endpoints_cache_bi.yaml",
    )


def test_onboarding_failure_does_not_abort_others(monkeypatch, tmp_path):
    good = ep("m/good")
    bad = ep("m/bad")

    async def fake_reinit(client, strategy, endpoint, old_bis, now):
        if endpoint == bad:
            raise RuntimeError("onboarding blew up")
        return ReinitResult(
            epoch=Epoch(
                start=now,
                border_inputs=["x"],
                reference={"x": [(now.isoformat(), "t")]},
            ),
            reason="ok",
        )

    # Selection is a no-op here: every candidate is monitored.
    def select_all(candidates, policy, popular_models):
        return list(candidates), {e: "test" for e in candidates}

    _patch_lifecycle_deps(monkeypatch, tmp_path, select=select_all, reinit=fake_reinit)

    asyncio.run(update_endpoints_bi_lifecycle([good, bad]))

    state_dir = config.bi.state_dir
    good_state = EndpointBIState.load(state_dir / f"{slugify_eq(good)}.json")
    assert good_state.status == "monitoring"
    assert len(good_state.epochs) == 1
    # The failing endpoint must not have produced a state file.
    assert not (state_dir / f"{slugify_eq(bad)}.json").exists()


def test_bad_temperature_is_cached_not_monitored(monkeypatch, tmp_path):
    bad_temp = ep("m/badtemp")

    def select_all(candidates, policy, popular_models):
        return list(candidates), {e: "test" for e in candidates}

    async def fake_reinit(client, strategy, endpoint, old_bis, now):
        return ReinitResult(epoch=None, reason="bad_temperature")

    _patch_lifecycle_deps(monkeypatch, tmp_path, select=select_all, reinit=fake_reinit)

    asyncio.run(update_endpoints_bi_lifecycle([bad_temp]))

    from trackllm_website.bi.vetting import EndpointCache
    from trackllm_website.update_endpoints import ENDPOINTS_CACHE_BI_PATH

    cache = EndpointCache.load(ENDPOINTS_CACHE_BI_PATH)
    assert cache.bucket_of(bad_temp) == "bad_temperature"
    # No monitoring state file for a temperature-ignoring endpoint.
    assert not (config.bi.state_dir / f"{slugify_eq(bad_temp)}.json").exists()


def test_only_selected_candidates_are_onboarded(monkeypatch, tmp_path):
    chosen_a, chosen_b = ep("m/a"), ep("m/b")
    rejected = ep("m/c")
    candidates = [chosen_a, chosen_b, rejected]

    # Selection keeps only the first two of the three candidates.
    def select_subset(cands, policy, popular_models):
        selected = [chosen_a, chosen_b]
        return selected, {e: "test" for e in selected}

    onboarded = []

    async def fake_reinit(client, strategy, endpoint, old_bis, now):
        onboarded.append(endpoint)
        return ReinitResult(
            epoch=Epoch(
                start=now,
                border_inputs=["x"],
                reference={"x": [(now.isoformat(), "t")]},
            ),
            reason="ok",
        )

    _patch_lifecycle_deps(
        monkeypatch, tmp_path, select=select_subset, reinit=fake_reinit
    )

    asyncio.run(update_endpoints_bi_lifecycle(candidates))

    assert {e.model for e in onboarded} == {"m/a", "m/b"}
    state_dir = config.bi.state_dir
    assert (state_dir / f"{slugify_eq(chosen_a)}.json").exists()
    assert (state_dir / f"{slugify_eq(chosen_b)}.json").exists()
    assert not (state_dir / f"{slugify_eq(rejected)}.json").exists()


def test_executor_stamps_and_clears_deselected_since(monkeypatch, tmp_path):
    """A monitored endpoint dropping out of selection gets deselected_since stamped
    (but not delisted within grace); one returning to selection gets it cleared."""
    dropped, returning = ep("m/dropped"), ep("m/returning")

    # Selection keeps only `returning`; `dropped` falls out of the selected set.
    def select_returning(candidates, policy, popular_models):
        return [returning], {returning: "test"}

    async def fake_reinit(client, strategy, endpoint, old_bis, now):
        raise AssertionError("no onboarding expected")

    _patch_lifecycle_deps(
        monkeypatch, tmp_path, select=select_returning, reinit=fake_reinit
    )

    state_dir = config.bi.state_dir
    EndpointBIState(endpoint=dropped, status="monitoring", epochs=[]).save(state_dir)
    EndpointBIState(
        endpoint=returning,
        status="monitoring",
        epochs=[],
        deselected_since=NOW - timedelta(days=5),
    ).save(state_dir)

    asyncio.run(update_endpoints_bi_lifecycle([returning]))

    dropped_state = EndpointBIState.load(state_dir / f"{slugify_eq(dropped)}.json")
    returning_state = EndpointBIState.load(state_dir / f"{slugify_eq(returning)}.json")
    # dropped: now out of the selected set, clock just started, still monitoring.
    assert dropped_state.deselected_since is not None
    assert dropped_state.status == "monitoring"
    # returning: back in the selected set, clock cleared.
    assert returning_state.deselected_since is None


def slugify_eq(endpoint):
    from trackllm_website.util import slugify

    return slugify(f"{endpoint.model}#{endpoint.provider}")
