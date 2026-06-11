from datetime import datetime, timedelta, timezone

from trackllm_website.bi.state import EndpointBIState, RetiredInfo
from trackllm_website.config import Endpoint
from trackllm_website.update_endpoints import select_lifecycle_actions

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
