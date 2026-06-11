from datetime import datetime, timezone

from trackllm_website.bi.state import Epoch, EndpointBIState, RetiredInfo
from trackllm_website.config import Endpoint


def make_state() -> EndpointBIState:
    return EndpointBIState(
        endpoint=Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 2)),
        status="monitoring",
        epochs=[
            Epoch(
                start=datetime(2026, 1, 14, tzinfo=timezone.utc),
                border_inputs=["a", "b"],
                reference={"a": [["2026-01-14T00:00:00+00:00", "tok"]]},
            )
        ],
    )


def test_round_trip(tmp_path):
    state = make_state()
    state.save(tmp_path)
    loaded = EndpointBIState.load(tmp_path / f"{state.slug}.json")
    assert loaded == state


def test_current_epoch_open_and_closed():
    state = make_state()
    assert state.current_epoch is state.epochs[0]
    state.epochs[0].end = datetime(2026, 2, 1, tzinfo=timezone.utc)
    state.epochs[0].end_reason = "change_detected"
    assert state.current_epoch is None


def test_retired_requires_info():
    state = make_state()
    state.status = "retired"
    state.retired = RetiredInfo(
        reason="stalled",
        since=datetime(2026, 2, 1, tzinfo=timezone.utc),
        last_recheck=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    assert state.retired.reason == "stalled"
