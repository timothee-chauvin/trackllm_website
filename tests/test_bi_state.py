from datetime import datetime, timezone

import pytest

from trackllm_website.bi.state import Epoch, EndpointBIState, RetiredInfo
from trackllm_website.config import Endpoint, config
from trackllm_website.util import endpoint_from_slug


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


def test_filter_results_stops_at_the_epoch_end():
    """A closed epoch owns no sample past its end: the next epoch does.

    Without the upper bound, every day after a closure is measured a second
    time against the stale pre-change reference (z-ai/glm-5.2 @ cloudflare).
    """
    epoch = Epoch(
        start=datetime(2026, 1, 14, tzinfo=timezone.utc),
        border_inputs=["a"],
        reference={},
        end=datetime(2026, 1, 16, tzinfo=timezone.utc),
        end_reason="change_detected",
    )
    results = {
        "a": {
            "2026-01-13T00:00:00+00:00": ["before"],
            "2026-01-15T00:00:00+00:00": ["inside"],
            "2026-01-16T00:00:00+00:00": ["at the end"],
            "2026-01-17T00:00:00+00:00": ["after"],
        }
    }
    assert sorted(epoch.filter_results(results)["a"]) == [
        "2026-01-15T00:00:00+00:00",
        "2026-01-16T00:00:00+00:00",
    ]


def test_filter_results_of_an_open_epoch_has_no_upper_bound():
    epoch = Epoch(
        start=datetime(2026, 1, 14, tzinfo=timezone.utc),
        border_inputs=["a"],
        reference={},
    )
    results = {"a": {"2026-09-01T00:00:00+00:00": ["much later"]}}
    assert list(epoch.filter_results(results)["a"]) == ["2026-09-01T00:00:00+00:00"]


def test_retired_requires_info():
    state = make_state()
    state.status = "retired"
    state.retired = RetiredInfo(
        reason="stalled",
        since=datetime(2026, 2, 1, tzinfo=timezone.utc),
        last_recheck=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    assert state.retired.reason == "stalled"


def test_endpoint_from_slug_falls_back_to_state_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config.bi, "data_dir", tmp_path)
    endpoint = Endpoint(
        api="openrouter",
        model="made-up/model-not-in-config",
        provider="nowhere",
        cost=(3, 4),
    )
    state = EndpointBIState(
        endpoint=endpoint,
        status="monitoring",
        epochs=[
            Epoch(
                start=datetime(2026, 1, 14, tzinfo=timezone.utc),
                border_inputs=["a"],
                reference={},
            )
        ],
    )
    state.save(config.bi.state_dir)

    assert endpoint_from_slug(state.slug) == state.endpoint

    with pytest.raises(ValueError):
        endpoint_from_slug("this-slug-does-not-exist")
