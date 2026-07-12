from datetime import datetime, timezone

import pytest

from trackllm_website.bi.state import EndpointBIState, Epoch, RetiredInfo
from trackllm_website.config import Endpoint
from trackllm_website.generate_site.b3it import derive_b3it, discover_b3it_views


def _ep():
    return Endpoint(
        api="openrouter", model="m/a", provider="p", cost=[0.1, 0.2], max_logprobs=None
    )


def test_retired_no_reference_yields_empty_tv_but_full_timeline():
    state = EndpointBIState(
        endpoint=_ep(),
        status="retired",
        retired=RetiredInfo(
            reason="no_bis",
            since=datetime(2026, 2, 5, tzinfo=timezone.utc),
            last_recheck=datetime(2026, 2, 5, tzinfo=timezone.utc),
        ),
        epochs=[
            Epoch(
                start=datetime(2026, 1, 14, tzinfo=timezone.utc),
                border_inputs=[],
                reference={},
                end=datetime(2026, 2, 5, tzinfo=timezone.utc),
                end_reason="gap",
            )
        ],
    )
    view = derive_b3it(state, {})
    assert view.status == "retired"
    assert view.retired_reason == "no_bis"
    assert view.tv_series == {"dates": [], "values": []}
    assert len(view.epochs) == 1
    assert view.epochs[0]["end_reason"] == "gap"
    assert view.n_bis == 0


def test_discover_loads_phase2_for_closed_epochs(tmp_path, monkeypatch):
    """Closed/retired epochs must be scanned so historical changes stay visible."""
    state = EndpointBIState(
        endpoint=_ep(),
        status="retired",
        retired=RetiredInfo(
            reason="stalled",
            since=datetime(2026, 2, 5, tzinfo=timezone.utc),
            last_recheck=datetime(2026, 2, 5, tzinfo=timezone.utc),
        ),
        epochs=[
            Epoch(
                start=datetime(2026, 1, 14, tzinfo=timezone.utc),
                border_inputs=[],
                reference={},
                end=datetime(2026, 2, 5, tzinfo=timezone.utc),
                end_reason="gap",
            )
        ],
    )
    state.save(tmp_path / "state")

    loaded: list = []

    def _spy(path):
        loaded.append(path)
        return {}

    monkeypatch.setattr("trackllm_website.generate_site.b3it.load_phase2_results", _spy)
    views = discover_b3it_views(tmp_path / "state", tmp_path / "phase_2")
    assert loaded, "phase_2 must be loaded for closed-epoch endpoints"
    assert views[state.slug].status == "retired"


def _daily_batch(day: int, token: str):
    ts = f"2026-01-{day:02d}T00:00:00+00:00"
    return ts, [(ts, token)] * 10


def test_closed_epoch_with_results_yields_tv_and_changes():
    """A change inside a closed epoch is surfaced (previously derived as empty)."""
    ref = {"p1": [("2026-01-01T00:00:00Z", "A")] * 10}
    # 12 stable days (token A -> TV 0), then 6 shifted days (token B -> TV 1)
    results = {
        "p1": dict(
            [_daily_batch(d, "A") for d in range(1, 13)]
            + [_daily_batch(d, "B") for d in range(13, 19)]
        )
    }
    state = EndpointBIState(
        endpoint=_ep(),
        status="retired",
        retired=RetiredInfo(
            reason="stalled",
            since=datetime(2026, 2, 1, tzinfo=timezone.utc),
            last_recheck=datetime(2026, 2, 1, tzinfo=timezone.utc),
        ),
        epochs=[
            Epoch(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                border_inputs=["p1"],
                reference=ref,
                end=datetime(2026, 2, 1, tzinfo=timezone.utc),
                end_reason="gap",
            )
        ],
    )
    view = derive_b3it(state, results)
    assert view.tv_series["values"], "closed epoch must produce a TV series"
    assert view.changes, "a change onset must be detected in the closed epoch"
    assert view.changes[0]["kind"] == "onset"


def test_derivation_restricts_to_top_k_ranked_bis(monkeypatch):
    """TV is computed over the top-k ranked BIs, not the full (diluting) set."""
    ref = {"signal": [("t0", "A")] * 10, "noise": [("t0", "A")] * 10}
    results = {
        "signal": {
            "2026-01-01T00:00:00+00:00": [("x", "A")] * 10,
            "2026-01-02T00:00:00+00:00": [("x", "B")] * 10,  # flips -> TV 1
        },
        "noise": {
            "2026-01-01T00:00:00+00:00": [("x", "A")] * 10,
            "2026-01-02T00:00:00+00:00": [("x", "A")] * 10,  # stable -> TV 0
        },
    }
    state = EndpointBIState(
        endpoint=_ep(),
        status="monitoring",
        retired=None,
        epochs=[
            Epoch(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                border_inputs=["signal", "noise"],
                reference=ref,
            )
        ],
    )
    # Ranking keeps only the signal BI; the diluting noise BI is dropped.
    monkeypatch.setattr(
        "trackllm_website.generate_site.b3it.select_top_bis",
        lambda reference, k: ["signal"],
    )
    view = derive_b3it(state, results)
    # Full set would average to 0.5; top-k (signal only) is 1.0.
    assert view.tv_series["values"] == [pytest.approx(1.0)]


def test_monitoring_with_reference_yields_tv_series():
    ref = {"p1": [("2026-06-01T00:00:00Z", "A")] * 10}
    results = {
        "p1": {
            "2026-06-01T00:00:00+00:00": [("2026-06-01T00:00:00Z", "A")] * 10,
            "2026-06-02T00:00:00+00:00": [("2026-06-02T00:00:00Z", "B")] * 10,
        }
    }
    state = EndpointBIState(
        endpoint=_ep(),
        status="monitoring",
        retired=None,
        epochs=[
            Epoch(
                start=datetime(2026, 6, 1, tzinfo=timezone.utc),
                border_inputs=["p1"],
                reference=ref,
            )
        ],
    )
    view = derive_b3it(state, results)
    assert view.status == "monitoring"
    assert view.n_bis == 1
    assert view.tv_series["values"]  # non-empty
    assert view.tv_series["values"][0] == pytest.approx(1.0)
