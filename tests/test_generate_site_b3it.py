import json
from datetime import datetime, timezone

import pytest

from trackllm_website.bi.state import EndpointBIState, Epoch, RetiredInfo
from trackllm_website.config import Endpoint, config
from trackllm_website.generate_site.b3it import (
    derive_b3it,
    discover_b3it_views,
    to_json,
    votes_json,
)


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
    view = derive_b3it(state, {}, [])
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
    views = discover_b3it_views(
        tmp_path / "state", tmp_path / "phase_2", tmp_path / "scan_backfill.json"
    )
    assert loaded, "phase_2 must be loaded for closed-epoch endpoints"
    assert views[state.slug].status == "retired"


def test_discover_reads_the_backfill_it_is_given_not_the_configured_one(
    tmp_path, monkeypatch
):
    """The scan backfill belongs to the site being built: a synthetic one must not
    inherit production's events through config.bi.data_dir."""
    state = EndpointBIState(
        endpoint=_ep(), status="monitoring", retired=None, epochs=[]
    )
    state.save(tmp_path / "state")
    production = tmp_path / "production"
    production.mkdir()
    (production / "scan_backfill.json").write_text(
        json.dumps({state.slug: [{"date": "2026-03-01T00:00:00Z"}]})
    )
    monkeypatch.setattr(config.bi, "data_dir", production)

    views = discover_b3it_views(
        tmp_path / "state", tmp_path / "phase_2", tmp_path / "scan_backfill.json"
    )
    assert views[state.slug].changes == []


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
    view = derive_b3it(state, results, [])
    assert view.tv_series["values"], "closed epoch must produce a TV series"
    assert view.changes, "a change onset must be detected in the closed epoch"
    assert view.changes[0]["kind"] == "onset"


def test_derivation_restricts_to_top_k_ranked_bis(monkeypatch):
    """TV is computed over the top-k ranked BIs, not the full (diluting) set."""
    day1, day2 = "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"
    ref = {
        "signal": [("2025-12-31T00:00:00+00:00", "A")] * 10,
        "noise": [("2025-12-31T00:00:00+00:00", "A")] * 10,
    }
    results = {
        "signal": {
            day1: [(day1, "A")] * 10,
            day2: [(day2, "B")] * 10,  # flips -> TV 1
        },
        "noise": {
            day1: [(day1, "A")] * 10,
            day2: [(day2, "A")] * 10,  # stable -> TV 0
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
    view = derive_b3it(state, results, [])
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
    view = derive_b3it(state, results, [])
    assert view.status == "monitoring"
    assert view.n_bis == 1
    assert view.tv_series["values"]  # non-empty
    assert view.tv_series["values"][0] == pytest.approx(1.0)


def test_consecutive_epochs_do_not_measure_the_same_day_twice():
    """Each day belongs to exactly one epoch, so the series stays sorted and unique.

    When a re-initialized epoch keeps some of the previous epoch's border inputs,
    those keep being sampled; a closed epoch that ignored its own end would score
    every later day a second time against its stale reference.
    """
    ref_a = {"p1": [("2026-01-01T00:00:00+00:00", "A")] * 10}
    ref_b = {"p1": [("2026-01-10T00:00:00+00:00", "B")] * 10}
    results = {
        "p1": dict(
            [_daily_batch(d, "A") for d in range(1, 10)]
            + [_daily_batch(d, "B") for d in range(10, 13)]
        )
    }
    state = EndpointBIState(
        endpoint=_ep(),
        status="monitoring",
        retired=None,
        epochs=[
            Epoch(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                border_inputs=["p1"],
                reference=ref_a,
                end=datetime(2026, 1, 10, tzinfo=timezone.utc),
                end_reason="change_detected",
                change_date=datetime(2026, 1, 10, tzinfo=timezone.utc),
            ),
            Epoch(
                start=datetime(2026, 1, 10, tzinfo=timezone.utc),
                border_inputs=["p1"],
                reference=ref_b,
            ),
        ],
    )
    dates = derive_b3it(state, results, []).tv_series["dates"]
    assert len(dates) == len(set(dates))
    assert dates == sorted(dates)
    # the closing day is the last evidence of the old epoch, and the new epoch's
    # reference batch -- scored once, against the reference it ended.
    assert dates[-1] == "2026-01-12T00:00:00+00:00"


def test_backfill_events_surface_as_scan_changes():
    state = EndpointBIState(
        endpoint=_ep(),
        status="monitoring",
        retired=None,
        epochs=[
            Epoch(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                border_inputs=["p1"],
                reference={"p1": [("2026-01-01T00:00:00Z", "A")] * 10},
            )
        ],
    )
    backfill = [{"date": "2026-01-05T00:00:00+00:00", "p_value": 0.001}]
    view = derive_b3it(state, {}, backfill)
    assert {
        "date": "2026-01-05T00:00:00+00:00",
        "kind": "scan",
        "detector": "scan",
    } in view.changes


def test_raw_votes_are_exported_beside_the_series(monkeypatch):
    """The hover readout shows the counts each TV point was computed from: one
    batch per series point, each ranked border input's votes next to the epoch's
    reference votes, tokens by descending count."""
    day1, day2 = "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"
    ref = {
        "signal": [("2025-12-31T00:00:00+00:00", t) for t in "AAAB"],
        "noise": [("2025-12-31T00:00:00+00:00", "A")] * 4,
    }
    results = {
        "signal": {day1: [(day1, t) for t in "AB"], day2: [(day2, t) for t in "BBA"]},
        "noise": {day1: [(day1, "A")], day2: [(day2, "A")]},
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
                params={"detector": "adaptive"},
            )
        ],
    )
    monkeypatch.setattr(
        "trackllm_website.generate_site.b3it.select_top_bis",
        lambda reference, k: ["signal"],
    )
    view = derive_b3it(state, results, [])
    assert view.bis == ["signal"]
    assert view.references == [[[0, {"A": 3, "B": 1}]]]
    assert view.epochs[0]["n_ref"] == 1
    assert view.epochs[0]["detector"] == "adaptive"
    # the reference batch (day1) is not a series point; day2 is
    assert view.tv_series["dates"] == [day2]
    # TV of the day's votes vs the reference: |3/4 - 1/3| = |1/4 - 2/3| = 5/12
    assert view.batches == [[[0, {"B": 2, "A": 1}, 0.417]]]
    assert votes_json(view) == {
        "bis": ["signal"],
        "reference": [[[0, {"A": 3, "B": 1}]]],
        "batches": view.batches,
    }
    assert "batches" not in to_json(view)


def test_long_tokens_are_cut_for_display_but_scored_whole():
    long_a, long_b = "x" * 60 + "a", "x" * 60 + "b"
    ts0, ts1 = "2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"
    ref = {"p": [(ts0, long_a)] * 2}
    results = {"p": {ts0: [(ts0, long_a)], ts1: [(ts1, long_a), (ts1, long_b)]}}
    state = EndpointBIState(
        endpoint=_ep(),
        status="monitoring",
        retired=None,
        epochs=[
            Epoch(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                border_inputs=["p"],
                reference=ref,
            )
        ],
    )
    (batch,) = derive_b3it(state, results, []).batches
    ((_, votes, tv),) = batch
    assert votes == {"x" * 40 + "…": 2}
    assert tv == pytest.approx(0.5)


def test_epoch_without_params_has_no_detector():
    state = EndpointBIState(
        endpoint=_ep(),
        status="monitoring",
        retired=None,
        epochs=[
            Epoch(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                border_inputs=[],
                reference={},
            )
        ],
    )
    view = derive_b3it(state, {}, [])
    assert view.epochs[0]["detector"] is None
    assert view.epochs[0]["n_ref"] == 0
    assert view.bis == [] and view.references == [[]] and view.batches == []


def test_change_magnitude_is_the_tv_shift_on_the_full_reference_series(monkeypatch):
    """backfill.py and monitor.decide gate on tv_shift over every reference BI; the
    plot is ranked (top-k) for visibility. What is published is the gate's number,
    so a change can never show a magnitude the gate would have rejected."""
    days = [f"2026-01-{d:02d}T00:00:00+00:00" for d in range(1, 9)]
    ref = {
        "signal": [("2025-12-31T00:00:00+00:00", "A")] * 10,
        "noise": [("2025-12-31T00:00:00+00:00", "A")] * 10,
    }
    results = {
        "signal": {d: [(d, "A" if i < 4 else "B")] * 10 for i, d in enumerate(days)},
        "noise": {d: [(d, "A")] * 10 for d in days},
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
    monkeypatch.setattr(
        "trackllm_website.generate_site.b3it.select_top_bis",
        lambda reference, k: ["signal"],
    )
    backfill = [{"date": days[4], "p_value": 0.001}]
    view = derive_b3it(state, results, backfill)
    # ranked plot: the signal BI alone flips to TV 1; the gate saw both BIs: 0.5
    assert view.tv_series["values"][-1] == pytest.approx(1.0)
    assert view.change_mags[days[4][:10]] == pytest.approx(0.5)
