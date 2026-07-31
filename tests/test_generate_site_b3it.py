import json
from datetime import datetime, timezone

import pytest

from trackllm_website.bi.state import EndpointBIState, Epoch, RetiredInfo
from trackllm_website.config import Endpoint, config
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

    When a re-initialised epoch keeps some of the previous epoch's border inputs,
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
    assert {"date": "2026-01-05T00:00:00+00:00", "kind": "scan"} in view.changes
