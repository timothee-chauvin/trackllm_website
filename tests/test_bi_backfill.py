from datetime import datetime, timezone

from trackllm_website.bi.backfill import (
    already_logged,
    eligible,
    segment_events,
    spaced_events,
)
from trackllm_website.bi.scan import ScanEvent
from trackllm_website.bi.state import Epoch


def _batch(day: int, token: str):
    ts = f"2026-01-{day:02d}T00:00:00+00:00"
    return ts, [(ts, token)] * 10


def _results(*runs: tuple[range, str]) -> dict:
    return {"p1": dict(_batch(d, tok) for days, tok in runs for d in days)}


def test_segment_finds_single_split():
    results = _results((range(1, 11), "A"), (range(11, 21), "B"))
    events = segment_events(results)
    assert [e.split_ts[8:10] for e in events] == ["11"]


def test_segment_recurses_into_both_sides():
    results = _results(
        (range(1, 11), "A"), (range(11, 21), "B"), (range(21, 31), "C")
    )
    events = segment_events(results)
    assert [e.split_ts[8:10] for e in events] == ["11", "21"]


def test_segment_stable_no_events():
    assert segment_events(_results((range(1, 21), "A"))) == []


def test_spaced_events_merges_one_messy_transition():
    evs = [
        ScanEvent(split_ts=f"2026-05-{d}T00:00:00+00:00", p_value=0.001)
        for d in ("23", "26", "28")
    ]
    assert spaced_events(evs, days=7) == [evs[0]]
    assert spaced_events([evs[0], evs[2]], days=4) == [evs[0], evs[2]]


def test_epoch_with_logged_change_is_skipped():
    epoch = Epoch(
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        border_inputs=[],
        reference={},
        change_date=datetime(2026, 1, 20, tzinfo=timezone.utc),
    )
    assert already_logged(epoch, [])
    assert not already_logged(epoch.model_copy(update={"change_date": None}), [])


def test_open_epoch_within_live_window_is_not_eligible():
    open_epoch = Epoch(
        start=datetime(2026, 7, 1, tzinfo=timezone.utc), border_inputs=[], reference={}
    )
    closed = open_epoch.model_copy(
        update={"end": datetime(2026, 7, 20, tzinfo=timezone.utc)}
    )
    assert not eligible(open_epoch, n_batches=10)  # the live scan's job
    assert eligible(open_epoch, n_batches=60)  # past the live window
    assert eligible(closed, n_batches=10)
