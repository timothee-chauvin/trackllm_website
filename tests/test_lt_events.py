from datetime import datetime, timezone

from trackllm_website.lt_events import LTChangeEvent, merge_events
from trackllm_website.lt_scores import ChangePoint

NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)
DATES = [datetime(2026, 1, 1, tzinfo=timezone.utc)] * 300  # placeholder index→date map


def test_new_change_appended():
    events = merge_events("slug", [], [ChangePoint(index=150, sigma=14.0)], DATES, NOW)
    assert len(events) == 1
    assert events[0].first_detected == NOW


def test_recomputed_change_near_existing_is_same_event():
    existing = [
        LTChangeEvent(
            endpoint="slug", index=150, date=DATES[150], sigma=14.0, first_detected=NOW
        )
    ]
    events = merge_events(
        "slug", existing, [ChangePoint(index=160, sigma=15.0)], DATES, NOW
    )
    assert len(events) == 1
    assert events[0].first_detected == NOW  # original detection date kept
    assert events[0].sigma == 15.0  # magnitude refreshed
