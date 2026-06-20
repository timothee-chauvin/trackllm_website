from datetime import datetime, timezone

import orjson

from trackllm_website.lt_events import (
    LTChangeEvent,
    load_events,
    merge_events,
    save_events,
)
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


def test_load_events_tolerates_null_sigma(tmp_path):
    """Regression: a `sigma: null` on disk (from a non-finite value serialized by
    orjson) must load instead of raising a ValidationError and stalling the job."""
    path = tmp_path / "lt_changes.json"
    path.write_bytes(
        orjson.dumps(
            {
                "slug": [
                    {
                        "endpoint": "slug",
                        "index": 150,
                        "date": "2026-01-01T00:00:00+00:00",
                        "sigma": None,
                        "first_detected": "2026-06-15T00:00:00+00:00",
                    }
                ]
            }
        )
    )
    events = load_events(path)
    assert events["slug"][0].sigma is None


def test_events_roundtrip_with_none_sigma(tmp_path):
    path = tmp_path / "lt_changes.json"
    evt = LTChangeEvent(
        endpoint="slug", index=150, date=NOW, sigma=None, first_detected=NOW
    )
    save_events(path, {"slug": [evt]})
    assert load_events(path)["slug"][0].sigma is None
