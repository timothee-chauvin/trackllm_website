from trackllm_website.generate_site.changes import merge_changes

# --- magnitude_display tests (TDD RED first) ---


def _lt_event(sigma):
    return {
        "endpoint": "s1",
        "index": 1,
        "date": "2026-01-01T00:00:00Z",
        "sigma": sigma,
        "first_detected": "2026-06-01T00:00:00Z",
    }


class _LT:
    model = "m/a"
    provider = "p"


def test_lt_null_sigma_shows_inf():
    events = merge_changes({"s1": [_lt_event(None)]}, {"s1": _LT()}, {})
    assert events[0].magnitude_display == "∞σ"


def test_lt_huge_sigma_shows_inf():
    events = merge_changes({"s1": [_lt_event(2.0e38)]}, {"s1": _LT()}, {})
    assert events[0].magnitude_display == "∞σ"


def test_lt_normal_sigma_shows_rounded():
    events = merge_changes({"s1": [_lt_event(12.0)]}, {"s1": _LT()}, {})
    assert events[0].magnitude_display == "12σ"


def test_b3it_change_detected_magnitude_display_blank():
    class V:
        model = "m/b"
        provider = "q"
        epochs = [
            {
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-05-01T00:00:00Z",
                "end_reason": "change_detected",
                "change_date": "2026-04-15T00:00:00Z",
            }
        ]

    events = merge_changes({}, {}, {"s2": V()})
    assert events[0].magnitude_display == ""


# --- existing tests ---


def test_merge_sorts_newest_first_across_methods():
    lt_changes = {
        "s1": [
            {
                "endpoint": "s1",
                "index": 5,
                "date": "2026-03-01T00:00:00Z",
                "sigma": 12.0,
                "first_detected": "2026-06-01T00:00:00Z",
            }
        ]
    }

    class V:
        model = "m/b"
        provider = "q"
        epochs = [
            {
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-05-01T00:00:00Z",
                "end_reason": "change_detected",
                "change_date": "2026-04-15T00:00:00Z",
            }
        ]

    class LT:
        model = "m/a"
        provider = "p"

    events = merge_changes(lt_changes, {"s1": LT()}, {"s2": V()})
    assert [e.method for e in events] == ["B3IT", "LT"]  # 2026-04-15 > 2026-03-01
    assert events[0].slug == "s2"
    assert events[1].magnitude == 12.0


def test_empty_inputs_yield_empty_feed():
    assert merge_changes({}, {}, {}) == []


def test_lt_event_missing_slug_falls_back_to_slug():
    lt_changes = {
        "unknown2fslug": [
            {
                "endpoint": "unknown2fslug",
                "index": 1,
                "date": "2026-02-01T00:00:00Z",
                "sigma": 5.0,
                "first_detected": "2026-06-01T00:00:00Z",
            }
        ]
    }
    events = merge_changes(lt_changes, {}, {})
    assert events[0].model == "unknown2fslug"
    assert events[0].provider == ""
    assert events[0].magnitude == 5.0
