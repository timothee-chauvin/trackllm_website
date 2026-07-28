from datetime import datetime, timezone

import pytest

from trackllm_website.generate_site.freshness import as_utc, last_phase2_query, latest


def test_as_utc_accepts_both_spellings_of_utc():
    assert as_utc("2026-07-24T05:10:54Z") == as_utc("2026-07-24T05:10:54+00:00")
    assert as_utc("2026-07-24T05:10:54Z") == datetime(
        2026, 7, 24, 5, 10, 54, tzinfo=timezone.utc
    )


def test_as_utc_reads_a_naive_timestamp_as_utc():
    assert as_utc("2026-07-24T05:10:54") == datetime(
        2026, 7, 24, 5, 10, 54, tzinfo=timezone.utc
    )


def test_as_utc_converts_another_offset():
    assert as_utc("2026-07-24T07:10:54+02:00") == as_utc("2026-07-24T05:10:54Z")


def test_as_utc_raises_on_an_unparseable_timestamp():
    with pytest.raises(ValueError):
        as_utc("last tuesday")


def test_latest_compares_across_spellings():
    assert (
        latest(["2026-07-25T00:00:00+00:00", "2026-07-24T23:59:00Z"])
        == "2026-07-25T00:00:00Z"
    )


def test_latest_normalizes_datetimes_to_z():
    dt = datetime(2026, 7, 25, 6, 30, tzinfo=timezone.utc)
    assert latest([dt, "2026-07-24T00:00:00Z"]) == "2026-07-25T06:30:00Z"


def test_latest_skips_missing_values():
    assert latest([None, "2026-07-24T00:00:00Z", None]) == "2026-07-24T00:00:00Z"


def test_latest_of_nothing_is_none():
    assert latest([]) is None
    assert latest([None]) is None


def test_last_phase2_query_prefers_the_sample_times_over_the_batch_key():
    """The batch key is when the run started; a long batch trails it by an hour."""
    results = {
        "p1": {
            "2026-07-24T04:13:08+00:00": [
                ("2026-07-24T05:10:54+00:00", "A"),
                ("2026-07-24T05:17:03+00:00", "B"),
            ]
        }
    }
    assert last_phase2_query(results) == "2026-07-24T05:17:03Z"


def test_last_phase2_query_ignores_older_batches():
    results = {
        "p1": {
            "2026-07-27T21:56:04+00:00": [("2026-07-27T21:57:46+00:00", "A")],
            "2026-07-24T04:13:08+00:00": [("2026-07-24T05:17:03+00:00", "A")],
        },
        "p2": {"2026-07-24T04:13:08+00:00": [("2026-07-24T05:18:00+00:00", "A")]},
    }
    assert last_phase2_query(results) == "2026-07-27T21:57:46Z"


def test_last_phase2_query_falls_back_to_the_batch_key_without_samples():
    assert last_phase2_query({"p1": {"2026-07-24T04:13:08+00:00": []}}) == (
        "2026-07-24T04:13:08Z"
    )


def test_last_phase2_query_none_without_results():
    assert last_phase2_query({}) is None
    assert last_phase2_query({"p1": {}}) is None
