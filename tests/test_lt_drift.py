from datetime import datetime, timedelta, timezone

from trackllm_website.lt_drift import REFERENCE_DAYS, compute_drift_series


def _obs(day, dist):
    return datetime(2026, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=day), dist


def test_too_few_days_returns_empty():
    assert compute_drift_series([_obs(0, {"A": -0.1}), _obs(1, {"A": -0.1})]) == []


def test_stable_series_stays_near_zero():
    s = compute_drift_series([_obs(d, {"A": -0.02, "B": -4.0}) for d in range(30)])
    assert len(s) == 30 and max(v for _, v in s) < 0.05


def test_sustained_shift_raises_drift():
    stable = [_obs(d, {"A": -0.02, "B": -4.0}) for d in range(15)]
    shifted = [_obs(d, {"A": -4.0, "B": -0.02}) for d in range(15, 30)]
    s = compute_drift_series(stable + shifted)
    assert max(v for dt, v in s if dt.day <= 10 and dt.month == 1) < 0.3
    assert (
        min(v for dt, v in s if dt >= datetime(2026, 1, 25, tzinfo=timezone.utc)) > 1.0
    )


def test_unsorted_input_is_handled():
    s = compute_drift_series([_obs(d, {"A": -0.02}) for d in reversed(range(5))])
    assert [dt.day for dt, _ in s] == [1, 2, 3, 4, 5]


def test_duplicate_timestamps_do_not_crash():
    """Observations can share an exact timestamp (seen in real data, e.g.
    x-ai/grok-3-beta @ xai). Sorting must not fall back to comparing dicts."""
    ts = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    obs = []
    for d in range(5):
        obs.append((ts + timedelta(days=d), {"A": -0.02, "B": -4.0}))
        obs.append((ts + timedelta(days=d), {"A": -0.03, "B": -4.1}))  # dup timestamp
    s = compute_drift_series(obs)
    assert [dt.day for dt, _ in s] == [1, 2, 3, 4, 5]
    assert all(isinstance(v, float) for _, v in s)


def test_single_day_excursion_is_not_smoothed_away():
    """A one-day spike must reach the series. The 5-point rolling median this
    module used to apply deleted exactly this shape -- which on a site whose job
    is surfacing undisclosed changes is the signal, not the noise."""
    obs = [_obs(d, {"A": -0.02, "B": -4.0}) for d in range(20)]
    obs[10] = _obs(10, {"A": -4.0, "B": -0.02})
    s = compute_drift_series(obs)
    spike = next(v for dt, v in s if dt.day == 11)
    assert spike > 1.0, "the one-day excursion was flattened"
    assert max(v for dt, v in s if dt.day != 11) < 0.3, "neighbours were dragged up"


def test_no_repeated_runs_from_filtering():
    """The filter turned distinct daily values into long constant runs; without it,
    distinct daily inputs stay distinct.

    Drift is an *absolute* deviation from the reference mean, so the days that vary
    have to move away from a settled reference -- a ramp spanning the reference
    period would fold onto itself and duplicate for reasons that are not filtering.
    """
    reference = [_obs(d, {"A": -0.02, "B": -4.0}) for d in range(REFERENCE_DAYS)]
    drifting = [
        _obs(REFERENCE_DAYS + d, {"A": -0.02 - (d + 1) * 0.05, "B": -4.0})
        for d in range(12)
    ]
    vals = [v for _, v in compute_drift_series(reference + drifting)]
    varying = vals[REFERENCE_DAYS:]  # the reference days are legitimately all 0.0
    assert len(set(varying)) == len(varying), (
        "distinct days collapsed into repeated values"
    )
    assert varying == sorted(varying), "a day's value depends on its neighbours"
