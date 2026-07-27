from datetime import datetime, timedelta, timezone

from trackllm_website.lt_drift import compute_drift_series


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
    assert min(v for dt, v in s if dt >= datetime(2026, 1, 25, tzinfo=timezone.utc)) > 1.0


def test_unsorted_input_is_handled():
    s = compute_drift_series([_obs(d, {"A": -0.02}) for d in reversed(range(5))])
    assert [dt.day for dt, _ in s] == [1, 2, 3, 4, 5]
