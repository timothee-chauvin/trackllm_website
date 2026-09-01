"""Drift-from-reference for LT: distance of daily behavior from a baseline period.

Display-only companion to the change-detection statistic in lt_scores.py: the LT
analog of B3IT total variation. 0 while the endpoint matches its reference
period; rises and stays elevated after a real change.
"""

import statistics
from collections import defaultdict
from datetime import datetime, timezone

REFERENCE_DAYS = 14
LOGPROB_FLOOR = -30.0


def _mean_vector(
    dicts: list[dict[str, float]], extra_tokens: set[str], floor: float
) -> dict[str, float]:
    """Left-censor missing tokens to `floor` (mirroring build_tensor's censoring,
    but with one floor for the whole series: a group's own minimum would make
    a day that returned fewer distinct top-k tokens than the reference read as
    drifted by the gap between the two floors alone)."""
    # sorted for the same reason as build_tensor: iteration order feeds the
    # summation below, and hash order would make every recompute churn.
    tokens = sorted({t for d in dicts for t in d} | extra_tokens)
    return {t: statistics.mean([d.get(t, floor) for d in dicts]) for t in tokens}


def compute_drift_series(
    observations: list[tuple[datetime, dict[str, float]]],
    first_change: datetime | None,
) -> list[tuple[datetime, float]]:
    """Compute daily drift series from reference period baseline.

    The reference is the REFERENCE_DAYS before `first_change` (the endpoint's
    earliest detected changepoint), so a young endpoint's baseline is never a
    blend of both regimes -- which made drift read *higher* before the change
    than after it. Without a change (or with no observation before it), the
    reference is the first REFERENCE_DAYS of the series.
    """
    obs = sorted(
        (
            (dt, {t: max(LOGPROB_FLOOR, v) for t, v in d.items()})
            for dt, d in observations
            if d
        ),
        key=lambda x: x[0],
    )
    if len({dt.date() for dt, _ in obs}) < 3:
        return []
    ref_dicts = []
    if first_change is not None:
        ref_dicts = [
            d
            for dt, d in obs
            if dt < first_change and (first_change - dt).days < REFERENCE_DAYS
        ]
    if not ref_dicts:
        start = obs[0][0]
        ref_dicts = [d for dt, d in obs if (dt - start).days < REFERENCE_DAYS]
    floor = min(min(d.values()) for _, d in obs)
    ref_tokens = {t for d in ref_dicts for t in d}
    ref_mean = _mean_vector(ref_dicts, ref_tokens, floor)
    by_day = defaultdict(list)
    for dt, d in obs:
        by_day[dt.date()].append(d)
    # The daily mean is the only aggregation. There used to be a 5-point rolling
    # median on top of it; it erased single-day excursions -- the exact shape this
    # site exists to surface -- and collapsed distinct days into constant runs.
    series = []
    for day in sorted(by_day):
        day_mean = _mean_vector(by_day[day], ref_tokens, floor)
        tokens = sorted(set(day_mean) | set(ref_mean))
        drift = statistics.mean(
            abs(day_mean.get(t, floor) - ref_mean.get(t, floor)) for t in tokens
        )
        series.append(
            (
                datetime(day.year, day.month, day.day, tzinfo=timezone.utc),
                round(drift, 4),
            )
        )
    return series
