"""Drift-from-reference for LT: distance of daily behavior from a baseline period.

Display-only companion to the change-detection statistic in lt_scores.py: the LT
analog of B3IT total variation. 0 while the endpoint matches its reference
period; rises and stays elevated after a real change.
"""

import statistics
from collections import defaultdict
from datetime import datetime, timezone

REFERENCE_DAYS = 14
# Every logprob is clipped here before anything else sees it. Some providers return
# a sentinel in place of a logprob (-9999.0; the float32 minimum for -inf), and one
# such response used to make the detection statistic astronomically large and its
# σ infinite. -40 sits below 99.995% of genuine top-20 logprobs (0.01% quantile
# -37.5 over 14M values) and far above every sentinel.
LOGPROB_FLOOR = -40.0

# Publication gate. A detected changepoint is published once the drift level
# after it differs from the level before it: |mean of the first
# LT_SHIFT_WINDOW_DAYS days on/after the change - mean of the last
# LT_SHIFT_WINDOW_DAYS days before| >= LT_MIN_SHIFT, decided as soon as
# LT_MIN_POST_DAYS post days exist. This is also the magnitude every page shows.
LT_SHIFT_WINDOW_DAYS = 7
LT_MIN_POST_DAYS = 3
LT_MIN_SHIFT = 0.1


def level_shift(
    day_pairs: list[tuple[str, float]], day: str, window: int, min_post: int
) -> float | None:
    """|mean of the `window` daily values on/after `day` - mean of the `window`
    before|. None while fewer than `min_post` post days exist (or no pre days):
    the level reached is then unknown, never 0."""
    before = [v for d, v in day_pairs if d < day][-window:]
    after = [v for d, v in day_pairs if d >= day][:window]
    if not before or len(after) < min_post:
        return None
    return abs(statistics.mean(after) - statistics.mean(before))


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
