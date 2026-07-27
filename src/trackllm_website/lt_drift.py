"""Drift-from-reference for LT: distance of daily behaviour from a baseline period.

Display-only companion to the change-detection statistic in lt_scores.py: the LT
analogue of B3IT total variation. 0 while the endpoint matches its reference
period; rises and stays elevated after a real change.
"""

import statistics
from collections import defaultdict
from datetime import datetime, timezone

REFERENCE_DAYS = 14
LOGPROB_FLOOR = -30.0
SMOOTH_WINDOW = 5


def _mean_vector(
    dicts: list[dict[str, float]], extra_tokens: set[str]
) -> tuple[dict[str, float], float]:
    """Left-censor missing tokens to the group minimum (mirroring build_tensor)."""
    floor = min(min(d.values()) for d in dicts)
    tokens = {t for d in dicts for t in d} | extra_tokens
    return {t: statistics.mean([d.get(t, floor) for d in dicts]) for t in tokens}, floor


def compute_drift_series(
    observations: list[tuple[datetime, dict[str, float]]],
) -> list[tuple[datetime, float]]:
    """Compute daily drift series from reference period baseline."""
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
    start = obs[0][0]
    ref_dicts = [d for dt, d in obs if (dt - start).days < REFERENCE_DAYS]
    ref_tokens = {t for d in ref_dicts for t in d}
    ref_mean, ref_floor = _mean_vector(ref_dicts, ref_tokens)
    by_day = defaultdict(list)
    for dt, d in obs:
        by_day[dt.date()].append(d)
    raw = []
    for day in sorted(by_day):
        day_mean, day_floor = _mean_vector(by_day[day], ref_tokens)
        floor = min(day_floor, ref_floor)
        tokens = set(day_mean) | set(ref_mean)
        drift = statistics.mean(
            abs(day_mean.get(t, floor) - ref_mean.get(t, floor)) for t in tokens
        )
        raw.append((datetime(day.year, day.month, day.day, tzinfo=timezone.utc), drift))
    vals = [v for _, v in raw]
    half = SMOOTH_WINDOW // 2
    return [
        (dt, round(statistics.median(vals[max(0, i - half) : i + half + 1]), 4))
        for i, (dt, _) in enumerate(raw)
    ]
