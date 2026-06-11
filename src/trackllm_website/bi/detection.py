"""Pure change-detection functions on BI sampling data (no I/O, no state)."""

import statistics
from collections import Counter

from trackllm_website.bi.analyze import compute_tv_distance, get_distribution
from trackllm_website.bi.phase_2 import Timestamp
from trackllm_website.config import config


def epoch_tv_series(
    reference: dict[str, list], results: dict
) -> list[tuple[Timestamp, float]]:
    """Daily mean TV of each batch vs the epoch reference, for reference prompts.

    `results` is the phase 2 dict {prompt: {timestamp: [(ts, token), ...]}}.
    Days at or before the reference batch are excluded.
    """
    ref_dists = {p: get_distribution(samples) for p, samples in reference.items()}
    all_ts = sorted({ts for p in ref_dists for ts in results.get(p, {})})
    if not all_ts:
        return []
    out: list[tuple[Timestamp, float]] = []
    for ts in all_ts[1:]:
        tvs = []
        for p, ref_dist in ref_dists.items():
            samples = results.get(p, {}).get(ts)
            if not samples:
                continue
            tv = compute_tv_distance(ref_dist, get_distribution(samples))
            if tv is not None:
                tvs.append(tv)
        if tvs:
            out.append((ts, statistics.mean(tvs)))
    return out


def adaptive_transitions(
    tv_over_time: list[tuple[Timestamp, float]],
) -> list[Timestamp]:
    """Sustained deviations from a trailing baseline (see design doc).

    A day deviates when |tv - baseline_mean| exceeds both abs_delta and
    sigma_k * baseline_std; an event fires after `persistence` consecutive
    deviating days, dated at the first one (onset). The baseline excludes the
    most recent `exclusion` days.
    """
    d = config.bi.detection
    timestamps = [ts for ts, _ in tv_over_time]
    vals = [v for _, v in tv_over_time]
    events: list[Timestamp] = []
    streak = 0
    last_event_idx: int | None = None

    for i in range(len(vals)):
        baseline = vals[max(0, i - d.exclusion - d.window) : i - d.exclusion]
        if len(baseline) < d.min_baseline:
            continue
        mean = statistics.mean(baseline)
        std = statistics.stdev(baseline)
        dev = abs(vals[i] - mean)
        if dev > d.abs_delta and dev > d.sigma_k * std:
            streak += 1
        else:
            streak = 0
        if streak == d.persistence:
            onset = i - d.persistence + 1
            if last_event_idx is None or onset - last_event_idx >= d.cooldown:
                events.append(timestamps[onset])
                last_event_idx = onset
    return events


def is_unstable(tv_over_time: list[tuple[Timestamp, float]]) -> bool:
    """Median TV over the trailing window exceeds the instability threshold."""
    d = config.bi.detection
    if not tv_over_time:
        return False
    tail = [v for _, v in tv_over_time[-d.instability_window :]]
    return statistics.median(tail) >= d.instability_threshold


def balance_score(dist: Counter) -> float:
    counts = sorted(dist.values(), reverse=True)
    return counts[1] / counts[0] if len(counts) > 1 else 0.0


def select_top_bis(reference: dict[str, list], k: int) -> list[str]:
    """Top-k prompts by top-2 balance (p2/p1) of their reference distribution."""
    scored = sorted(
        reference,
        key=lambda p: balance_score(get_distribution(reference[p])),
        reverse=True,
    )
    return scored[:k]
