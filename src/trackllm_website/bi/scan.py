"""Early-epoch changepoint scan (no I/O, no state).

Young epochs are a blind spot for the adaptive rule: it cannot fire before ~9
batches, and past that its trailing baseline absorbs whatever level it finds,
so a change in an epoch's first days is never logged (13% of epochs as of
July 2026). The scan tests every admissible split of the epoch's batches —
reference batch included, so a change between the reference burst and the
first daily batch is detectable — against a batch-permutation null. Batches
are whole-day blocks, so day-level serving effects stay inside the null; the
scan needs no trailing baseline and fires as soon as three post-change batches
exist.
"""

import numpy as np
from pydantic import BaseModel

from trackllm_website.bi.results import get_distribution
from trackllm_website.bi.phase_2 import Timestamp
from trackllm_website.config import config


class ScanEvent(BaseModel):
    split_ts: Timestamp  # first post-change batch
    p_value: float


def split_statistic(pre: dict[str, dict], post: dict[str, dict]) -> float:
    """Sum over prompts of Σ_tokens (freq_pre − freq_post)²."""
    total = 0.0
    for prompt, pre_counts in pre.items():
        post_counts = post.get(prompt)
        if not pre_counts or not post_counts:
            continue
        n_pre, n_post = sum(pre_counts.values()), sum(post_counts.values())
        for t in set(pre_counts) | set(post_counts):
            total += (
                pre_counts.get(t, 0) / n_pre - post_counts.get(t, 0) / n_post
            ) ** 2
    return total


def _max_over_splits(
    counts: list[np.ndarray], orders: np.ndarray, min_side_total: int
) -> tuple[np.ndarray, np.ndarray]:
    """Max statistic over admissible splits, for each batch ordering.

    counts: per prompt, an (n_batches, n_tokens) count matrix.
    orders: (n_orderings, n_batches) permutations of batch indices.
    Returns (max_stat, argmax_split) per ordering; -inf where no split is
    admissible (min_side_total samples summed over prompts on each side).
    """
    n = orders.shape[1]
    stats = np.zeros((orders.shape[0], n - 1))
    pre_sizes = np.zeros_like(stats)
    for c in counts:
        prefix = np.cumsum(c[orders], axis=1)[:, :-1, :].astype(np.float64)
        total = c.sum(axis=0, dtype=np.float64)
        n_pre = prefix.sum(axis=2)
        n_post = total.sum() - n_pre
        pre_sizes += n_pre
        with np.errstate(invalid="ignore"):
            diff = prefix / n_pre[:, :, None] - (total - prefix) / n_post[:, :, None]
        stats += np.where((n_pre > 0) & (n_post > 0), np.nansum(diff**2, axis=2), 0.0)
    total_all = sum(float(c.sum()) for c in counts)
    admissible = (pre_sizes >= min_side_total) & (
        total_all - pre_sizes >= min_side_total
    )
    stats = np.where(admissible, stats, -np.inf)
    return stats.max(axis=1), stats.argmax(axis=1) + 1


def scan_pvalue(
    results: dict, rng: np.random.Generator | None = None
) -> ScanEvent | None:
    """Best split of an epoch's batches with its batch-permutation p-value.

    `results` is the epoch-filtered phase 2 dict {prompt: {ts: [(ts, token)]}}.
    Applies only the per-side sample minimum (no batch-window or alpha gate,
    so the calibration sweep can probe any window); None if no split is
    admissible.
    """
    cfg = config.bi.scan
    all_ts = sorted({ts for batches in results.values() for ts in batches})
    n = len(all_ts)
    if n < 2:
        return None

    ts_index = {ts: i for i, ts in enumerate(all_ts)}
    counts = []
    for prompt in sorted(results):
        dists = {ts: get_distribution(s) for ts, s in results[prompt].items() if s}
        tokens = sorted({t for d in dists.values() for t in d})
        if not tokens:
            continue
        c = np.zeros((n, len(tokens)), dtype=np.int64)
        col = {t: j for j, t in enumerate(tokens)}
        for ts, d in dists.items():
            for t, k in d.items():
                c[ts_index[ts], col[t]] = k
        counts.append(c)
    if not counts:
        return None

    min_side_total = cfg.min_side_samples * len(counts)
    identity = np.arange(n)[None, :]
    obs_max, obs_split = _max_over_splits(counts, identity, min_side_total)
    if not np.isfinite(obs_max[0]):
        return None

    if rng is None:
        rng = np.random.default_rng(0)
    perms = np.array([rng.permutation(n) for _ in range(cfg.permutations)])
    perm_max, _ = _max_over_splits(counts, perms, min_side_total)
    p = float((1 + (perm_max >= obs_max[0] - 1e-12).sum()) / (cfg.permutations + 1))
    # The admissible argmax dates the change up to 2 batches early when it
    # fires at the earliest possible moment (the true boundary leaves fewer
    # than min_side_samples on the post side). min_side_total exists to
    # calibrate the test, not to localize: re-locate the split without it.
    _, loc_split = _max_over_splits(counts, identity, 1)
    return ScanEvent(split_ts=all_ts[int(loc_split[0])], p_value=p)


def changepoint_scan(
    results: dict, rng: np.random.Generator | None = None
) -> ScanEvent | None:
    """The production gate: scan_pvalue within the batch window, alpha applied.

    Returns the split (dated at the first post-change batch) when the epoch
    has min_batches..max_batches batches and the p-value clears
    config.bi.scan.alpha, else None.
    """
    cfg = config.bi.scan
    n = len({ts for batches in results.values() for ts in batches})
    if n < cfg.min_batches or n > cfg.max_batches:
        return None
    event = scan_pvalue(results, rng)
    if event is None or event.p_value > cfg.alpha:
        return None
    return event
