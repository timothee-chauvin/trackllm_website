"""LT (logprob tracking) score computation.

Implements a permutation-free two-sample test on logprob vectors. At each time
point, two adjacent windows of size N_PER_TEST are compared using the L1 norm
of their mean logprob difference, averaged over tokens. When multiple prompts
are present, the score is the mean of per-prompt statistics.

Change detection uses running mean/std normalization + peak detection.
"""

import fire
import numpy as np
import orjson
from datetime import datetime
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from scipy.signal import find_peaks

from pydantic import BaseModel

from trackllm_website.config import config, logger
from trackllm_website.storage import MonthlyData

N_PER_TEST = 24
STAT_SIGMA_THRESHOLD = 12.0
STAT_RUNNING_STD_WINDOW = 100
STAT_EXCLUSION_ZONE = 2 * N_PER_TEST
STAT_ABSOLUTE_THRESHOLD = 1.0
PEAK_DISTANCE = N_PER_TEST

SCORES_FILENAME = "lt_scores.json"


class ChangePoint(BaseModel):
    index: int
    sigma: float


class LTScores(BaseModel):
    n_per_test: int
    dates: list[datetime]
    scores: list[float]
    sigmas: list[float | None]
    changes: list[ChangePoint]


def load_prompt_logprobs(
    prompt_dir: Path,
) -> list[tuple[datetime, dict[str, float]]]:
    """Load all logprob responses for a prompt, sorted by date.

    Returns list of (date, {token: logprob}) pairs.
    """
    results: list[tuple[datetime, dict[str, float]]] = []
    month_dirs = sorted(d for d in prompt_dir.iterdir() if d.is_dir() and "-" in d.name)
    for month_dir in month_dirs:
        year, month = map(int, month_dir.name.split("-"))
        monthly = MonthlyData.load_existing(path=month_dir, year=year, month=month)
        for date, rl in monthly.logprob_responses:
            results.append(
                (date, {tok: float(lp) for tok, lp in zip(rl.tokens, rl.logprobs)})
            )
    results.sort(key=lambda x: x[0])
    return results


def build_tensor(logprob_dicts: list[dict[str, float]]) -> np.ndarray:
    """Build (N, n_tokens) array with left censoring for missing tokens."""
    all_tokens = list({tok for d in logprob_dicts for tok in d})
    tok_idx = {tok: i for i, tok in enumerate(all_tokens)}
    tensor = np.empty((len(logprob_dicts), len(all_tokens)), dtype=np.float64)
    for i, d in enumerate(logprob_dicts):
        min_val = min(d.values())
        tensor[i, :] = min_val
        for tok, lp in d.items():
            tensor[i, tok_idx[tok]] = lp
    return tensor


def compute_statistics(tensor: np.ndarray, n_per_test: int) -> np.ndarray:
    """Compute two-sample test statistics for all consecutive window pairs.

    At index i (ranging from n_per_test to N - n_per_test), compares
    tensor[i-n_per_test:i] vs tensor[i:i+n_per_test].

    Returns array of length max(0, N - 2*n_per_test + 1).
    """
    N = tensor.shape[0]
    if N < 2 * n_per_test:
        return np.array([], dtype=np.float32)

    # sliding_window_view on (N, nt) with axis=0, window=w → (N-w+1, nt, w)
    windows = sliding_window_view(tensor, window_shape=n_per_test, axis=0)
    window_means = windows.mean(axis=2)  # (N-n_per_test+1, nt)

    n_stats = N - 2 * n_per_test + 1
    t1_means = window_means[:n_stats]
    t2_means = window_means[n_per_test : n_per_test + n_stats]
    return np.abs(t1_means - t2_means).mean(axis=1)


def detect_changes(
    scores: np.ndarray,
) -> tuple[list[ChangePoint], np.ndarray]:
    """Detect changes using running mean/std normalization + peak detection.

    Returns (change_points, sigmas) where sigmas[i] is (score - running_mean) /
    running_std (NaN where not yet computable).
    """
    extended_window = STAT_RUNNING_STD_WINDOW + STAT_EXCLUSION_ZONE
    sigmas = np.full(len(scores), np.nan)

    if len(scores) <= extended_window:
        return [], sigmas

    scores = scores.astype(np.float64)
    exceedances = np.zeros(len(scores))

    for i in range(extended_window, len(scores)):
        window = scores[i - extended_window : i - STAT_EXCLUSION_ZONE]
        mean = window.mean()
        std = window.std()
        dev = (scores[i] - mean) / std if std > 0 else float("inf")
        sigmas[i] = dev
        if dev > STAT_SIGMA_THRESHOLD:
            exceedances[i] = scores[i] - (mean + STAT_SIGMA_THRESHOLD * std)

    peaks = find_peaks(exceedances, height=1e-20, distance=PEAK_DISTANCE)[0]
    peaks = [int(p) for p in peaks if scores[p] > STAT_ABSOLUTE_THRESHOLD]
    return [ChangePoint(index=p, sigma=float(sigmas[p])) for p in peaks], sigmas


def compute_endpoint_scores(endpoint_dir: Path) -> LTScores | None:
    """Compute LT scores for an endpoint, averaged across prompts."""
    prompt_dirs = sorted(
        d for d in endpoint_dir.iterdir() if d.is_dir() and (d / "info.json").exists()
    )

    per_prompt_stats: list[np.ndarray] = []
    per_prompt_dates: list[list[datetime]] = []

    for prompt_dir in prompt_dirs:
        data = load_prompt_logprobs(prompt_dir)
        if len(data) < 2 * N_PER_TEST:
            continue
        logprob_dicts = [d for _, d in data]
        dates = [dt for dt, _ in data]
        tensor = build_tensor(logprob_dicts)
        stats = compute_statistics(tensor, N_PER_TEST)
        per_prompt_stats.append(stats)
        per_prompt_dates.append(dates[N_PER_TEST : N_PER_TEST + len(stats)])

    if not per_prompt_stats:
        return None

    longest = max(range(len(per_prompt_stats)), key=lambda i: len(per_prompt_stats[i]))
    ref_dates = per_prompt_dates[longest]
    ref_ts = np.array([d.timestamp() for d in ref_dates])

    # Interpolate all prompts onto the longest prompt's timeline, average where available
    grid = np.full((len(per_prompt_stats), len(ref_dates)), np.nan)
    for i, (stats, dates) in enumerate(zip(per_prompt_stats, per_prompt_dates)):
        ts = np.array([d.timestamp() for d in dates])
        mask = (ref_ts >= ts[0]) & (ref_ts <= ts[-1])
        grid[i, mask] = np.interp(ref_ts[mask], ts, stats)

    avg_scores = np.nanmean(grid, axis=0)
    changes, sigmas = detect_changes(avg_scores)

    return LTScores(
        n_per_test=N_PER_TEST,
        dates=ref_dates,
        scores=avg_scores.tolist(),
        sigmas=[None if np.isnan(v) else v for v in sigmas.tolist()],
        changes=changes,
    )


def _save_scores(path: Path, scores: LTScores):
    path.write_bytes(
        orjson.dumps(
            scores.model_dump(mode="json"),
            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
        )
    )


def _is_scores_current(endpoint_dir: Path) -> bool:
    """Check if scores file is newer than all data files for this endpoint."""
    scores_path = endpoint_dir / SCORES_FILENAME
    if not scores_path.exists():
        return False
    scores_mtime = scores_path.stat().st_mtime
    for prompt_dir in endpoint_dir.iterdir():
        if not prompt_dir.is_dir():
            continue
        for month_dir in prompt_dir.iterdir():
            if not month_dir.is_dir() or "-" not in month_dir.name:
                continue
            queries_path = month_dir / "queries.json"
            if queries_path.exists() and queries_path.stat().st_mtime > scores_mtime:
                return False
    return True


def compute_all():
    """Compute LT scores for all endpoints from scratch."""
    data_dir = Path(config.data_dir)
    n_computed = 0
    for endpoint_dir in sorted(data_dir.iterdir()):
        if not endpoint_dir.is_dir():
            continue
        scores = compute_endpoint_scores(endpoint_dir)
        if scores is None:
            logger.info(f"{endpoint_dir.name}: not enough data")
            continue
        _save_scores(endpoint_dir / SCORES_FILENAME, scores)
        n_computed += 1
        logger.info(
            f"{endpoint_dir.name}: {len(scores.scores)} scores, "
            f"{len(scores.changes)} changes"
        )
    logger.info(f"Computed scores for {n_computed} endpoints")


def compute_latest():
    """Recompute LT scores only for endpoints with new data since last run."""
    data_dir = Path(config.data_dir)
    n_skipped = 0
    n_computed = 0
    for endpoint_dir in sorted(data_dir.iterdir()):
        if not endpoint_dir.is_dir():
            continue
        if _is_scores_current(endpoint_dir):
            n_skipped += 1
            continue
        scores = compute_endpoint_scores(endpoint_dir)
        if scores is None:
            continue
        _save_scores(endpoint_dir / SCORES_FILENAME, scores)
        n_computed += 1
        logger.info(
            f"{endpoint_dir.name}: {len(scores.scores)} scores, "
            f"{len(scores.changes)} changes"
        )
    logger.info(f"Computed: {n_computed}, skipped (up-to-date): {n_skipped}")


if __name__ == "__main__":
    fire.Fire({"all": compute_all, "latest": compute_latest})
