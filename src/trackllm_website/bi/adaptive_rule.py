"""Endpoint-adaptive change detection on BI TV series (LT-style), and
eligibility analysis when requiring a minimum number of BIs per endpoint.

The adaptive rule compares each day's TV to a trailing baseline (excluding the
most recent days): a change requires the deviation to exceed BOTH an absolute
delta and a multiple of the baseline std, for `persistence` consecutive days.
"""

import statistics
from pathlib import Path

import fire
from plotly.subplots import make_subplots

from trackllm_website.bi.analyze import (
    compute_endpoint_tv_over_time,
    find_transitions,
    get_endpoint_legend_name,
)
from trackllm_website.bi.bi_quality import (
    ref_distributions,
    select_prompts,
    subsample,
)
from trackllm_website.bi.phase_2 import Timestamp
from trackllm_website.bi.sampling_sweep import (
    DAYS_AFTER,
    DAYS_BEFORE,
    THRESHOLD,
    add_series_traces,
    load_all_results,
)
from trackllm_website.config import config, logger

REPORT_DIR = Path("reports/bi_quality")
N_SAMPLES = 10
N_GRID = [5, 10, 20, 30, 40, 50]

WINDOW = 14
EXCLUSION = 4
MIN_BASELINE = 5
SIGMA_K = 4.0
ABS_DELTA = 0.2
PERSISTENCE = 3
COOLDOWN = 10


def adaptive_transitions(
    tv_over_time: list[tuple[Timestamp, float]],
) -> list[Timestamp]:
    """Detect changes as sustained deviations from a trailing baseline.

    A day deviates when |tv - baseline_mean| exceeds both ABS_DELTA and
    SIGMA_K * baseline_std; a change is declared after PERSISTENCE consecutive
    deviating days (dated at the first one). The baseline excludes the last
    EXCLUSION days so it is not contaminated during the persistence window.
    """
    timestamps = [ts for ts, _ in tv_over_time]
    vals = [v for _, v in tv_over_time]
    events: list[Timestamp] = []
    streak = 0
    last_event_idx: int | None = None

    for i in range(len(vals)):
        baseline = vals[max(0, i - EXCLUSION - WINDOW) : i - EXCLUSION]
        if len(baseline) < MIN_BASELINE:
            continue
        mean = statistics.mean(baseline)
        std = statistics.stdev(baseline)
        dev = abs(vals[i] - mean)
        if dev > ABS_DELTA and dev > SIGMA_K * std:
            streak += 1
        else:
            streak = 0
        if streak == PERSISTENCE:
            onset = i - PERSISTENCE + 1
            if last_event_idx is None or onset - last_event_idx >= COOLDOWN:
                events.append(timestamps[onset])
                last_event_idx = onset
    return events


def run_matrix(n_prompts: int) -> None:
    """Compare current vs adaptive rule × random vs balanced selection,
    restricted to endpoints with at least n_prompts BIs."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = load_all_results()
    eligible = {
        slug: results
        for slug, results in all_results.items()
        if len(ref_distributions(results)) >= n_prompts
    }
    logger.info(f"{len(eligible)}/{len(all_results)} endpoints with >= {n_prompts} BIs")

    conditions = [
        (rule, strategy)
        for rule in ["current", "adaptive"]
        for strategy in ["random", "balanced"]
    ]
    titles = []
    grid = []
    for rule, strategy in conditions:
        series: dict[str, list[tuple[Timestamp, float]]] = {}
        detected: dict[str, list[Timestamp]] = {}
        for slug, results in eligible.items():
            prompts = select_prompts(results, slug, strategy, n_prompts)
            tv = compute_endpoint_tv_over_time(
                subsample(results, slug, prompts, N_SAMPLES)
            )
            if not tv:
                continue
            series[slug] = tv
            if rule == "current":
                trans = find_transitions(tv, THRESHOLD, DAYS_BEFORE, DAYS_AFTER)
            else:
                trans = adaptive_transitions(tv)
            if trans:
                detected[slug] = trans
        label = f"{rule} rule, {strategy} selection"
        titles.append(f"{label} — {len(detected)} detected")
        grid.append((label, series, detected))
        print(f"{label}: {len(detected)} detected")
        for slug, trans in sorted(detected.items()):
            print(
                f"    {get_endpoint_legend_name(slug):55s} "
                + ", ".join(ts[:10] for ts in trans)
            )

    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, shared_xaxes=True)
    for i, (label, series, detected) in enumerate(grid):
        row, col = i // 2 + 1, i % 2 + 1
        add_series_traces(fig, series, detected, row=row, col=col)
    fig.update_layout(
        template=config.plotting.template,
        font_family=config.plotting.font_family,
        height=800,
        width=1400,
        font_size=11,
        margin=dict(l=50, r=20, t=50, b=30),
    )
    fig.update_yaxes(range=[0, 1.05])
    out = REPORT_DIR / f"rule_matrix_{n_prompts}_bis.png"
    fig.write_image(out)
    logger.info(f"Saved {out}")


def eligibility() -> None:
    """For each N, report eligible endpoints and detections at exactly N BIs."""
    all_results = load_all_results()
    for n in N_GRID:
        eligible = {
            slug: r for slug, r in all_results.items() if len(ref_distributions(r)) >= n
        }
        detected: dict[str, list[Timestamp]] = {}
        for slug, results in eligible.items():
            prompts = select_prompts(results, slug, "random", n)
            tv = compute_endpoint_tv_over_time(
                subsample(results, slug, prompts, N_SAMPLES)
            )
            trans = find_transitions(tv, THRESHOLD, DAYS_BEFORE, DAYS_AFTER)
            if trans:
                detected[slug] = trans
        names = ", ".join(
            f"{get_endpoint_legend_name(s)} ({', '.join(t[:10] for t in trans)})"
            for s, trans in sorted(detected.items())
        )
        print(
            f"N={n:2d}: {len(eligible):2d}/{len(all_results)} endpoints eligible, "
            f"{len(detected)} detected [current rule]: {names}"
        )


if __name__ == "__main__":
    fire.Fire({"eligibility": eligibility, "matrix": run_matrix})
