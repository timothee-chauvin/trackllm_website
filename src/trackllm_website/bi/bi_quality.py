"""Noise-floor decomposition and BI-quality (phase 1b) simulation on phase 2 data.

`noise`: compare the observed TV noise floor of stable endpoints against the
binomial sampling-noise prediction, and against within-day split-half TV, to
separate sampling noise from day-to-day drift.

`quality`: simulate phase 1b — select BIs by top-2 balance (p2/p1, estimated
from the reference batch) instead of at random, and compare.
"""

import math
import random
import statistics
from collections import Counter
from pathlib import Path

import fire
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trackllm_website.bi.analyze import (
    Results,
    compute_endpoint_tv_over_time,
    compute_tv_distance,
    find_transitions,
    get_distribution,
    get_endpoint_legend_name,
)
from trackllm_website.bi.phase_2 import Prompt, Timestamp
from trackllm_website.bi.sampling_sweep import (
    DAYS_AFTER,
    DAYS_BEFORE,
    SEED,
    THRESHOLD,
    add_series_traces,
    load_all_results,
)
from trackllm_website.config import config, logger

REPORT_DIR = Path("reports/bi_quality")
N_SAMPLES = 10
N_PROMPTS_GRID = [5, 10, 20]
STRATEGIES = ["balanced", "random", "unbalanced"]
MIN_REF_SAMPLES = 20


def ref_timestamp(results: Results) -> Timestamp | None:
    all_ts = {ts for batches in results.values() for ts in batches}
    return min(all_ts) if all_ts else None


def ref_distributions(results: Results) -> dict[Prompt, Counter]:
    ref_ts = ref_timestamp(results)
    if ref_ts is None:
        return {}
    return {
        p: get_distribution(batches[ref_ts])
        for p, batches in results.items()
        if ref_ts in batches and len(batches[ref_ts]) >= MIN_REF_SAMPLES
    }


def predicted_tv(dist: Counter, m: int, n: int) -> float:
    """Expected TV between empirical distributions of m and n samples of `dist`,
    under no change (Gaussian approximation, plug-in probabilities)."""
    total = sum(dist.values())
    probs = [c / total for c in dist.values()]
    return (
        0.5
        * math.sqrt(2 / math.pi)
        * math.sqrt(1 / m + 1 / n)
        * sum(math.sqrt(p * (1 - p)) for p in probs)
    )


def balance_score(dist: Counter) -> float:
    counts = sorted(dist.values(), reverse=True)
    return counts[1] / counts[0] if len(counts) > 1 else 0.0


def select_prompts(
    results: Results, slug: str, strategy: str, n_prompts: int
) -> list[Prompt]:
    refs = ref_distributions(results)
    prompts = sorted(refs)
    if strategy == "random":
        random.Random(f"{SEED}:{slug}").shuffle(prompts)
    else:
        reverse = strategy == "balanced"
        prompts.sort(key=lambda p: balance_score(refs[p]), reverse=reverse)
    return prompts[:n_prompts]


def subsample(
    results: Results, slug: str, prompts: list[Prompt], n_samples: int
) -> Results:
    """Keep the reference batch intact, subsample other batches to n_samples."""
    ref_ts = ref_timestamp(results)
    out: Results = {}
    for p in prompts:
        out[p] = {}
        for ts, samples in results[p].items():
            if ts != ref_ts and len(samples) > n_samples:
                samples = list(samples)
                random.Random(f"{SEED}:{slug}:{p}:{ts}").shuffle(samples)
                samples = samples[:n_samples]
            out[p][ts] = samples
    return out


def noise() -> None:
    """Decompose the TV noise floor: predicted sampling noise vs observed."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = load_all_results()

    rows = []
    for slug, results in all_results.items():
        refs = ref_distributions(results)
        if not refs:
            continue
        ref_ts = ref_timestamp(results)
        m = statistics.mean(sum(d.values()) for d in refs.values())

        sub = subsample(results, slug, sorted(refs), 50)
        tv = compute_endpoint_tv_over_time(sub)
        if len(tv) < 10:
            continue
        vals = [v for _, v in tv]
        if max(vals) >= THRESHOLD:
            continue  # endpoint may have changed; keep only clearly stable ones

        predicted = statistics.mean(predicted_tv(d, round(m), 50) for d in refs.values())
        observed = statistics.median(vals)

        within_days: list[float] = []
        for ts in sorted({t for b in results.values() for t in b}):
            if ts == ref_ts:
                continue
            per_prompt = []
            for p in refs:
                samples = results[p].get(ts, [])
                if len(samples) < 50:
                    continue
                samples = list(samples)
                random.Random(f"{SEED}:{slug}:{p}:{ts}:split").shuffle(samples)
                tv_split = compute_tv_distance(
                    get_distribution(samples[:25]), get_distribution(samples[25:50])
                )
                if tv_split is not None:
                    per_prompt.append(tv_split)
            if per_prompt:
                within_days.append(statistics.mean(per_prompt))
        if not within_days:
            continue
        predicted_within = statistics.mean(
            predicted_tv(d, 25, 25) for d in refs.values()
        )
        rows.append(
            {
                "slug": slug,
                "n_bis": len(refs),
                "predicted": predicted,
                "observed": observed,
                "predicted_within": predicted_within,
                "observed_within": statistics.median(within_days),
            }
        )

    med = lambda key: statistics.median(r[key] for r in rows)  # noqa: E731
    print(f"{len(rows)} stable endpoints (max TV < {THRESHOLD} at 50 samples/day)")
    print(f"day-vs-reference TV @ n=50, m~100: predicted {med('predicted'):.3f}, observed {med('observed'):.3f}")
    print(f"within-day split-half TV @ 25v25:  predicted {med('predicted_within'):.3f}, observed {med('observed_within'):.3f}")
    print()
    print("largest observed-minus-predicted gaps (drifting endpoints):")
    for r in sorted(rows, key=lambda r: r["predicted"] - r["observed"])[:10]:
        print(
            f"  {get_endpoint_legend_name(r['slug']):55s} n_bis={r['n_bis']:3d} "
            f"predicted={r['predicted']:.3f} observed={r['observed']:.3f} "
            f"within-day={r['observed_within']:.3f}"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[r["predicted"] for r in rows],
            y=[r["observed"] for r in rows],
            mode="markers",
            text=[get_endpoint_legend_name(r["slug"]) for r in rows],
            marker=dict(size=8, color=[r["n_bis"] for r in rows], colorscale="Viridis",
                        colorbar=dict(title="# BIs"), cmin=0),
            showlegend=False,
        )
    )
    lim = max(max(r["predicted"] for r in rows), max(r["observed"] for r in rows)) * 1.1
    fig.add_trace(
        go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                   line=dict(color="grey", dash="dash"), showlegend=False)
    )
    fig.update_layout(
        template=config.plotting.template,
        font_family=config.plotting.font_family,
        title="Stable endpoints: observed TV floor vs binomial sampling-noise prediction (50 samples/day)",
        xaxis_title="Predicted E[TV] (sampling noise only)",
        yaxis_title="Observed median TV",
        width=800,
        height=700,
        font_size=13,
    )
    out = REPORT_DIR / "noise_floor_predicted_vs_observed.png"
    fig.write_image(out)
    logger.info(f"Saved {out}")


def quality() -> None:
    """Compare BI selection strategies (phase 1b simulation) at N_SAMPLES/day."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = load_all_results()

    rows, cols = len(STRATEGIES), len(N_PROMPTS_GRID)
    titles = []
    grid = []
    for strategy in STRATEGIES:
        for n_prompts in N_PROMPTS_GRID:
            series: dict[str, list[tuple[Timestamp, float]]] = {}
            transitions: dict[str, list[Timestamp]] = {}
            for slug, results in all_results.items():
                prompts = select_prompts(results, slug, strategy, n_prompts)
                if not prompts:
                    continue
                sub = subsample(results, slug, prompts, N_SAMPLES)
                tv = compute_endpoint_tv_over_time(sub)
                if not tv:
                    continue
                series[slug] = tv
                transitions[slug] = find_transitions(
                    tv, THRESHOLD, DAYS_BEFORE, DAYS_AFTER
                )
            detected = {s: t for s, t in transitions.items() if t}
            stable_means = [
                statistics.mean(v for _, v in tv)
                for slug, tv in series.items()
                if not transitions.get(slug) and len(tv) >= 10
            ]
            stable_stds = [
                statistics.stdev(v for _, v in tv)
                for slug, tv in series.items()
                if not transitions.get(slug) and len(tv) >= 10
            ]
            label = f"{strategy}, {n_prompts} BIs × {N_SAMPLES} samples"
            titles.append(f"{label} — {len(detected)} detected")
            grid.append((label, series, transitions, detected))
            print(
                f"{label}: {len(detected)} detected, "
                f"floor={statistics.median(stable_means):.3f}, "
                f"jitter={statistics.median(stable_stds):.3f}"
            )
            for slug, trans in sorted(detected.items()):
                print(
                    f"    {get_endpoint_legend_name(slug):55s} "
                    + ", ".join(ts[:10] for ts in trans)
                )

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, shared_xaxes=True)
    for i, (label, series, transitions, detected) in enumerate(grid):
        row, col = i // cols + 1, i % cols + 1
        add_series_traces(fig, series, detected, row=row, col=col)
        fig.add_hline(
            y=THRESHOLD, line_dash="dash", line_color="gray", opacity=0.5,
            row=row, col=col,
        )
    fig.update_layout(
        template=config.plotting.template,
        font_family=config.plotting.font_family,
        height=350 * rows,
        width=550 * cols,
        font_size=11,
        margin=dict(l=50, r=20, t=50, b=30),
    )
    fig.update_yaxes(range=[0, 1.05])
    out = REPORT_DIR / "quality_grid.png"
    fig.write_image(out)
    logger.info(f"Saved {out}")


if __name__ == "__main__":
    fire.Fire({"noise": noise, "quality": quality})
