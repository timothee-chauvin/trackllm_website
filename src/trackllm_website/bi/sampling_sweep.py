"""Sweep the number of BIs and detection samples/BI on real phase 2 data.

For each (n_prompts, n_samples) config, recompute TV time series for all
endpoints via nested subsampling (so configs are directly comparable), run the
current detection rule, and report detections, noise floor and yearly cost.
"""

import random
import statistics
from pathlib import Path

import fire
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trackllm_website.bi.analyze import (
    TOL_MUTED,
    Results,
    compute_endpoint_tv_over_time,
    find_transitions,
    get_endpoint_legend_name,
    load_phase1_token_counts,
    load_phase2_results,
)
from trackllm_website.bi.phase_2 import Timestamp
from trackllm_website.config import config, logger
from trackllm_website.util import slugify

N_PROMPTS_GRID = [5, 10, 20, None]  # None = all available BIs
N_SAMPLES_GRID = [3, 10, 50]
SEED = 0
THRESHOLD = 0.5
DAYS_BEFORE = 4
DAYS_AFTER = 4
RUNS_PER_YEAR = 365
OUTPUT_TOKENS_PER_CALL = 1
REPORT_DIR = Path("reports/bi_sampling_sweep")


def subsample_nested(
    results: Results, slug: str, n_prompts: int | None, n_samples: int
) -> Results:
    """Subsample prompts and per-batch samples, nested across configs.

    Prompt and sample order comes from stable seeded shuffles, so e.g. the 5
    prompts used at n_prompts=5 are a subset of the 10 used at n_prompts=10.
    The reference batch (earliest timestamp) keeps all its samples.
    """
    prompts = sorted(results)
    random.Random(f"{SEED}:{slug}").shuffle(prompts)
    if n_prompts is not None:
        prompts = prompts[:n_prompts]

    all_timestamps = {ts for p in prompts for ts in results[p]}
    if not all_timestamps:
        return {}
    ref_ts = min(all_timestamps)

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


def load_all_results() -> dict[str, Results]:
    results = {}
    for d in sorted(config.bi.phase_2_dir.iterdir()):
        if not d.is_dir():
            continue
        r = load_phase2_results(d)
        if r:
            results[d.name] = r
    return results


def endpoint_costs_by_slug() -> dict[str, tuple[float, float]]:
    return {slugify(f"{e.model}#{e.provider}"): e.cost for e in config.endpoints_bi}


def yearly_cost(
    slug: str,
    n_prompts: int | None,
    n_samples: int,
    n_available: int,
    costs: dict[str, tuple[float, float]],
) -> float | None:
    """Yearly cost of daily monitoring for one endpoint, in USD."""
    if slug not in costs:
        return None
    token_counts = load_phase1_token_counts(slug)
    if not token_counts:
        return None
    avg_input_tokens = statistics.mean(token_counts.values())
    input_cost, output_cost = costs[slug]
    per_call = (
        avg_input_tokens * input_cost / 1e6 + OUTPUT_TOKENS_PER_CALL * output_cost / 1e6
    )
    n_used = n_available if n_prompts is None else min(n_prompts, n_available)
    return per_call * n_used * n_samples * RUNS_PER_YEAR


class ConfigResult:
    def __init__(self, n_prompts: int | None, n_samples: int):
        self.n_prompts = n_prompts
        self.n_samples = n_samples
        self.series: dict[str, list[tuple[Timestamp, float]]] = {}
        self.transitions: dict[str, list[Timestamp]] = {}
        self.avg_cost: float = 0.0

    @property
    def label(self) -> str:
        p = "all" if self.n_prompts is None else str(self.n_prompts)
        return f"{p} BIs × {self.n_samples} samples"

    @property
    def detected(self) -> dict[str, list[Timestamp]]:
        return {s: t for s, t in self.transitions.items() if t}

    def noise_floor(self) -> tuple[float, float]:
        """Median over non-detected endpoints of (mean TV, std TV)."""
        means, stds = [], []
        for slug, tv in self.series.items():
            if self.transitions.get(slug) or len(tv) < 10:
                continue
            vals = [v for _, v in tv]
            means.append(statistics.mean(vals))
            stds.append(statistics.stdev(vals))
        if not means:
            return float("nan"), float("nan")
        return statistics.median(means), statistics.median(stds)


def run_config(
    all_results: dict[str, Results],
    costs: dict[str, tuple[float, float]],
    n_prompts: int | None,
    n_samples: int,
) -> ConfigResult:
    res = ConfigResult(n_prompts, n_samples)
    cost_values = []
    for slug, results in all_results.items():
        sub = subsample_nested(results, slug, n_prompts, n_samples)
        tv = compute_endpoint_tv_over_time(sub)
        if not tv:
            continue
        res.series[slug] = tv
        res.transitions[slug] = find_transitions(tv, THRESHOLD, DAYS_BEFORE, DAYS_AFTER)
        cost = yearly_cost(slug, n_prompts, n_samples, len(results), costs)
        if cost is not None:
            cost_values.append(cost)
    res.avg_cost = statistics.mean(cost_values) if cost_values else float("nan")
    return res


def add_series_traces(
    fig: go.Figure,
    series: dict[str, list[tuple[Timestamp, float]]],
    detected: dict[str, list[Timestamp]],
    row=None,
    col=None,
) -> None:
    for slug, tv in series.items():
        if detected.get(slug):
            continue
        timestamps, tv_values = zip(*tv)
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=tv_values,
                mode="lines",
                line=dict(color="lightgrey", width=1),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    for color_idx, slug in enumerate(sorted(detected)):
        timestamps, tv_values = zip(*series[slug])
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=tv_values,
                mode="lines+markers",
                name=get_endpoint_legend_name(slug),
                line_color=TOL_MUTED[color_idx % len(TOL_MUTED)],
                marker_size=4,
                showlegend=row is None,
            ),
            row=row,
            col=col,
        )


def plot_grid(config_results: list[ConfigResult], output_path: Path) -> None:
    rows, cols = len(N_PROMPTS_GRID), len(N_SAMPLES_GRID)
    titles = [
        f"{r.label} — {len(r.detected)} detected, avg ${r.avg_cost:.2f}/yr"
        for r in config_results
    ]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, shared_xaxes=True)
    for i, res in enumerate(config_results):
        row, col = i // cols + 1, i % cols + 1
        add_series_traces(fig, res.series, res.detected, row=row, col=col)
        fig.add_hline(
            y=THRESHOLD,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            row=row,
            col=col,
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
    fig.write_image(output_path)
    logger.info(f"Saved {output_path}")


def plot_single(res: ConfigResult, output_path: Path) -> None:
    fig = go.Figure()
    add_series_traces(fig, res.series, res.detected)
    fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        template=config.plotting.template,
        font_family=config.plotting.font_family,
        title=f"{res.label} — avg ${res.avg_cost:.2f}/endpoint/yr (daily monitoring)",
        yaxis_title="Mean TV distance from initialization",
        height=500,
        width=1300,
        font_size=13,
        yaxis=dict(range=[0, 1.05]),
        legend=dict(font_size=11),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.write_image(output_path)
    logger.info(f"Saved {output_path}")


def write_report(config_results: list[ConfigResult], report_path: Path) -> None:
    lines = [
        "# BI sampling sweep: number of BIs × detection samples/BI",
        "",
        f"Detection rule held fixed (current): TV ≥ {THRESHOLD} for {DAYS_AFTER} consecutive days "
        f"after {DAYS_BEFORE} consecutive days below, plus a ≥ {THRESHOLD} single-day jump.",
        "Reference = full first batch (~100 samples). Subsampling is nested: "
        "smaller configs use subsets of larger ones. Costs assume daily monitoring, "
        "1 output token/call, averaged over endpoints with cost data.",
        "",
        "## Summary",
        "",
        "| Config | Avg cost/endpoint/yr | Endpoints detected | Noise floor (median TV) | Noise (median std) |",
        "|---|---|---|---|---|",
    ]
    for r in config_results:
        nf_mean, nf_std = r.noise_floor()
        lines.append(
            f"| {r.label} | ${r.avg_cost:.2f} | {len(r.detected)} | {nf_mean:.3f} | {nf_std:.3f} |"
        )

    all_detected = sorted({s for r in config_results for s in r.detected})
    lines += [
        "",
        "## Detections per config",
        "",
        "| Endpoint | " + " | ".join(r.label for r in config_results) + " |",
        "|---|" + "---|" * len(config_results),
    ]
    for slug in all_detected:
        cells = []
        for r in config_results:
            dates = r.detected.get(slug)
            cells.append(", ".join(ts[:10] for ts in dates) if dates else "—")
        lines.append(
            f"| {get_endpoint_legend_name(slug)} | " + " | ".join(cells) + " |"
        )

    lines += [
        "",
        "## Plots",
        "",
        "Overview grid: `sweep_grid.png`",
        "",
    ]
    for r in config_results:
        lines.append(f"- {r.label}: `{slugify(r.label)}.png`")
    report_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Saved {report_path}")


def sweep() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = load_all_results()
    costs = endpoint_costs_by_slug()
    logger.info(f"Loaded {len(all_results)} endpoints")

    config_results = []
    for n_prompts in N_PROMPTS_GRID:
        for n_samples in N_SAMPLES_GRID:
            res = run_config(all_results, costs, n_prompts, n_samples)
            config_results.append(res)
            logger.info(
                f"{res.label}: {len(res.detected)} detected, avg ${res.avg_cost:.2f}/yr"
            )

    plot_grid(config_results, REPORT_DIR / "sweep_grid.png")
    for res in config_results:
        plot_single(res, REPORT_DIR / f"{slugify(res.label)}.png")
    write_report(config_results, REPORT_DIR / "report.md")


if __name__ == "__main__":
    fire.Fire(sweep)
