"""Analyze phase 2 results and plot TV distance over time."""

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import orjson
import plotly.graph_objects as go

from trackllm_website.bi.phase_2 import (
    Prompt,
    ResponseToken,
    Timestamp,
    load_border_inputs,
    load_existing_results,
)
from trackllm_website.config import config, logger
from trackllm_website.util import endpoint_from_slug, slugify

Results = dict[Prompt, dict[Timestamp, list[list[str]]]]

# Paul Tol's muted color scheme
TOL_MUTED = [
    "#CC6677",
    "#332288",
    "#DDCC77",
    "#117733",
    "#88CCEE",
    "#882255",
    "#44AA99",
    "#999933",
    "#AA4499",
]


def get_endpoint_legend_name(slug: str) -> str:
    endpoint = endpoint_from_slug(slug)
    model_without_creator = endpoint.model.split("/")[-1]
    return f"{model_without_creator} ({endpoint.provider})"


def load_phase2_results(
    endpoint_dir: Path,
    max_prompts: int | None = None,
    max_samples_per_timestamp: int | None = None,
    seed: int | None = None,
) -> Results:
    """Load all phase 2 results for an endpoint across all months."""
    combined: Results = {}
    for json_file in endpoint_dir.glob("*.json"):
        data = load_existing_results(json_file)
        for prompt, batches in data.items():
            if prompt not in combined:
                combined[prompt] = {}
            combined[prompt].update(batches)

    if not combined:
        return combined

    no_filtering = max_prompts is None and max_samples_per_timestamp is None
    if no_filtering:
        return combined

    rng = random.Random(seed)

    if max_prompts is not None and len(combined) > max_prompts:
        selected_prompts = rng.sample(list(combined.keys()), max_prompts)
        combined = {p: combined[p] for p in selected_prompts}

    if max_samples_per_timestamp is not None:
        for batches in combined.values():
            for timestamp, samples in batches.items():
                if len(samples) > max_samples_per_timestamp:
                    batches[timestamp] = rng.sample(samples, max_samples_per_timestamp)

    return combined


def compute_tv_distance(
    dist_p: Counter[ResponseToken], dist_q: Counter[ResponseToken]
) -> float | None:
    """Compute total variation distance between two distributions."""
    all_tokens = set(dist_p.keys()) | set(dist_q.keys())
    total_p = sum(dist_p.values())
    total_q = sum(dist_q.values())
    if total_p == 0 or total_q == 0:
        return None
    tv = 0.0
    for token in all_tokens:
        p_prob = dist_p[token] / total_p
        q_prob = dist_q[token] / total_q
        tv += abs(p_prob - q_prob)
    return tv / 2


def get_distribution(
    responses: Sequence[Sequence[str]],
) -> Counter[str]:
    """Get token distribution from responses (each item is (timestamp, token))."""
    return Counter(token for _, token in responses)


def compute_endpoint_tv_over_time(
    results: Results,
    reference_samples: int | None = None,
    test_samples: int | None = None,
    seed: int | None = None,
) -> list[tuple[Timestamp, float]]:
    """Compute average TV distance over time for an endpoint.

    Args:
        results: Full results dict (no sample filtering applied).
        reference_samples: Number of samples to use for the reference (first timestamp).
        test_samples: Number of samples to use for each test timestamp.
        seed: Random seed for reproducible subsampling.

    Returns list of (timestamp, avg_tv_distance) sorted by timestamp.
    """
    all_timestamps: set[Timestamp] = set()
    for batches in results.values():
        all_timestamps.update(batches.keys())
    sorted_timestamps = sorted(all_timestamps)

    if len(sorted_timestamps) < 2:
        return []

    rng = random.Random(seed)
    init_timestamp = sorted_timestamps[0]
    tv_over_time: list[tuple[Timestamp, float]] = []

    for timestamp in sorted_timestamps[1:]:
        tv_distances: list[float] = []
        for prompt, batches in results.items():
            if init_timestamp not in batches or timestamp not in batches:
                continue

            init_samples = batches[init_timestamp]
            curr_samples = batches[timestamp]

            if reference_samples is not None and len(init_samples) > reference_samples:
                init_samples = rng.sample(init_samples, reference_samples)
            if test_samples is not None and len(curr_samples) > test_samples:
                curr_samples = rng.sample(curr_samples, test_samples)

            init_dist = get_distribution(init_samples)
            curr_dist = get_distribution(curr_samples)
            tv = compute_tv_distance(init_dist, curr_dist)
            if tv is not None:
                tv_distances.append(tv)

        if tv_distances:
            avg_tv = sum(tv_distances) / len(tv_distances)
            tv_over_time.append((timestamp, avg_tv))

    return tv_over_time


def plot_tv_distance_over_time(
    output_path: Path | None = None,
    min_tv_threshold: float | None = 0.4,
    restrict_to_endpoints: list[str] | None = None,
) -> None:
    """Plot TV distance over time for all endpoints.

    Args:
        output_path: Path to save the figure. If None, displays interactively.
        min_tv_threshold: Only include endpoints where TV ever exceeds this value.
            Set to None to include all endpoints.
        restrict_to_endpoints: Only include endpoints in this list. If None, include all endpoints.
    """
    phase_2_dir = config.bi.phase_2_dir

    fig = go.Figure()
    endpoint_dirs = sorted(phase_2_dir.iterdir())

    if restrict_to_endpoints:
        all_slugs = {d.name for d in endpoint_dirs if d.is_dir()}
        restrict_slugs = {slugify(e) for e in restrict_to_endpoints}
        unknown = restrict_slugs - all_slugs
        if unknown:
            raise ValueError(f"Unknown endpoints: {unknown}")
    else:
        restrict_slugs = None

    for endpoint_dir in endpoint_dirs:
        if not endpoint_dir.is_dir():
            continue

        if restrict_slugs and endpoint_dir.name not in restrict_slugs:
            continue

        results = load_phase2_results(endpoint_dir)
        if not results:
            logger.warning(f"No results found for {endpoint_dir.name}")

        tv_over_time = compute_endpoint_tv_over_time(results)

        timestamps, tv_values = zip(*tv_over_time)

        if min_tv_threshold is not None and max(tv_values) < min_tv_threshold:
            continue

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=tv_values,
                mode="lines+markers",
                name=get_endpoint_legend_name(endpoint_dir.name),
            )
        )

    fig.update_layout(
        template=config.plotting.template,
        font_family=config.plotting.font_family,
        xaxis_title="Time",
        yaxis_title="Average TV Distance from Initialization",
        font_size=14,
        height=600,
        width=1000,
        yaxis=dict(range=[0, 1]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
        margin=dict(l=60, r=300, t=40, b=60),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(output_path)
        print(f"Saved figure to {output_path}")
    else:
        fig.show()


def plot_tv_distance_over_time_one(
    output_path: Path | None = None,
) -> None:
    """Interactively select an endpoint via fuzzy finder, then plot its TV over time."""
    import shutil
    import subprocess

    phase_2_dir = config.bi.phase_2_dir
    slugs = sorted(d.name for d in phase_2_dir.iterdir() if d.is_dir())
    if not slugs:
        logger.warning("No endpoint directories found")
        return

    fuzzy_finder = shutil.which("fzf") or shutil.which("sk")
    if not fuzzy_finder:
        raise RuntimeError("fzf or sk must be installed")

    result = subprocess.run(
        [fuzzy_finder],
        input="\n".join(slugs),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return

    selected = result.stdout.strip()
    plot_tv_distance_over_time(
        output_path=output_path,
        min_tv_threshold=None,
        restrict_to_endpoints=[selected],
    )


def find_transitions(
    tv_over_time: list[tuple[Timestamp, float]],
    threshold: float,
    days_before: int,
    days_after: int,
    min_jump: float = 0.5,
) -> list[Timestamp]:
    """Find all transition points where TV crosses the threshold sustainably.

    A transition requires `days_before` consecutive days below threshold followed by
    `days_after` consecutive days above (or vice versa), plus at least one single-day
    jump of `min_jump` anywhere in the series.

    Returns list of timestamps at each crossover point.
    """
    if len(tv_over_time) < days_before + days_after:
        return []

    tv_values = [tv for _, tv in tv_over_time]
    timestamps = [ts for ts, _ in tv_over_time]

    max_jump = max(tv_values[i] - tv_values[i - 1] for i in range(1, len(tv_values)))
    if max_jump < min_jump:
        return []

    # Must have at least one below→above transition to qualify
    has_upward = False
    transitions = []
    for i in range(len(tv_values) - days_before - days_after + 1):
        window_before = tv_values[i : i + days_before]
        window_after = tv_values[i + days_before : i + days_before + days_after]
        below_then_above = all(tv < threshold for tv in window_before) and all(
            tv >= threshold for tv in window_after
        )
        above_then_below = all(tv >= threshold for tv in window_before) and all(
            tv < threshold for tv in window_after
        )
        if below_then_above:
            has_upward = True
            transitions.append(timestamps[i + days_before])
        elif above_then_below:
            transitions.append(timestamps[i + days_before])

    return transitions if has_upward else []


def plot_tv_distance_transitions(
    output_path: Path | None = None,
    threshold: float = 0.5,
    days_before: int = 4,
    days_after: int = 4,
    max_prompts: int = 5,
    reference_samples: int = 50,
    transition_lines: Literal["none", "grey", "colored"] = "none",
    test_samples: int = 3,
    show_others: bool = False,
) -> go.Figure:
    """Plot TV distance for endpoints showing a clear transition pattern.

    Only includes endpoints with at least `days_before` consecutive days below
    `threshold` followed by at least `days_after` consecutive days above `threshold`.
    If `show_others`, endpoints without transitions are drawn as light grey
    background traces.
    """
    phase_2_dir = config.bi.phase_2_dir

    fig = go.Figure()
    endpoint_dirs = sorted(phase_2_dir.iterdir())
    color_idx = 0
    colors = TOL_MUTED

    series: list[tuple[str, list[tuple[Timestamp, float]], list[Timestamp]]] = []
    for endpoint_dir in endpoint_dirs:
        if not endpoint_dir.is_dir():
            continue

        results = load_phase2_results(endpoint_dir, max_prompts=max_prompts, seed=0)
        if not results:
            continue

        tv_over_time = compute_endpoint_tv_over_time(
            results,
            reference_samples=reference_samples,
            test_samples=test_samples,
            seed=0,
        )
        if not tv_over_time:
            continue

        transitions = find_transitions(tv_over_time, threshold, days_before, days_after)
        series.append((endpoint_dir.name, tv_over_time, transitions))

    if show_others:
        for name, tv_over_time, transitions in series:
            if transitions:
                continue
            timestamps, tv_values = zip(*tv_over_time)
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=tv_values,
                    mode="lines",
                    name=get_endpoint_legend_name(name),
                    line=dict(color="lightgrey", width=1),
                    showlegend=False,
                )
            )

    for name, tv_over_time, transitions in series:
        if not transitions:
            continue

        timestamps, tv_values = zip(*tv_over_time)
        print(
            f"{name}: {len(tv_over_time)} days of detection + 1 reference day, "
            f"{len(transitions)} change(s): {', '.join(ts[:10] for ts in transitions)}"
        )

        color = colors[color_idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=tv_values,
                mode="lines+markers",
                name=get_endpoint_legend_name(name),
                line_color=color,
            )
        )
        if transition_lines != "none":
            for ts in transitions:
                vline_color = color if transition_lines == "colored" else "grey"
                fig.add_shape(
                    type="line",
                    x0=ts,
                    x1=ts,
                    y0=0,
                    y1=1.05,
                    line=dict(color=vline_color, width=2, dash="dash"),
                    opacity=0.4 if transition_lines == "colored" else 0.3,
                )
        color_idx += 1

    fig.add_hline(y=threshold, line_dash="dash", line_color="gray", opacity=0.5)

    # Sort by name length ascending. Works well with data as of Mar 31 2026.
    # Grey background traces (showlegend=False) stay first so colored ones draw on top.
    fig.data = sorted(
        fig.data, key=lambda t: (t.showlegend is not False, len(t.name or ""))
    )

    fig.update_layout(
        template=config.plotting.template,
        font_family=config.plotting.font_family,
        yaxis_title="Mean TV Distance from Initialization",
        font_size=25,
        height=650,
        width=800,
        yaxis=dict(range=[0, 1.05], title_standoff=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0, 0, 0, 0)",
            font_size=17,
        ),
        margin=dict(l=0, r=0, t=120, b=70),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(output_path)
        print(f"Saved figure to {output_path}")
    else:
        fig.show()
    return fig


def plot_tv_distance_heatmap(
    output_path: Path | None = None,
    tv_threshold: float = 0.4,
) -> None:
    """Plot TV distance as a binary heatmap grid with models on Y-axis and days on X-axis.

    Args:
        output_path: Path to save the figure. If None, displays interactively.
        tv_threshold: Threshold for binary coloring. White if TV < threshold, red otherwise.
    """
    phase_2_dir = config.bi.phase_2_dir
    endpoint_dirs = sorted(phase_2_dir.iterdir())

    # Collect all data first
    endpoint_data: dict[str, list[tuple[Timestamp, float]]] = {}
    all_timestamps: set[Timestamp] = set()

    for endpoint_dir in endpoint_dirs:
        if not endpoint_dir.is_dir():
            continue

        results = load_phase2_results(endpoint_dir)
        if not results:
            logger.warning(f"No results found for {endpoint_dir.name}")
            continue

        tv_over_time = compute_endpoint_tv_over_time(results)
        if not tv_over_time:
            continue

        timestamps, _ = zip(*tv_over_time)

        endpoint_data[endpoint_dir.name] = tv_over_time
        all_timestamps.update(timestamps)

    if not endpoint_data:
        logger.warning("No endpoint data found for heatmap")
        return

    # Sort timestamps
    sorted_timestamps = sorted(all_timestamps)

    # Build the binary heatmap matrix and count red points per endpoint
    endpoint_red_counts: dict[str, int] = {}
    endpoint_rows: dict[str, list[int | None]] = {}

    for endpoint, tv_over_time in endpoint_data.items():
        tv_dict = dict(tv_over_time)
        row: list[int | None] = []
        red_count = 0
        for ts in sorted_timestamps:
            tv = tv_dict.get(ts)
            if tv is None:
                row.append(None)
            elif tv >= tv_threshold:
                row.append(1)
                red_count += 1
            else:
                row.append(0)
        endpoint_rows[endpoint] = row
        endpoint_red_counts[endpoint] = red_count

    # Sort endpoints by number of red points (decreasing)
    sorted_endpoints = sorted(
        endpoint_data.keys(), key=lambda e: endpoint_red_counts[e], reverse=True
    )

    # Build final matrix in sorted order
    z_matrix = [endpoint_rows[endpoint] for endpoint in sorted_endpoints]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=sorted_timestamps,
            y=list(range(len(sorted_endpoints))),
            colorscale=[[0, "white"], [1, "red"]],
            zmin=0,
            zmax=1,
            showscale=False,
            hoverongaps=False,
            xgap=1,
            ygap=1,
        )
    )

    fig.update_layout(
        template=config.plotting.template,
        font_family=config.plotting.font_family,
        xaxis_title="Time",
        yaxis_title="Model",
        font_size=14,
        height=max(400, 20 * len(sorted_endpoints)),
        width=1000,
        margin=dict(l=60, r=60, t=40, b=60),
        yaxis=dict(showticklabels=False),
        plot_bgcolor="lightgrey",
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(output_path)
        print(f"Saved figure to {output_path}")
    else:
        fig.show()


def load_phase1_token_counts(slug: str, temperature: float = 0.0) -> dict[str, int]:
    """Load actual input token counts from phase 1 results.

    Returns dict mapping prompt -> input_token_count.
    """
    phase_1_dir = config.bi.get_phase_1_dir(temperature)
    json_path = phase_1_dir / f"{slug}.json"
    if not json_path.exists():
        return {}
    with open(json_path, "rb") as f:
        data = orjson.loads(f.read())
    prompt_to_tokens: dict[str, int] = {}
    for token_count_str, prompts_dict in data.items():
        token_count = int(token_count_str)
        for prompt in prompts_dict:
            prompt_to_tokens[prompt] = token_count
    return prompt_to_tokens


def get_prompt_stats() -> dict[str, dict]:
    """Get prompt statistics from border_inputs.json.

    Returns dict mapping endpoint key to stats (prompt_count, prompts).
    """
    border_inputs = load_border_inputs(temperature=0.0)
    stats = {}
    for endpoint_key, prompts in border_inputs.items():
        if not prompts:
            continue
        stats[endpoint_key] = {
            "prompt_count": len(prompts),
            "prompts": prompts,
        }
    return stats


def compute_yearly_monitoring_cost(
    prompts_per_endpoint: int = 5,
    samples_per_prompt: int = 3,
    runs_per_year: int = 8760,  # hourly
    output_tokens_per_call: int = 1,
) -> dict[str, dict]:
    """Compute yearly cost of BI monitoring for all phase_2 endpoints.

    Returns dict mapping endpoint slug to cost info dict.
    """
    phase_2_dir = config.bi.phase_2_dir
    endpoint_slugs = [d.name for d in phase_2_dir.iterdir() if d.is_dir()]

    prompt_stats = get_prompt_stats()

    endpoint_costs: dict[str, tuple[float, float]] = {}
    endpoint_keys: dict[str, str] = {}
    for ep in config.endpoints_bi:
        slug = slugify(f"{ep.model}#{ep.provider}")
        endpoint_costs[slug] = ep.cost
        endpoint_keys[slug] = str(ep)

    calls_per_run = prompts_per_endpoint * samples_per_prompt
    calls_per_year = calls_per_run * runs_per_year

    results: dict[str, dict] = {}
    total_cost = 0.0

    for slug in sorted(endpoint_slugs):
        if slug not in endpoint_costs:
            logger.warning(f"No cost info for endpoint: {slug}")
            continue

        endpoint_key = endpoint_keys.get(slug)
        stats = prompt_stats.get(endpoint_key, {})
        prompt_count = stats.get("prompt_count", 0)
        prompts = stats.get("prompts", [])

        token_counts = load_phase1_token_counts(slug)
        input_tokens = [token_counts.get(p) for p in prompts]
        valid_token_counts = [t for t in input_tokens if t is not None]
        if not valid_token_counts:
            logger.warning(f"No token count data for {slug}, skipping")
            continue
        avg_input_tokens = sum(valid_token_counts) / len(valid_token_counts)

        input_cost, output_cost = endpoint_costs[slug]
        cost_per_call = (
            avg_input_tokens * input_cost / 1e6
            + output_tokens_per_call * output_cost / 1e6
        )
        yearly_cost = cost_per_call * calls_per_year

        results[slug] = {
            "yearly_cost": yearly_cost,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "prompt_count": prompt_count,
            "avg_input_tokens": avg_input_tokens,
        }
        total_cost += yearly_cost

    results["__total__"] = {"yearly_cost": total_cost}
    return results


def print_yearly_monitoring_cost() -> None:
    """Print yearly monitoring cost breakdown."""
    costs = compute_yearly_monitoring_cost()
    total_info = costs.pop("__total__")
    total = total_info["yearly_cost"]

    input_costs = [c["input_cost"] for c in costs.values()]
    output_costs = [c["output_cost"] for c in costs.values()]
    avg_input = sum(input_costs) / len(input_costs) if input_costs else 0
    avg_output = sum(output_costs) / len(output_costs) if output_costs else 0

    avg_input_tokens_list = [c["avg_input_tokens"] for c in costs.values()]
    avg_input_tokens = (
        sum(avg_input_tokens_list) / len(avg_input_tokens_list)
        if avg_input_tokens_list
        else 0
    )

    prompt_counts = [c["prompt_count"] for c in costs.values() if c["prompt_count"] > 0]
    avg_prompts = sum(prompt_counts) / len(prompt_counts) if prompt_counts else 0

    print("Yearly monitoring cost (5 prompts × 3 samples, hourly):")
    print(f"{'Endpoint':<60} {'Cost (USD)':>12}")
    print("-" * 74)
    for slug, info in sorted(costs.items(), key=lambda x: -x[1]["yearly_cost"]):
        print(f"{slug:<60} ${info['yearly_cost']:>10.2f}")
    print("-" * 74)
    print(f"{'TOTAL':<60} ${total:>10.2f}")
    print(f"Average: ${total / len(costs):>10.2f}")
    print(f"\nNumber of endpoints: {len(costs)}")
    print(f"Average cost per Mtokens: input=${avg_input:.3f}, output=${avg_output:.3f}")
    print(f"Average input tokens per prompt: {avg_input_tokens:.1f}")
    print(f"Average prompts per endpoint: {avg_prompts:.1f}")


def compute_phase1_cost(
    target_bis: int,
    samples_per_prompt: int,
    reference_samples: int,
    output_tokens_per_call: int = 1,
) -> dict[str, dict]:
    """Compute phase 1 initialization cost for all endpoints in phase_1/T=0.

    Phase 1 cost has two components:
    - Phase 1a: Query prompts until finding target_bis border inputs
    - Reference establishment: Sample each BI reference_samples times
    """
    phase_1_dir = config.bi.get_phase_1_dir(0.0)

    from trackllm_website.bi.state import load_all_states

    endpoint_costs: dict[str, tuple[float, float]] = {}
    # Historical endpoints may be absent from config.endpoints_bi but still
    # present in the state-file registry.
    for slug, state in load_all_states(config.bi.state_dir).items():
        endpoint_costs[slug] = state.endpoint.cost
    for ep in config.endpoints_bi:
        slug = slugify(f"{ep.model}#{ep.provider}")
        endpoint_costs[slug] = ep.cost

    results: dict[str, dict] = {}
    total_cost = 0.0

    for json_path in sorted(phase_1_dir.glob("*.json")):
        if json_path.name == "border_inputs.json":
            continue

        slug = json_path.stem
        if slug not in endpoint_costs:
            logger.warning(f"No cost info for endpoint: {slug}")
            continue

        with open(json_path, "rb") as f:
            data = orjson.loads(f.read())

        # Collect prompts in order and identify BIs
        prompts_in_order: list[
            tuple[str, int, bool]
        ] = []  # (prompt, token_count, is_bi)
        for token_count_str, prompts_dict in data.items():
            token_count = int(token_count_str)
            for prompt, outputs in prompts_dict.items():
                is_bi = len(set(outputs)) >= 2
                prompts_in_order.append((prompt, token_count, is_bi))

        total_prompts = len(prompts_in_order)
        total_bis = sum(1 for _, _, is_bi in prompts_in_order if is_bi)

        if total_bis == 0:
            continue

        bi_rate = total_bis / total_prompts
        estimated_prompts_for_target = min(total_prompts, int(target_bis / bi_rate) + 1)

        avg_input_tokens = (
            sum(tc for _, tc, _ in prompts_in_order) / total_prompts
            if prompts_in_order
            else 0
        )

        input_cost, output_cost = endpoint_costs[slug]
        cost_per_call = (
            avg_input_tokens * input_cost / 1e6
            + output_tokens_per_call * output_cost / 1e6
        )

        phase_1a_cost = (
            estimated_prompts_for_target * samples_per_prompt * cost_per_call
        )
        reference_cost = target_bis * reference_samples * cost_per_call
        endpoint_cost = phase_1a_cost + reference_cost

        results[slug] = {
            "phase_1_cost": endpoint_cost,
            "phase_1a_cost": phase_1a_cost,
            "reference_cost": reference_cost,
            "total_prompts": total_prompts,
            "total_bis": total_bis,
            "bi_rate": bi_rate,
            "estimated_prompts": estimated_prompts_for_target,
            "avg_input_tokens": avg_input_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
        }
        total_cost += endpoint_cost

    results["__total__"] = {"phase_1_cost": total_cost}
    return results


def print_phase1_cost(
    target_bis: int = 5,
    samples_per_prompt: int = 3,
    reference_samples: int = 50,
) -> None:
    """Print phase 1 initialization cost breakdown."""
    costs = compute_phase1_cost(target_bis, samples_per_prompt, reference_samples)
    total_info = costs.pop("__total__")
    total = total_info["phase_1_cost"]

    bi_rates = [c["bi_rate"] for c in costs.values()]
    avg_bi_rate = sum(bi_rates) / len(bi_rates) if bi_rates else 0

    avg_input_tokens_list = [c["avg_input_tokens"] for c in costs.values()]
    avg_input_tokens = (
        sum(avg_input_tokens_list) / len(avg_input_tokens_list)
        if avg_input_tokens_list
        else 0
    )

    estimated_prompts_list = [c["estimated_prompts"] for c in costs.values()]
    avg_estimated_prompts = (
        sum(estimated_prompts_list) / len(estimated_prompts_list)
        if estimated_prompts_list
        else 0
    )

    print(
        f"Phase 1 cost ({target_bis} BIs × {samples_per_prompt} samples, "
        f"{reference_samples} reference samples):"
    )
    print(f"{'Endpoint':<60} {'Cost (USD)':>12}")
    print("-" * 74)
    for slug, info in sorted(costs.items(), key=lambda x: -x[1]["phase_1_cost"]):
        print(f"{slug:<60} ${info['phase_1_cost']:>10.4f}")
    print("-" * 74)
    print(f"{'TOTAL':<60} ${total:>10.4f}")
    print(f"Average per endpoint: ${total / len(costs):>10.4f}")
    print(f"\nNumber of endpoints: {len(costs)}")
    print(f"Average BI discovery rate: {avg_bi_rate:.2%}")
    print(f"Average prompts to find {target_bis} BIs: {avg_estimated_prompts:.0f}")
    print(f"Average input tokens per prompt: {avg_input_tokens:.1f}")


def print_active_endpoints_over_time() -> None:
    """Print the number of endpoints with non-empty response data per day."""
    phase_2_dir = config.bi.phase_2_dir
    day_active: defaultdict[str, int] = defaultdict(int)
    total_endpoints = 0

    for d in sorted(phase_2_dir.iterdir()):
        if not d.is_dir():
            continue
        results = load_phase2_results(d)
        if not results:
            continue
        total_endpoints += 1

        active_days: set[str] = set()
        for batches in results.values():
            for ts, samples in batches.items():
                if samples:
                    active_days.add(ts[:10])
        for day in active_days:
            day_active[day] += 1

    for day in sorted(day_active):
        active = day_active[day]
        print(f"{day}: {active}/{total_endpoints} ({active / total_endpoints:.0%})")


if __name__ == "__main__":
    # print_phase1_cost()
    # print()
    # print_yearly_monitoring_cost()
    # plot_tv_distance_heatmap(Path("heatmap.pdf"))
    plot_tv_distance_transitions(
        Path("tv_distance_transitions.pdf"),
    )
    # plot_tv_distance_over_time(
    #     Path("tv_distance_over_time.pdf"),
    # )
    # plot_tv_distance_over_time_one(
    #     Path("tv_distance_over_time_one.pdf"),
    # )
