"""Visualize border input results from phase 1."""

from pathlib import Path

import orjson
import plotly.graph_objects as go

from trackllm_website.config import config
from trackllm_website.util import slugify

MIN_SAMPLES = 500


def load_all_phase1_results(
    temperature: float | int = 0,
) -> dict[str, dict[int, dict[str, dict[str, int]]]]:
    """Load all phase 1 results from JSON files.

    Returns a dict mapping endpoint slug to results.
    """
    results = {}
    phase_1_dir = config.bi.get_phase_1_dir(temperature)
    for path in phase_1_dir.glob("*.json"):
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
        endpoint_slug = path.stem
        results[endpoint_slug] = {int(k): v for k, v in data.items()}
    return results


def compute_stats(
    endpoint_results: dict[int, dict[str, dict[str, int]]],
) -> tuple[int, int, int]:
    """Compute stats for an endpoint's results.

    Returns (total_samples, total_tokens, border_tokens).
    """
    merged: dict[str, dict[str, int]] = {}
    for token_results in endpoint_results.values():
        for token, outputs in token_results.items():
            if token not in merged:
                merged[token] = {}
            for output, count in outputs.items():
                merged[token][output] = merged[token].get(output, 0) + count

    total_samples = sum(sum(outputs.values()) for outputs in merged.values())
    total_tokens = len(merged)
    border_tokens = sum(1 for outputs in merged.values() if len(outputs) >= 2)
    return total_samples, total_tokens, border_tokens


def plot_border_input_fractions(
    min_samples: int = MIN_SAMPLES,
    output_path: Path | None = None,
    temperature: float | int = 0,
) -> None:
    """Plot bar chart of border input fractions for endpoints with enough samples.

    Args:
        min_samples: Minimum number of samples required to include an endpoint.
        output_path: If provided, save figure to this path instead of displaying.
        temperature: Temperature used in phase 1.
    """
    all_results = load_all_phase1_results(temperature)

    endpoint_stats: list[tuple[str, float, int, int]] = []
    for endpoint_slug, results in all_results.items():
        total_samples, total_tokens, border_tokens = compute_stats(results)
        if total_samples >= min_samples and total_tokens > 0:
            fraction = border_tokens / total_tokens
            endpoint_stats.append(
                (endpoint_slug, fraction, total_tokens, border_tokens)
            )

    if not endpoint_stats:
        print(f"No endpoints found with at least {min_samples} samples")
        return

    endpoint_stats.sort(key=lambda x: x[1], reverse=True)
    for stat in endpoint_stats:
        print(f"{stat[0]}: BI={stat[3]}/{stat[2]} = {stat[1] * 100:.1f}%")

    labels = [slugify(s[0], max_length=15, hash_length=4) for s in endpoint_stats]
    fractions = [s[1] * 100 for s in endpoint_stats]
    token_counts = [s[2] for s in endpoint_stats]
    border_counts = [s[3] for s in endpoint_stats]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=fractions,
                text=[f"S={s} BI={bi}" for s, bi in zip(token_counts, border_counts)],
                textposition="outside",
                textangle=-90,
                # textfont=dict(size=14),
                marker_color="steelblue",
            )
        ]
    )

    endpoints_above_threshold = sum(1 for f in fractions if f >= 0.5)
    above_threshold_frac = endpoints_above_threshold / len(labels)
    above_threshold_str = f"{above_threshold_frac:.1%} ({endpoints_above_threshold}/{len(labels)}) endpoints with ≥0.5% border inputs"
    fig.update_layout(
        title=f"Fraction of Border Inputs by Endpoint (T={temperature:g}, min {min_samples} samples)<br><sup>{above_threshold_str}</sup>",
        xaxis_title="Endpoint",
        yaxis_title="Border Inputs (%)",
        yaxis_range=[0, 100],
        xaxis_tickangle=45,
        height=600,
        width=max(800, len(labels) * 25),
        # for some reason, textfont in go.Bar didn't work without these uniformtext parameters
        uniformtext_minsize=12,
        uniformtext_mode="show",
    )

    fig.write_image(output_path)
    print(f"Saved figure to {output_path}")


def get_border_inputs(
    endpoint_results: dict[int, dict[str, dict[str, int]]],
) -> list[tuple[int, str]]:
    """Get all border inputs (token_count, input_token) for an endpoint."""
    merged: dict[tuple[int, str], dict[str, int]] = {}
    for token_count, token_results in endpoint_results.items():
        for token, outputs in token_results.items():
            key = (token_count, token)
            if key not in merged:
                merged[key] = {}
            for output, count in outputs.items():
                merged[key][output] = merged[key].get(output, 0) + count

    return [(tc, tok) for (tc, tok), outputs in merged.items() if len(outputs) >= 2]


def phase_2_est_cost(
    temperature: float | int = 0,
    samples_per_day: int = 100,
    min_samples: int = MIN_SAMPLES,
) -> None:
    """Estimate cost for phase 2 based on phase 1 border inputs.

    Args:
        temperature: Temperature used in phase 1.
        samples_per_day: Number of times to sample each border input per day.
        min_samples: Minimum samples required to include an endpoint.
    """
    all_results = load_all_phase1_results(temperature)

    endpoint_costs: dict[str, tuple[float, float]] = {}
    for endpoint in config.endpoints_bi:
        slug = slugify(f"{endpoint.model}#{endpoint.provider}")
        endpoint_costs[slug] = endpoint.cost

    total_cost_per_day = 0.0
    endpoint_details: list[tuple[str, int, float]] = []

    for endpoint_slug, results in all_results.items():
        total_samples, total_tokens, border_tokens = compute_stats(results)
        if total_samples < min_samples:
            continue

        if endpoint_slug not in endpoint_costs:
            print(f"Warning: {endpoint_slug} not found in endpoints_bi.yaml")
            continue

        border_inputs = get_border_inputs(results)
        if not border_inputs:
            continue

        input_cost_per_mtok, output_cost_per_mtok = endpoint_costs[endpoint_slug]

        # Each border input: input tokens from token_count, output is 1 token
        total_input_tokens = sum(tc for tc, _ in border_inputs)
        total_output_tokens = len(border_inputs)

        daily_input_tokens = total_input_tokens * samples_per_day
        daily_output_tokens = total_output_tokens * samples_per_day

        daily_cost = (
            daily_input_tokens * input_cost_per_mtok / 1_000_000
            + daily_output_tokens * output_cost_per_mtok / 1_000_000
        )
        total_cost_per_day += daily_cost
        endpoint_details.append((endpoint_slug, len(border_inputs), daily_cost))

    endpoint_details.sort(key=lambda x: x[2], reverse=True)

    print(f"Phase 2 cost estimate (T={temperature}, {samples_per_day} samples/day)")
    print(f"{'Endpoint':<60} {'BIs':>6} {'$/day':>10} {'$/month':>12}")
    print("-" * 90)
    for slug, bi_count, daily in endpoint_details:
        monthly = daily * 30
        print(f"{slug:<60} {bi_count:>6} {daily:>10.4f} {monthly:>12.2f}")
    print("-" * 90)
    monthly_total = total_cost_per_day * 30
    print(f"{'TOTAL':<60} {'':<6} {total_cost_per_day:>10.4f} {monthly_total:>12.2f}")
    print(f"\nTotal endpoints: {len(endpoint_details)}")
    print(f"Total border inputs: {sum(e[1] for e in endpoint_details)}")


if __name__ == "__main__":
    phase_2_est_cost(temperature=0)
    print()
    for temperature in [0, 1e-10, 1e-5, 1e-2, 1]:
        output_path = Path(f"plots/bi_prevalence/T={temperature}.pdf")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_border_input_fractions(
            min_samples=MIN_SAMPLES,
            output_path=output_path,
            temperature=temperature,
        )
