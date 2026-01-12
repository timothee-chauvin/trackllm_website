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
    results: dict[int, dict[str, dict[str, int]]],
) -> tuple[int, int, int]:
    """Compute stats for an endpoint's results.

    Returns (total_samples, total_tokens, border_tokens).
    """
    merged: dict[str, dict[str, int]] = {}
    for token_results in results.values():
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

    fig.update_layout(
        title=f"Fraction of Border Inputs by Endpoint (T={temperature:g}, min {min_samples} samples)",
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


if __name__ == "__main__":
    TEMPERATURE = 1e-10
    output_path = Path(f"plots/bi_prevalence/T={TEMPERATURE}.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_border_input_fractions(
        min_samples=MIN_SAMPLES,
        output_path=output_path,
        temperature=TEMPERATURE,
    )
