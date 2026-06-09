"""Visualize border input results from phase 1."""

from collections import defaultdict
from pathlib import Path

import orjson
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap

from trackllm_website.config import config
from trackllm_website.util import slugify

MIN_SAMPLES = 450


def get_phase_1_paths(
    temperature: float | int = 0,
    base_dir: Path | None = None,
) -> list[Path]:
    """Get all phase 1 result JSON paths for a given temperature."""
    phase_1_dir = config.bi.get_phase_1_dir(temperature, base_dir)
    return [p for p in phase_1_dir.glob("*.json") if p.name != "border_inputs.json"]


def load_all_phase1_results(
    temperature: float | int = 0,
    base_dir: Path | None = None,
) -> dict[str, dict[int, dict[str, list[str]]]]:
    """Load all phase 1 results from JSON files.

    Returns a dict mapping endpoint slug to results.
    Results format: {endpoint_slug: {token_count: {input_token: [output1, output2, ...]}}}
    """
    results = {}
    for path in get_phase_1_paths(temperature, base_dir):
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
        endpoint_slug = path.stem
        data.pop("_meta", None)
        results[endpoint_slug] = {int(k): v for k, v in data.items()}
    if not results:
        raise ValueError(f"No results found for temperature {temperature}")
    return results


def merge_results(
    endpoint_results: dict[int, dict[str, list[str]]],
) -> dict[str, list[str]]:
    """Merge results across token counts, keeping only complete inputs."""
    merged: dict[str, list[str]] = {}
    for token_results in endpoint_results.values():
        for token, outputs in token_results.items():
            if token not in merged:
                merged[token] = []
            merged[token].extend(outputs)
    return {
        token: outputs
        for token, outputs in merged.items()
        if len(outputs) == config.bi.phase_1.queries_per_token
    }


def compute_stats(
    endpoint_results: dict[int, dict[str, list[str]]],
) -> tuple[int, int, int]:
    """Compute stats for an endpoint's results.

    Returns (total_samples, total_tokens, border_tokens).
    """
    merged = merge_results(endpoint_results)
    total_samples = sum(len(outputs) for outputs in merged.values())
    total_tokens = len(merged)
    border_tokens = sum(1 for outputs in merged.values() if len(set(outputs)) >= 2)
    return total_samples, total_tokens, border_tokens


def get_all_unique_outputs(
    endpoint_results: dict[int, dict[str, list[str]]],
) -> set[str]:
    """Return all unique outputs across all inputs (ignoring completeness filter)."""
    all_outputs: set[str] = set()
    for token_results in endpoint_results.values():
        for outputs in token_results.values():
            all_outputs.update(outputs)
    return all_outputs


def get_output_input_counts(
    endpoint_results: dict[int, dict[str, list[str]]],
) -> dict[str, int]:
    """Return count of inputs that produced each unique output."""
    output_counts: dict[str, int] = defaultdict(int)
    for token_results in endpoint_results.values():
        for outputs in token_results.values():
            for output in set(outputs):
                output_counts[output] += 1
    return dict(output_counts)


def get_output_to_inputs(
    endpoint_results: dict[int, dict[str, list[str]]],
) -> dict[str, set[str]]:
    """Return mapping from each output to the input tokens that produced it."""
    output_inputs: dict[str, set[str]] = defaultdict(set)
    for token_results in endpoint_results.values():
        for input_token, outputs in token_results.items():
            for output in set(outputs):
                output_inputs[output].add(input_token)
    return dict(output_inputs)


def get_common_endpoints(
    temperatures: list[float | int],
    min_samples: int = MIN_SAMPLES,
    base_dir: Path | None = None,
) -> set[str]:
    """Get endpoints present at ALL temperatures with enough samples."""
    endpoints_per_temp: list[set[str]] = []
    for t in temperatures:
        all_results = load_all_phase1_results(t, base_dir)
        valid_endpoints: set[str] = set()
        for endpoint_slug, results in all_results.items():
            total_samples, total_tokens, _ = compute_stats(results)
            if total_samples >= min_samples and total_tokens > 0:
                valid_endpoints.add(endpoint_slug)
        endpoints_per_temp.append(valid_endpoints)
    return set.intersection(*endpoints_per_temp)


def plot_border_input_fractions(
    temperatures: list[float | int],
    min_samples: int = MIN_SAMPLES,
    output_path: Path | None = None,
    label_padding: float = 0.03,  # fraction of plot height for label spacing
    base_dir: Path | None = None,
) -> None:
    """Plot strip plot of model counts by BI fraction."""
    plot_data = []
    xaxis_cats = []
    green_frac_x = []
    green_frac_y = []
    bi_threshold = 0.005

    def create_strip_trace(stats, color, name, x_label, show_legend):
        fractions = [s[1] for s in stats]
        box = go.Box(
            x=[x_label] * len(fractions),
            y=fractions,
            boxpoints=False,
            fillcolor="rgba(0, 128, 0, 0.2)",
            line_color="rgba(0, 128, 0, 0.6)",
            showlegend=False,
            offsetgroup="0",
        )
        points = go.Box(
            x=[x_label] * len(fractions),
            y=fractions,
            boxpoints="all",
            jitter=0.5,
            pointpos=0,
            fillcolor="rgba(0,0,0,0)",
            line_color="rgba(0,0,0,0)",
            marker=dict(size=5, color=color),
            name=name,
            showlegend=show_legend,
            hovertext=[f"{s[0]}: {s[3]}/{s[2]}" for s in stats],
            offsetgroup="0",
            legendgroup="dots",
        )
        return box, points

    # Track which endpoints are ever green across all temperatures
    ever_green_slugs: set[str] = set()
    # Track unique outputs per endpoint across all temperatures
    endpoint_unique_outputs: dict[str, set[str]] = {}

    common_endpoints = get_common_endpoints(temperatures, min_samples, base_dir)
    print(f"Endpoints present at all temperatures: {len(common_endpoints)}")

    first_temp = True
    for t in temperatures:
        all_results = load_all_phase1_results(t, base_dir)
        endpoint_stats: list[tuple[str, float, int, int]] = []
        for endpoint_slug, results in all_results.items():
            if endpoint_slug not in common_endpoints:
                continue
            total_samples, total_tokens, border_tokens = compute_stats(results)
            if total_samples >= min_samples and total_tokens > 0:
                fraction = border_tokens / total_tokens
                endpoint_stats.append(
                    (endpoint_slug, fraction, total_tokens, border_tokens)
                )
                # Accumulate unique outputs across all temperatures
                unique_outputs = get_all_unique_outputs(results)
                if endpoint_slug not in endpoint_unique_outputs:
                    endpoint_unique_outputs[endpoint_slug] = set()
                endpoint_unique_outputs[endpoint_slug].update(unique_outputs)
        if not endpoint_stats:
            print(f"No endpoints found for T={t} with at least {min_samples} samples")
            continue
        endpoint_stats.sort(key=lambda x: x[1], reverse=True)

        stats_green = [s for s in endpoint_stats if s[1] > bi_threshold]
        stats_red = [s for s in endpoint_stats if s[1] <= bi_threshold]

        for s in stats_green:
            ever_green_slugs.add(s[0])

        x_label = f"T={t}"
        box, points = create_strip_trace(
            stats_red,
            "red",
            f"Endpoints with ≤ {bi_threshold * 100:.1f}% BI",
            x_label,
            first_temp,
        )
        plot_data.extend([box, points])
        box, points = create_strip_trace(
            stats_green,
            "green",
            f"Endpoints with > {bi_threshold * 100:.1f}% BI",
            x_label,
            first_temp,
        )
        plot_data.extend([box, points])
        green_fraction = len(stats_green) / len(endpoint_stats)
        green_frac_x.append(x_label)
        green_frac_y.append(green_fraction)
        xaxis_cats.append(x_label)
        first_temp = False

    # Print unique output analysis across all temperatures
    # Green = ever green at any temperature, Red = always red at all temperatures
    always_red_slugs = set(endpoint_unique_outputs.keys()) - ever_green_slugs
    green_unique_counts = [
        len(endpoint_unique_outputs[slug])
        for slug in ever_green_slugs
        if slug in endpoint_unique_outputs
    ]
    red_endpoint_outputs = [
        (slug, endpoint_unique_outputs[slug]) for slug in always_red_slugs
    ]

    if green_unique_counts:
        print(
            f"Ever-green endpoints ({len(green_unique_counts)}) avg total unique outputs: "
            f"{sum(green_unique_counts) / len(green_unique_counts):.1f}"
        )
    if red_endpoint_outputs:
        red_unique_counts = [len(o) for _, o in red_endpoint_outputs]
        print(
            f"Always-red endpoints ({len(red_endpoint_outputs)}) avg total unique outputs: "
            f"{sum(red_unique_counts) / len(red_unique_counts):.1f}"
        )
        print("Always-red endpoint unique outputs (across all temperatures):")
        for slug, unique_outputs in sorted(
            red_endpoint_outputs, key=lambda x: len(x[1])
        ):
            print(f"  {slug}: {unique_outputs!r}")

    plot_data.append(
        go.Scatter(
            x=green_frac_x,
            y=green_frac_y,
            mode="lines+markers",
            name=f"Fraction of Endpoints with > {bi_threshold * 100:.1f}% BI",
            line=dict(color="black", width=2),
            marker=dict(color="white", size=10, line=dict(color="black", width=2)),
            legendgroup="line",
            legendrank=0,
        )
    )

    fig = go.Figure(data=plot_data)
    for x, y in zip(green_frac_x, green_frac_y):
        fig.add_annotation(
            x=x,
            y=y,
            text=f"{y:.0%}",
            showarrow=False,
            yshift=-15,
            xshift=35,
            font=dict(size=24),
        )
    fig.update_layout(
        template=config.plotting.template,
        font_family=config.plotting.font_family,
        yaxis_title="BI Prevalence per Endpoint",
        font_size=24,
        height=500,
        width=800,
        boxmode="overlay",
        xaxis=dict(
            categoryorder="array",
            categoryarray=xaxis_cats,
            ticklabelstandoff=int(label_padding * 500),
        ),
        yaxis=dict(range=[0, 1.1], title_standoff=25),
        legend=dict(yanchor="top", y=1.05, xanchor="left", x=0.01, traceorder="normal"),
        margin=dict(l=0, r=0, t=0, b=50, autoexpand=True),
    )
    if output_path:
        fig.write_image(output_path)
        print(f"Saved figure to {output_path}")
    else:
        fig.show()


def get_border_inputs(
    endpoint_results: dict[int, dict[str, list[str]]],
) -> list[tuple[int, str]]:
    """Get all border inputs (token_count, input_token) for an endpoint."""
    merged: dict[tuple[int, str], list[str]] = {}
    for token_count, token_results in endpoint_results.items():
        for token, outputs in token_results.items():
            key = (token_count, token)
            if key not in merged:
                merged[key] = []
            merged[key].extend(outputs)

    return [
        (tc, tok) for (tc, tok), outputs in merged.items() if len(set(outputs)) >= 2
    ]


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


def parse_endpoint_slug(slug: str) -> tuple[str, str]:
    """Parse endpoint slug into (model, provider).

    Slug format: {model}23{provider} where 23 is hex-encoded '#',
    and '/' in names is hex-encoded as '2f'.
    Uses the last occurrence of '23' as the separator since model names
    may contain '23' (e.g., 235b for 235 billion params).
    """
    idx = slug.rfind("23")
    if idx == -1:
        raise ValueError(f"Invalid endpoint slug (no separator): {slug}")
    model_slug = slug[:idx]
    provider_slug = slug[idx + 2 :]
    model = model_slug.replace("2f", "/")
    provider = provider_slug.replace("2f", "/")
    return model, provider


def plot_requests_per_bi_ecdf(
    temperatures: list[float | int],
    min_samples: int = MIN_SAMPLES,
    output_path: Path | None = None,
    base_dir: Path | None = None,
) -> None:
    """Plot eCDF of requests per successful border input across temperatures.

    X-axis: number of requests per successful BI
    Y-axis: cumulative fraction of endpoints
    Each temperature is a different line with a color scale from black to red.
    """
    import numpy as np

    common_endpoints = get_common_endpoints(temperatures, min_samples, base_dir)
    print(f"Endpoints present at all temperatures: {len(common_endpoints)}")

    # Paul Tol's YlOrBr color map
    color_map = LinearSegmentedColormap.from_list(
        "YlOrBr",
        [
            "#FFFFE5",
            "#FFF7BC",
            "#FEE391",
            "#FEC44F",
            "#FB9A29",
            "#EC7014",
            "#CC4C02",
            "#993404",
            "#662506",
        ],
    )

    n_temps = len(temperatures)
    colors = ["black"]  # for T=0
    color_scale_start = 0.9
    color_scale_end = 0.15
    colors.extend(
        [
            f"rgba({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)},{c[3]:.2f})"
            for c in color_map(
                np.linspace(color_scale_start, color_scale_end, n_temps - 1)
            )
        ]
    )

    fig = go.Figure()

    for idx, t in enumerate(temperatures):
        all_results = load_all_phase1_results(t, base_dir)
        requests_per_bi: list[float] = []

        for endpoint_slug, results in all_results.items():
            if endpoint_slug not in common_endpoints:
                continue
            total_samples, total_tokens, border_tokens = compute_stats(results)
            if total_samples >= min_samples and total_tokens > 0 and border_tokens > 0:
                requests_per_bi.append(
                    total_tokens * config.bi.phase_1.queries_per_token / border_tokens
                )

        if not requests_per_bi:
            print(f"No endpoints with BIs at T={t}")
            continue

        # Compute eCDF
        sorted_vals = np.sort(requests_per_bi)
        n_bi = len(requests_per_bi)
        n_all = len(common_endpoints)
        print(
            f"T={t} Fraction of endpoints with BIs: {n_bi}/{n_all} = {n_bi / n_all:.2%}"
        )
        ecdf_y = np.arange(1, n_bi + 1) / n_all

        color = colors[idx]
        fig.add_trace(
            go.Scatter(
                x=sorted_vals,
                y=ecdf_y,
                mode="lines",
                name=f"T={t}",
                line=dict(color=color, width=2),
                line_shape="hv",
            )
        )

    fig.update_layout(
        template=config.plotting.template,
        font_family=config.plotting.font_family,
        xaxis_title="Requests per Successful BI",
        yaxis_title="Cumulative Fraction of Endpoints",
        font_size=24,
        height=500,
        width=800,
        xaxis=dict(
            type="log",
            range=[np.log10(min(requests_per_bi)), np.log10(max(requests_per_bi))],
        ),
        yaxis=dict(range=[0, 1.05], side="right", title_standoff=25),
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=-0.1, bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=0, r=0, t=0, b=90, autoexpand=True),
    )
    if output_path:
        fig.write_image(output_path)
        print(f"Saved figure to {output_path}")
    else:
        fig.show()


def print_bi_fractions_by_model(
    temperature: float | int = 0,
    min_samples: int = MIN_SAMPLES,
) -> None:
    """Print BI fractions grouped by model, with providers listed."""
    all_results = load_all_phase1_results(temperature)

    # Group by model: model -> list of (provider, bi_fraction)
    model_data: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for endpoint_slug, results in all_results.items():
        total_samples, total_tokens, border_tokens = compute_stats(results)
        if total_samples < min_samples or total_tokens == 0:
            continue

        model, provider = parse_endpoint_slug(endpoint_slug)
        bi_fraction = 100 * border_tokens / total_tokens

        model_data[model].append((provider, bi_fraction))

    same_bi = []
    different_bi = []
    for model in sorted(model_data.keys()):
        providers_fractions = sorted(model_data[model], key=lambda x: x[1])
        fractions = [round(f, 1) for _, f in providers_fractions]
        providers = [p for p, _ in providers_fractions]
        if len(fractions) > 1:
            if len(set(fractions)) == 1:
                same_bi.append((model, providers, fractions))
            else:
                different_bi.append((model, providers, fractions))

    print(f"\nSame BI: {len(same_bi)}")
    for model, providers, fractions in same_bi:
        print(f"{model}: {providers}, {fractions}")
    print(f"\nDifferent BI: {len(different_bi)}")
    for model, providers, fractions in different_bi:
        print(f"{model}: {providers}, {fractions}")


def print_outputs_of_endpoints_without_bi_across_temps(
    temperatures: list[float | int],
    min_samples: int = MIN_SAMPLES,
    base_dir: Path | None = None,
) -> None:
    """Print outputs of endpoints that have no BIs at any temperature."""
    common_endpoints = get_common_endpoints(temperatures, min_samples, base_dir)

    endpoints_with_bi: set[str] = set()
    endpoint_output_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    endpoint_output_inputs: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for t in temperatures:
        all_results = load_all_phase1_results(t, base_dir)
        for endpoint_slug, results in all_results.items():
            if endpoint_slug not in common_endpoints:
                continue
            _, _, border_tokens = compute_stats(results)
            if border_tokens > 0:
                endpoints_with_bi.add(endpoint_slug)
            for output, count in get_output_input_counts(results).items():
                endpoint_output_counts[endpoint_slug][output] += count
            for output, inputs in get_output_to_inputs(results).items():
                endpoint_output_inputs[endpoint_slug][output].update(inputs)

    endpoints_without_bi = common_endpoints - endpoints_with_bi
    print(f"Endpoints without BIs at any temperature: {len(endpoints_without_bi)}")
    for endpoint_slug in sorted(endpoints_without_bi):
        counts = endpoint_output_counts[endpoint_slug]
        sorted_outputs = sorted(counts.items(), key=lambda x: -x[1])
        print(f"{endpoint_slug}: {dict(sorted_outputs)}")
        for output, count in sorted_outputs:
            if output != "":
                inputs = endpoint_output_inputs[endpoint_slug][output]
                print(f"  '{output}' ({count}): {sorted(inputs)}")


if __name__ == "__main__":
    # print_bi_fractions_by_model(temperature=0)
    # phase_2_est_cost(temperature=0)
    # print()
    temperatures_to_plot = [0.0, 1e-10, 1e-5, 1e-3, 1e-2, 1e-1, 0.5, 1.5]
    base_dir = config.bi.data_dir / "bi_prevalence"
    # output_path = Path("plots/bi_prevalence/all_temperatures.pdf")
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # plot_border_input_fractions(
    #     temperatures=temperatures_to_plot,
    #     min_samples=MIN_SAMPLES,
    #     output_path=output_path,
    #     base_dir=base_dir,
    # )
    # ecdf_output_path = Path("plots/bi_prevalence/requests_per_bi_ecdf.pdf")
    # plot_requests_per_bi_ecdf(
    #     temperatures=temperatures_to_plot,
    #     min_samples=MIN_SAMPLES,
    #     output_path=ecdf_output_path,
    #     base_dir=base_dir,
    # )
    print_outputs_of_endpoints_without_bi_across_temps(
        temperatures=temperatures_to_plot,
        min_samples=MIN_SAMPLES,
        base_dir=base_dir,
    )
