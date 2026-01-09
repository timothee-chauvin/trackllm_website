"""Test hardware noise by querying each endpoint with multiple prompts multiple times.

This measures the standard deviation of logprobs across different prompts for each model,
helping identify hardware-induced noise in logprob values.
"""

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import orjson
import plotly.graph_objects as go
from aiolimiter import AsyncLimiter
from tqdm import tqdm

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.download_tokenizers import (
    get_best_single_token_strings,
    load_existing_index,
)
from trackllm_website.bi.logprob_stats import FILTER_ENDPOINTS
from trackllm_website.bi.phase_1 import get_input_tokens
from trackllm_website.config import Endpoint, config, logger
from trackllm_website.util import slugify

NUM_PROMPTS = 100
QUERIES_PER_PROMPT = 20
DATA_DIR = config.bi.data_dir / "logprob_stats_noise"


def get_output_path(endpoint: Endpoint) -> Path:
    return DATA_DIR / f"{slugify(f'{endpoint.model}#{endpoint.provider}')}.json"


def load_existing_results(path: Path) -> dict[str, list[list[dict]]]:
    """Load existing results. Structure: {prompt: [[{token, logprob}, ...], ...]}"""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def save_results(path: Path, results: dict[str, list[list[dict]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(results))


@dataclass
class EndpointState:
    endpoint: Endpoint
    input_tokens: list[str]
    output_path: Path
    rate_limiter: AsyncLimiter
    concurrency_semaphore: asyncio.Semaphore
    results: dict[str, list[list[dict]]] = field(default_factory=dict)
    request_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    rate_limit_timestamps: deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    completed_queries: int = 0
    total_queries: int = 0
    recent_costs: deque[float] = field(default_factory=lambda: deque(maxlen=20))

    def __post_init__(self) -> None:
        self.results = load_existing_results(self.output_path)

    def get_requests_per_second(self) -> float:
        if len(self.request_timestamps) < 2:
            return 0.0
        now = time.monotonic()
        cutoff = now - 5.0
        recent = [t for t in self.request_timestamps if t > cutoff]
        if len(recent) < 2:
            return 0.0
        return len(recent) / (now - recent[0])

    def get_recent_rate_limits(self) -> int:
        now = time.monotonic()
        cutoff = now - 5.0
        return sum(1 for t in self.rate_limit_timestamps if t > cutoff)

    def get_pending_queries(self, token: str) -> int:
        """Return number of queries still needed for this token."""
        existing = len(self.results.get(token, []))
        return max(0, QUERIES_PER_PROMPT - existing)

    def get_completed_tokens(self) -> int:
        """Count tokens that have all queries completed."""
        return sum(
            1
            for token in self.input_tokens
            if len(self.results.get(token, [])) >= QUERIES_PER_PROMPT
        )

    def build_task_list(self) -> list[tuple[str, int]]:
        """Build list of (token, query_index) pairs, interleaved round-robin style."""
        tasks: list[tuple[str, int]] = []
        for query_idx in range(QUERIES_PER_PROMPT):
            for token in self.input_tokens:
                pending = self.get_pending_queries(token)
                if query_idx < pending:
                    tasks.append((token, query_idx))
        return tasks

    def record_result(self, token: str, logprobs: list[dict] | None) -> None:
        if logprobs is not None:
            if token not in self.results:
                self.results[token] = []
            self.results[token].append(logprobs)
            save_results(self.output_path, self.results)


async def query_and_record(
    client: OpenRouterClient,
    state: EndpointState,
    token: str,
) -> None:
    async with state.concurrency_semaphore:
        await state.rate_limiter.acquire()
        state.request_timestamps.append(time.monotonic())

        def on_retry(status: int) -> None:
            if status == 429:
                state.rate_limit_timestamps.append(time.monotonic())

        response = await client.query(
            state.endpoint, token, temperature=1.0, logprobs=True, on_retry=on_retry
        )
        state.completed_queries += 1
        if response.error:
            logger.warning(
                f"Error for {state.endpoint}: {token!r}: {response.error.message}"
            )
            return
        state.recent_costs.append(response.cost)

        logprobs_data = None
        if response.logprobs:
            logprobs_data = [
                {"token": t, "logprob": float(lp)}
                for t, lp in zip(response.logprobs.tokens, response.logprobs.logprobs)
            ]
        state.record_result(token, logprobs_data)


def log_status(states: list[EndpointState]) -> None:
    total_estimated_cost = 0.0
    for state in states:
        completed_tokens = state.get_completed_tokens()
        total_tokens = len(state.input_tokens)
        rate_limits = state.get_recent_rate_limits()
        rps = state.get_requests_per_second()
        avg_cost = (
            sum(state.recent_costs) / len(state.recent_costs)
            if state.recent_costs
            else 0
        )
        estimated_cost = avg_cost * state.total_queries
        total_estimated_cost += estimated_cost
        logger.info(
            f"{state.endpoint}: {completed_tokens}/{total_tokens} tokens, "
            f"{rps:.1f} rps, {rate_limits} recent 429s, "
            f"est. ${estimated_cost:.4f}"
        )
    logger.info(f"Total estimated cost: ${total_estimated_cost:.4f}")


def filter_endpoints(endpoints: list[Endpoint]) -> list[Endpoint]:
    return [ep for ep in endpoints if (ep.model, ep.provider) in FILTER_ENDPOINTS]


async def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer index and computing fallback tokens...")
    tokenizer_index = load_existing_index()
    fallback_tokens = get_best_single_token_strings()
    logger.info(
        f"Loaded {len(tokenizer_index)} tokenizers, {len(fallback_tokens)} fallback tokens"
    )

    endpoints = config.endpoints_lt
    endpoints = filter_endpoints(endpoints)
    for ep in endpoints:
        print(ep.model, ep.provider)
    logger.info(
        f"Running hardware noise collection for {len(endpoints)} endpoints, "
        f"{NUM_PROMPTS} prompts x {QUERIES_PER_PROMPT} queries each"
    )

    requests_per_second = config.bi.phase_1.requests_per_second_per_endpoint
    max_concurrent = config.bi.phase_1.max_concurrent_per_endpoint

    states = [
        EndpointState(
            endpoint=ep,
            input_tokens=get_input_tokens(
                ep, tokenizer_index, fallback_tokens, NUM_PROMPTS
            ),
            output_path=get_output_path(ep),
            rate_limiter=AsyncLimiter(requests_per_second, 1),
            concurrency_semaphore=asyncio.Semaphore(max_concurrent),
        )
        for ep in endpoints
    ]

    task_lists = [s.build_task_list() for s in states]

    for state, tasks in zip(states, task_lists):
        state.total_queries = len(tasks)

    client = OpenRouterClient()
    total_tasks = sum(len(tasks) for tasks in task_lists)
    logger.info(f"Total tasks to process: {total_tasks}")

    coros = []

    # Interleave tasks from all endpoints
    max_tasks = max(len(tasks) for tasks in task_lists) if task_lists else 0
    for i in range(max_tasks):
        for state, tasks in zip(states, task_lists):
            if i < len(tasks):
                token, _ = tasks[i]
                coros.append(query_and_record(client, state, token))

    last_status_time = time.monotonic()
    with tqdm(total=len(coros), desc="Requests") as pbar:
        for future in asyncio.as_completed(coros):
            await future
            pbar.update(1)
            now = time.monotonic()
            if now - last_status_time >= 5.0:
                log_status(states)
                last_status_time = now

    log_status(states)

    for state in states:
        missing = sum(state.get_pending_queries(t) for t in state.input_tokens)
        if missing > 0:
            logger.info(f"{state.endpoint}: {missing} missing responses")

    logger.info("Hardware noise collection complete")


def _compute_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return mean, math.sqrt(variance)


def _load_hardware_noise_stats() -> dict[str, dict[int, list[float]]]:
    """Load and compute hardware noise stats: {model: {rank: [stds per prompt]}}."""
    model_stats: dict[str, dict[int, list[float]]] = {}

    for path in sorted(DATA_DIR.glob("*.json")):
        results = load_existing_results(path)
        model_name = path.stem
        model_stats[model_name] = {1: [], 2: [], 3: [], 4: [], 5: []}

        for _, queries in results.items():
            if len(queries) < 2:
                continue

            top_logprobs_by_query = []
            for query_result in queries:
                sorted_lps = sorted(
                    [lp["logprob"] for lp in query_result if lp["logprob"] > -100],
                    reverse=True,
                )[:5]
                if len(sorted_lps) >= 5:
                    top_logprobs_by_query.append(sorted_lps)

            if len(top_logprobs_by_query) < 2:
                continue

            for rank in [1, 2, 3, 4, 5]:
                values = [q[rank - 1] for q in top_logprobs_by_query]
                _, std = _compute_stats(values)
                model_stats[model_name][rank].append(std)

    return model_stats


def print_hardware_noise_stats() -> None:
    """Analyze hardware noise: std of top logprobs across repeated queries."""
    model_stats = _load_hardware_noise_stats()

    print("\n=== Hardware Noise: Standard Deviation of Top Logprobs ===\n")
    print("std = standard deviation across repeated queries for the same prompt")
    print("Lower values = more deterministic logprobs\n")

    print(
        f"{'Model':<60} {'n':>6}  "
        f"{'std(top1)':>12}  {'std(top2)':>12}  {'std(top3)':>12}  "
        f"{'std(top4)':>12}  {'std(top5)':>12}"
    )
    print("-" * 136)

    all_stds: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: [], 5: []}

    for model_name in sorted(model_stats.keys()):
        stats = model_stats[model_name]
        n = len(stats[1])
        if n == 0:
            continue

        mean_stds = [_compute_stats(stats[r])[0] for r in [1, 2, 3, 4, 5]]

        for r in [1, 2, 3, 4, 5]:
            all_stds[r].extend(stats[r])

        print(
            f"{model_name:<60} {n:>6}  "
            f"{mean_stds[0]:>12.6f}  {mean_stds[1]:>12.6f}  {mean_stds[2]:>12.6f}  "
            f"{mean_stds[3]:>12.6f}  {mean_stds[4]:>12.6f}"
        )

    print("-" * 136)
    agg_means = [_compute_stats(all_stds[r])[0] for r in [1, 2, 3, 4, 5]]
    print(
        f"{'AGGREGATED':<60} {len(all_stds[1]):>6}  "
        f"{agg_means[0]:>12.6f}  {agg_means[1]:>12.6f}  {agg_means[2]:>12.6f}  "
        f"{agg_means[3]:>12.6f}  {agg_means[4]:>12.6f}"
    )


def plot_hardware_noise(output_path: Path | None = None) -> None:
    """Plot hardware noise: std of top logprobs across repeated queries."""
    model_stats = _load_hardware_noise_stats()
    sorted_models = sorted(model_stats.keys())
    labels = [slugify(m, max_length=20, hash_length=4) for m in sorted_models]

    colors = ["steelblue", "coral", "green", "purple", "orange"]

    fig = go.Figure()

    for rank in [1, 2, 3, 4, 5]:
        mean_stds = [_compute_stats(model_stats[m][rank])[0] for m in sorted_models]
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=mean_stds,
                mode="lines+markers",
                name=f"std(top{rank})",
                marker={"color": colors[rank - 1], "size": 8},
                line={"color": colors[rank - 1]},
            )
        )

    fig.update_layout(
        title="Hardware Noise: Std of Top Logprobs Across Repeated Queries",
        xaxis_title="Model",
        yaxis_title="Standard deviation",
        height=600,
        width=max(1000, len(labels) * 40),
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    fig.update_xaxes(tickangle=45)

    if output_path:
        fig.write_image(output_path)
        print(f"Saved figure to {output_path}")
    else:
        fig.show()


if __name__ == "__main__":
    # asyncio.run(main())
    # print_hardware_noise_stats()
    plot_hardware_noise(output_path=Path("hardware_noise.pdf"))
