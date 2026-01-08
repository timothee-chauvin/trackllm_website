"""Collect logprob statistics by querying LT endpoints with single-token inputs."""

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import orjson
from aiolimiter import AsyncLimiter
from tqdm import tqdm

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.download_tokenizers import (
    get_best_single_token_strings,
    load_existing_index,
)
from trackllm_website.bi.phase_1 import get_input_tokens
from trackllm_website.config import Endpoint, config, logger
from trackllm_website.util import slugify

NUM_PROMPTS = 1000
DATA_DIR = config.bi.data_dir / "logprob_stats"


def get_output_path(endpoint: Endpoint) -> Path:
    return DATA_DIR / f"{slugify(f'{endpoint.model}#{endpoint.provider}')}.json"


def load_existing_results(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def save_results(path: Path, results: dict[str, list[dict]]) -> None:
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
    results: dict[str, list[dict]] = field(default_factory=dict)
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

    def get_completed_tokens(self) -> int:
        return sum(1 for token in self.input_tokens if token in self.results)

    def get_pending_tokens(self) -> list[str]:
        return [token for token in self.input_tokens if token not in self.results]

    def record_result(self, token: str, logprobs: list[dict] | None) -> None:
        if logprobs is not None:
            self.results[token] = logprobs
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
    """Return only specific model+provider combinations."""
    allowed = {
        ("meta-llama/llama-3.1-8b-instruct", "cerebras/fp16"),
        ("deepseek/deepseek-v3.2", "deepseek"),
        ("openai/gpt-4o-mini-2024-07-18", "openai"),
        ("openai/gpt-oss-120b", "crusoe/bf16"),
        ("meta-llama/llama-3.3-70b-instruct", "crusoe/bf16"),
        ("qwen/qwen3-32b", "cerebras"),
        ("gryphe/mythomax-l2-13b", "mancer/fp8"),
        ("undi95/remm-slerp-l2-13b", "mancer/fp8"),
        ("minimax/minimax-m2.1", "fireworks"),
        ("mancer/weaver", "mancer/fp8"),
        ("qwen/qwen3-235b-a22b-2507", "cerebras"),
        ("deepseek/deepseek-chat-v3-0324", "fireworks"),
        ("qwen/qwen2.5-vl-32b-instruct", "fireworks"),
        ("openai/gpt-3.5-turbo", "openai"),
        ("qwen/qwen3-coder", "fireworks"),
        ("z-ai/glm-4.6", "mancer/fp8"),
        ("moonshotai/kimi-k2-thinking", "parasail/int4"),
        ("openai/gpt-3.5-turbo-0613", "azure"),
        ("moonshotai/kimi-k2-0905", "fireworks/fp8"),
        ("deepseek/deepseek-r1-0528", "crusoe/fp8"),
        ("anthracite-org/magnum-v4-72b", "mancer/fp8"),
        ("openai/gpt-4o-2024-08-06", "azure"),
        ("openai/gpt-4o-2024-11-20", "openai"),
        ("openai/chatgpt-4o-latest", "openai"),
    }
    return [ep for ep in endpoints if (ep.model, ep.provider) in allowed]


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
    logger.info(f"Running logprob stats collection for {len(endpoints)} endpoints")

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

    task_lists = [s.get_pending_tokens() for s in states]

    for state, tasks in zip(states, task_lists):
        state.total_queries = len(tasks)

    client = OpenRouterClient()
    total_tasks = sum(len(tasks) for tasks in task_lists)
    logger.info(f"Total tasks to process: {total_tasks}")

    coros = []

    max_tasks = max(len(tasks) for tasks in task_lists) if task_lists else 0
    for i in range(max_tasks):
        for state, tasks in zip(states, task_lists):
            if i < len(tasks):
                token = tasks[i]
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
        missing = len(state.get_pending_tokens())
        if missing > 0:
            logger.info(f"{state.endpoint}: {missing} missing responses")

    logger.info("Logprob stats collection complete")


def analyze() -> None:
    """Analyze logprob distributions: differences between top tokens."""
    all_diff_1_2: list[float] = []
    all_diff_2_3: list[float] = []

    model_stats: dict[str, tuple[list[float], list[float]]] = {}

    for path in sorted(DATA_DIR.glob("*.json")):
        results = load_existing_results(path)

        model_name = path.stem
        diff_1_2: list[float] = []
        diff_2_3: list[float] = []

        for _, logprobs in results.items():
            # Sort by logprob descending (highest probability first)
            sorted_lps = sorted(
                [lp["logprob"] for lp in logprobs if lp["logprob"] > -100],
                reverse=True,
            )
            if len(sorted_lps) < 3:
                continue
            diff_1_2.append(sorted_lps[0] - sorted_lps[1])
            diff_2_3.append(sorted_lps[1] - sorted_lps[2])

        if diff_1_2:
            model_stats[model_name] = (diff_1_2, diff_2_3)
            all_diff_1_2.extend(diff_1_2)
            all_diff_2_3.extend(diff_2_3)

    def compute_stats(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
        return mean, std

    print("\n=== Logprob Difference Statistics ===\n")
    print("Difference = logprob(rank_i) - logprob(rank_i+1)")
    print("Higher values = larger gap between tokens\n")

    print(f"{'Model':<60} {'n':>6}  {'diff_1_2':>20}  {'diff_2_3':>20}")
    print("-" * 112)

    for model_name in sorted(model_stats.keys()):
        diff_1_2, diff_2_3 = model_stats[model_name]
        mean_12, std_12 = compute_stats(diff_1_2)
        mean_23, std_23 = compute_stats(diff_2_3)
        print(
            f"{model_name:<60} {len(diff_1_2):>6}  "
            f"{mean_12:>8.4f} ± {std_12:<8.4f}  "
            f"{mean_23:>8.4f} ± {std_23:<8.4f}"
        )

    print("-" * 112)
    mean_12, std_12 = compute_stats(all_diff_1_2)
    mean_23, std_23 = compute_stats(all_diff_2_3)
    print(
        f"{'AGGREGATED':<60} {len(all_diff_1_2):>6}  "
        f"{mean_12:>8.4f} ± {std_12:<8.4f}  "
        f"{mean_23:>8.4f} ± {std_23:<8.4f}"
    )


if __name__ == "__main__":
    # asyncio.run(main())
    analyze()
