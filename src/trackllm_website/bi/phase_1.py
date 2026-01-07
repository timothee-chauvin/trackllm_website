"""Phase 1: Identify border inputs by querying endpoints with single-token inputs."""

import asyncio
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
    load_tokenizer_vocab,
)
from trackllm_website.config import Endpoint, config, logger
from trackllm_website.util import gather_with_concurrency_streaming, slugify


def get_first_endpoint_per_provider() -> list[Endpoint]:
    """Get first endpoint for each provider."""
    provider_to_first: dict[str, Endpoint] = {}
    for endpoint in config.endpoints_bi:
        if endpoint.provider_without_suffix not in provider_to_first:
            provider_to_first[endpoint.provider_without_suffix] = endpoint
    return list(provider_to_first.values())


def get_input_tokens_for_endpoint(
    endpoint: Endpoint,
    tokenizer_index: dict[str, str],
    fallback_tokens: list[str],
) -> list[str]:
    """Get input tokens for an endpoint, using its tokenizer if known."""
    num_tokens = config.bi.phase_1.tokens_per_endpoint
    if endpoint.model in tokenizer_index:
        vocab = load_tokenizer_vocab(tokenizer_index[endpoint.model])
        return vocab[:num_tokens]
    return fallback_tokens[:num_tokens]


def get_output_path(endpoint: Endpoint) -> Path:
    """Get the output JSON path for an endpoint."""
    return (
        config.bi.phase_1_dir
        / f"{slugify(f'{endpoint.model}#{endpoint.provider}')}.json"
    )


def load_existing_results(path: Path) -> dict[str, dict[str, int]]:
    """Load existing results from JSON file."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def save_results(path: Path, results: dict[str, dict[str, int]]) -> None:
    """Save results to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(results))


@dataclass
class EndpointState:
    """Tracks state for a single endpoint."""

    endpoint: Endpoint
    input_tokens: list[str]
    output_path: Path
    rate_limiter: AsyncLimiter
    results: dict[str, dict[str, int]] = field(default_factory=dict)
    request_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    rate_limit_timestamps: deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    completed_queries: int = 0
    total_queries: int = 0
    got_404: bool = False
    last_cost: float = 0.0

    def __post_init__(self) -> None:
        self.results = load_existing_results(self.output_path)

    def get_requests_per_second(self) -> float:
        """Calculate actual requests per second over the last few seconds."""
        if len(self.request_timestamps) < 2:
            return 0.0
        now = time.monotonic()
        cutoff = now - 5.0  # Look at last 5 seconds
        recent = [t for t in self.request_timestamps if t > cutoff]
        if len(recent) < 2:
            return 0.0
        return len(recent) / (now - recent[0])

    def get_recent_rate_limits(self) -> int:
        """Count 429 errors in the last 5 seconds."""
        now = time.monotonic()
        cutoff = now - 5.0
        return sum(1 for t in self.rate_limit_timestamps if t > cutoff)

    def get_completed_tokens(self) -> int:
        """Count tokens that have all queries completed."""
        queries_needed = config.bi.phase_1.queries_per_token
        return sum(
            1
            for token in self.input_tokens
            if sum(self.results.get(token, {}).values()) >= queries_needed
        )

    def get_border_tokens(self) -> int:
        """Count tokens that have at least two different outputs."""
        return sum(1 for token, outputs in self.results.items() if len(outputs) >= 2)

    def get_pending_queries(self, token: str) -> int:
        """Return number of queries still needed for this token."""
        existing = sum(self.results.get(token, {}).values())
        return max(0, config.bi.phase_1.queries_per_token - existing)

    def get_missing_responses(self) -> int:
        """Count total missing responses across all tokens."""
        return sum(self.get_pending_queries(token) for token in self.input_tokens)

    def build_task_list(self) -> list[tuple[str, int]]:
        """Build list of (token, query_index) pairs ordered by chunks then rounds.

        Processes tokens in chunks: for each chunk, does all queries in round-robin
        (round 0 for all tokens in chunk, then round 1, etc.) before moving to next chunk.
        """
        queries_per_token = config.bi.phase_1.queries_per_token
        chunk_size = config.bi.phase_1.chunk_size
        tasks: list[tuple[str, int]] = []

        for chunk_start in range(0, len(self.input_tokens), chunk_size):
            chunk_tokens = self.input_tokens[chunk_start : chunk_start + chunk_size]
            for query_idx in range(queries_per_token):
                for token in chunk_tokens:
                    pending = self.get_pending_queries(token)
                    if query_idx < pending:
                        tasks.append((token, query_idx))

        return tasks

    def record_result(self, token: str, content: str | None) -> None:
        """Record a query result and save."""
        if content is not None:
            if token not in self.results:
                self.results[token] = {}
            self.results[token][content] = self.results[token].get(content, 0) + 1
            save_results(self.output_path, self.results)


async def query_and_record(
    client: OpenRouterClient,
    state: EndpointState,
    token: str,
) -> None:
    """Query endpoint and record result, respecting per-endpoint rate limit."""
    if state.got_404:
        return

    await state.rate_limiter.acquire()

    if state.got_404:
        return

    state.request_timestamps.append(time.monotonic())

    def on_retry(status: int) -> None:
        if status == 429:
            state.rate_limit_timestamps.append(time.monotonic())

    response = await client.query(
        state.endpoint, token, temperature=0, logprobs=False, on_retry=on_retry
    )
    state.completed_queries += 1
    if response.error:
        if response.error.http_code == 404:
            if not state.got_404:
                state.got_404 = True
                logger.warning(f"Got 404 for {state.endpoint}, abandoning endpoint")
            return
        logger.warning(
            f"Error for {state.endpoint}: {token!r}: {response.error.message}"
        )
        return
    state.last_cost = response.cost
    state.record_result(token, response.content)


def log_status(states: list[EndpointState]) -> None:
    """Log status for each endpoint."""
    total_estimated_cost = 0.0
    for state in states:
        completed_tokens = state.get_completed_tokens()
        border_tokens = state.get_border_tokens()
        total_tokens = len(state.input_tokens)
        rate_limits = state.get_recent_rate_limits()
        rps = state.get_requests_per_second()
        bi_pct = border_tokens / completed_tokens if completed_tokens else 0
        estimated_cost = state.last_cost * state.total_queries
        total_estimated_cost += estimated_cost
        logger.info(
            f"{state.endpoint}: {completed_tokens}/{total_tokens} tokens, "
            f"{border_tokens} ({bi_pct:.1%}) BI, {rps:.1f} rps, {rate_limits} recent 429s, "
            f"est. ${estimated_cost:.4f}"
        )
    logger.info(f"Total estimated cost: ${total_estimated_cost:.4f}")


async def main() -> None:
    config.bi.phase_1_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer index and computing fallback tokens...")
    tokenizer_index = load_existing_index()
    fallback_tokens = get_best_single_token_strings()
    logger.info(
        f"Loaded {len(tokenizer_index)} tokenizers, {len(fallback_tokens)} fallback tokens"
    )

    endpoints = get_first_endpoint_per_provider()
    logger.info(f"Running phase 1 for {len(endpoints)} endpoints")

    requests_per_second = config.bi.phase_1.requests_per_second_per_endpoint

    # Create state for each endpoint with its own rate limiter
    states = [
        EndpointState(
            endpoint=ep,
            input_tokens=get_input_tokens_for_endpoint(
                ep, tokenizer_index, fallback_tokens
            ),
            output_path=get_output_path(ep),
            rate_limiter=AsyncLimiter(requests_per_second, 1),
        )
        for ep in endpoints
    ]

    # Build all tasks across all endpoints, interleaved
    task_lists = [s.build_task_list() for s in states]
    max_tasks = max(len(t) for t in task_lists) if task_lists else 0

    # Set total_queries for each state
    for state, tasks in zip(states, task_lists):
        state.total_queries = len(tasks)

    client = OpenRouterClient()
    coros = []

    # Interleave tasks from all endpoints (order preserved by gather_with_concurrency_streaming)
    for i in range(max_tasks):
        for state, tasks in zip(states, task_lists):
            if i < len(tasks):
                token, _ = tasks[i]
                coros.append(query_and_record(client, state, token))

    logger.info(f"Total tasks to process: {len(coros)}")

    # Concurrency per endpoint limited by rate limiter − leave enough headroom in total concurrency
    total_concurrency = len(endpoints) * 20
    completed = 0
    with tqdm(total=len(coros), desc="Requests") as pbar:
        async for _ in gather_with_concurrency_streaming(total_concurrency, *coros):
            completed += 1
            pbar.update(1)
            if completed % 50 == 0:
                log_status(states)

    log_status(states)
    logger.info("Phase 1 complete")


if __name__ == "__main__":
    asyncio.run(main())
