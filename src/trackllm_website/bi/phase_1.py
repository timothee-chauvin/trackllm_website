"""Phase 1: Identify border inputs by querying endpoints with single-token inputs."""

import asyncio
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import orjson
from aiolimiter import AsyncLimiter

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.download_tokenizers import (
    get_best_single_token_strings,
    load_existing_index,
    load_tokenizer_vocab,
)
from trackllm_website.config import Endpoint, config, logger
from trackllm_website.util import gather_with_concurrency_streaming, slugify

IS_TTY = sys.stdout.isatty()


def get_endpoints() -> list[Endpoint]:
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

    def get_pending_queries(self, token: str) -> int:
        """Return number of queries still needed for this token."""
        existing = sum(self.results.get(token, {}).values())
        return max(0, config.bi.phase_1.queries_per_token - existing)

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
    await state.rate_limiter.acquire()
    state.request_timestamps.append(time.monotonic())

    def on_retry(status: int) -> None:
        if status == 429:
            state.rate_limit_timestamps.append(time.monotonic())

    response = await client.query(
        state.endpoint, token, temperature=0, logprobs=False, on_retry=on_retry
    )
    state.completed_queries += 1
    if response.error:
        logger.warning(
            f"Error for {state.endpoint}: {token!r}: {response.error.message}"
        )
        return
    state.record_result(token, response.content)


def print_status(states: list[EndpointState], num_lines_to_clear: int = 0) -> int:
    """Print live status for each endpoint. Returns number of lines printed."""
    if num_lines_to_clear > 0:
        # Move cursor up and clear lines
        sys.stdout.write(f"\033[{num_lines_to_clear}A\033[J")

    lines = []
    for state in states:
        rps = state.get_requests_per_second()
        rate_limits = state.get_recent_rate_limits()
        completed_tokens = state.get_completed_tokens()
        total_tokens = len(state.input_tokens)
        pct = 100 * completed_tokens / total_tokens if total_tokens else 0
        lines.append(
            f"  {str(state.endpoint):100} {rps:5.2f} req/s  {rate_limits:2} 429s  "
            f"{state.completed_queries:5}/{state.total_queries} reqs  "
            f"{completed_tokens:4}/{total_tokens} tokens ({pct:5.1f}%)"
        )

    output = "\n".join(lines)
    sys.stdout.write(output + "\n")
    sys.stdout.flush()
    return len(lines)


async def live_status_updater(
    states: list[EndpointState], stop_event: asyncio.Event
) -> None:
    """Background task that updates status display every second."""
    num_lines = 0
    while not stop_event.is_set():
        num_lines = print_status(states, num_lines)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


async def main() -> None:
    config.bi.phase_1_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer index and computing fallback tokens...")
    tokenizer_index = load_existing_index()
    fallback_tokens = get_best_single_token_strings()
    logger.info(
        f"Loaded {len(tokenizer_index)} tokenizers, {len(fallback_tokens)} fallback tokens"
    )

    endpoints = get_endpoints()[:4]
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

    # Start live status updater if in TTY
    stop_event = asyncio.Event()
    status_task = None
    if IS_TTY:
        print()  # Blank line before status
        status_task = asyncio.create_task(live_status_updater(states, stop_event))

    # Concurrency per endpoint limited by rate limiter; global concurrency just needs headroom
    total_concurrency = len(endpoints) * 50
    async for _ in gather_with_concurrency_streaming(total_concurrency, *coros):
        pass

    # Stop status updater
    if status_task:
        stop_event.set()
        await status_task
        print_status(states, len(states))  # Final update

    logger.info("Phase 1 complete")


if __name__ == "__main__":
    asyncio.run(main())
