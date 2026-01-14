"""Phase 1: Identify border inputs by querying endpoints with single-token inputs."""

import asyncio
import random
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
from trackllm_website.util import slugify


def get_input_tokens(
    endpoint: Endpoint,
    tokenizer_index: dict[str, str],
    fallback_tokens: list[str],
    num_tokens: int,
) -> list[str]:
    """Get input tokens for an endpoint, using its tokenizer if known, in a fixed random order."""
    if endpoint.model in tokenizer_index:
        vocab = load_tokenizer_vocab(tokenizer_index[endpoint.model])
        random.Random(0).shuffle(vocab)
        return vocab[:num_tokens]
    else:
        random.Random(0).shuffle(fallback_tokens)
        return fallback_tokens[:num_tokens]


def get_input_tokens_for_endpoint(
    endpoint: Endpoint,
    tokenizer_index: dict[str, str],
    fallback_tokens: list[str],
) -> list[str]:
    """Get input tokens for an endpoint using phase_1 config."""
    return get_input_tokens(
        endpoint,
        tokenizer_index,
        fallback_tokens,
        config.bi.phase_1.tokens_per_endpoint,
    )


def get_output_path(endpoint: Endpoint, temperature: float) -> Path:
    """Get the output JSON path for an endpoint."""
    return (
        config.bi.get_phase_1_dir(temperature)
        / f"{slugify(f'{endpoint.model}#{endpoint.provider}')}.json"
    )


def load_existing_results(path: Path) -> dict[int, dict[str, dict[str, int]]]:
    """Load existing results from JSON file."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    # Convert string keys to int (JSON only supports string keys)
    return {int(k): v for k, v in data.items()}


def save_results(path: Path, results: dict[int, dict[str, dict[str, int]]]) -> None:
    """Save results to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    results_serializable = {str(k): v for k, v in results.items()}
    with open(path, "wb") as f:
        f.write(orjson.dumps(results_serializable))


@dataclass
class EndpointState:
    """Tracks state for a single endpoint."""

    endpoint: Endpoint
    input_tokens: list[str]
    output_path: Path
    rate_limiter: AsyncLimiter
    concurrency_semaphore: asyncio.Semaphore
    pending_before_new_semaphore: asyncio.Semaphore
    temperature: float
    results: dict[int, dict[str, dict[str, int]]] = field(default_factory=dict)
    request_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    rate_limit_timestamps: deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    completed_queries: int = 0
    total_queries: int = 0
    got_404: bool = False
    reached_target: bool = False
    recent_costs: deque[float] = field(default_factory=lambda: deque(maxlen=20))

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

    def _get_all_prompt_results(self) -> dict[str, dict[str, int]]:
        """Get merged results across all input token counts."""
        merged: dict[str, dict[str, int]] = {}
        for token_results in self.results.values():
            for token, outputs in token_results.items():
                if token not in merged:
                    merged[token] = {}
                for output, count in outputs.items():
                    merged[token][output] = merged[token].get(output, 0) + count
        return merged

    def get_completed_tokens(self) -> int:
        """Count tokens that have all queries completed."""
        queries_needed = config.bi.phase_1.queries_per_token
        results = self._get_all_prompt_results()
        return sum(
            1
            for token in self.input_tokens
            if sum(results.get(token, {}).values()) >= queries_needed
        )

    def get_border_tokens(self) -> int:
        """Count tokens that have at least two different outputs."""
        results = self._get_all_prompt_results()
        return sum(1 for token, outputs in results.items() if len(outputs) >= 2)

    def get_pending_queries(self, prompt: str) -> int:
        """Return number of queries still needed for this prompt."""
        results = self._get_all_prompt_results()
        existing = sum(results.get(prompt, {}).values())
        return max(0, config.bi.phase_1.queries_per_token - existing)

    def get_unfinished_prompts(self) -> list[tuple[str, int]]:
        """Get list of (prompt, pending_count) for prompts that still need queries."""
        return [
            (t, pending)
            for t in self.input_tokens
            if (pending := self.get_pending_queries(t)) > 0
        ]

    def record_result(
        self, token: str, content: str | None, num_input_tokens: int
    ) -> None:
        """Record a query result and save."""
        if content is not None:
            if num_input_tokens not in self.results:
                self.results[num_input_tokens] = {}
            results = self.results[num_input_tokens]
            if token not in results:
                results[token] = {}
            results[token][content] = results[token].get(content, 0) + 1
            save_results(self.output_path, self.results)


async def query_single(
    client: OpenRouterClient,
    state: EndpointState,
    token: str,
) -> bool:
    """Execute a single query. Returns False if a 404 error is encountered."""
    if state.reached_target:
        return False

    if state.got_404:
        return False

    await state.rate_limiter.acquire()

    if state.got_404:
        return False

    state.request_timestamps.append(time.monotonic())

    def on_retry(status: int) -> None:
        if status == 429:
            state.rate_limit_timestamps.append(time.monotonic())

    response = await client.query(
        state.endpoint,
        token,
        temperature=state.temperature,
        logprobs=False,
        on_retry=on_retry,
    )
    state.completed_queries += 1
    if response.error:
        if response.error.http_code == 404:
            if not state.got_404:
                state.got_404 = True
                logger.warning(f"Got 404 for {state.endpoint}, abandoning endpoint")
            return False
        logger.warning(
            f"Error for {state.endpoint}: {token!r}: {response.error.message}"
        )
        return True
    state.recent_costs.append(response.cost)
    state.record_result(token, response.content, response.input_tokens)
    if (
        state.get_border_tokens()
        >= config.bi.phase_1.border_input_candidate_ratio
        * config.bi.phase_1.target_border_inputs
    ):
        if state.reached_target is False:
            logger.info(f"Reached target border inputs for {state.endpoint}")
        state.reached_target = True
    return True


async def query_all_for_token(
    client: OpenRouterClient,
    state: EndpointState,
    token: str,
    pending: int,
    pbar: tqdm,
) -> None:
    """Query endpoint for all pending queries for a single token, with delays between requests."""
    delay = config.bi.phase_1.request_delay_seconds
    if state.reached_target:
        pbar.update(pending)
        return

    async with state.pending_before_new_semaphore:
        for i in range(pending):
            if state.got_404:
                return
            async with state.concurrency_semaphore:
                success = await query_single(client, state, token)
            pbar.update(1)
            if not success:
                return
            if i < pending - 1:
                await asyncio.sleep(delay)


def log_status(states: list[EndpointState]) -> None:
    """Log status for each endpoint in a dynamic table format."""
    total_estimated_cost = 0.0

    # 1. Calculate dynamic width for the Endpoint column
    # We default to 20 if the list is empty or names are short to keep a minimum width
    if states:
        max_name_len = max(len(str(s.endpoint)) for s in states)
        col_width = max(max_name_len + 2, 20)  # +2 for padding
    else:
        col_width = 20

    # 2. Define Dynamic Format String
    # {col_width} injects the calculated width into the format string
    fmt = f"{{:<{col_width}}} {{:>20}} {{:>18}} {{:>10}} {{:>10}} {{:>12}}"

    # 3. Print Header
    headers = ["Endpoint", "Tokens", "Border (BI%)", "RPS", "429s", "Est. Cost"]

    # Calculate separator length based on the sum of fixed columns (approx 70) + dynamic col_width
    separator_len = col_width + 75
    separator = "-" * separator_len

    logger.info(separator)
    logger.info(fmt.format(*headers))
    logger.info(separator)

    for state in states:
        completed_tokens = state.get_completed_tokens()
        border_tokens = state.get_border_tokens()
        total_tokens = len(state.input_tokens)
        rate_limits = state.get_recent_rate_limits()
        rps = state.get_requests_per_second()

        bi_pct = border_tokens / completed_tokens if completed_tokens else 0

        avg_cost = (
            sum(state.recent_costs) / len(state.recent_costs)
            if state.recent_costs
            else 0
        )
        estimated_cost = avg_cost * state.total_queries
        total_estimated_cost += estimated_cost

        # --- Formatting Data Fields ---
        token_str = f"{completed_tokens}/{total_tokens}"
        border_str = f"{border_tokens} ({bi_pct:.1%})"
        rps_str = f"{rps:.1f}"
        cost_str = f"${estimated_cost:.4f}"

        # --- Log the Row ---
        logger.info(
            fmt.format(
                str(state.endpoint),
                token_str,
                border_str,
                rps_str,
                rate_limits,
                cost_str,
            )
        )

    # 4. Footer
    logger.info(separator)
    logger.info(f"Total estimated cost: ${total_estimated_cost:.4f}")


async def main(temperature: float) -> None:
    phase_1_dir = config.bi.get_phase_1_dir(temperature)
    phase_1_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running phase 1 with temperature={temperature:g}")
    logger.info("Loading tokenizer index and computing fallback tokens...")
    tokenizer_index = load_existing_index()
    fallback_tokens = get_best_single_token_strings()
    logger.info(
        f"Loaded {len(tokenizer_index)} tokenizers, {len(fallback_tokens)} fallback tokens"
    )

    endpoints = config.endpoints_bi_phase_1
    logger.info(f"Running phase 1 for {len(endpoints)} endpoints")

    requests_per_second = config.bi.phase_1.requests_per_second_per_endpoint
    max_concurrent_requests = config.bi.phase_1.max_concurrent_requests_per_endpoint
    max_concurrent_tokens = config.bi.phase_1.max_concurrent_tokens_per_endpoint

    # Create state for each endpoint with its own rate limiter and concurrency semaphore
    states = [
        EndpointState(
            endpoint=ep,
            input_tokens=get_input_tokens_for_endpoint(
                ep, tokenizer_index, fallback_tokens
            ),
            output_path=get_output_path(ep, temperature),
            rate_limiter=AsyncLimiter(requests_per_second, 1),
            concurrency_semaphore=asyncio.Semaphore(max_concurrent_requests),
            pending_before_new_semaphore=asyncio.Semaphore(max_concurrent_tokens),
            temperature=temperature,
        )
        for ep in endpoints
    ]

    # Get pending tokens (with their pending counts) for each endpoint
    pending_lists = [s.get_unfinished_prompts() for s in states]

    # Set total_queries for each state
    for state, pending in zip(states, pending_lists):
        state.total_queries = sum(count for _, count in pending)

    client = OpenRouterClient()
    total_requests = sum(s.total_queries for s in states)
    logger.info(f"Total requests to process: {total_requests}")

    # Create one task per token (each task handles all pending queries for that token)
    async def run_with_status(pbar: tqdm) -> None:
        coros = []
        # Interleave tokens from all endpoints
        max_tokens = max(len(p) for p in pending_lists) if pending_lists else 0
        for i in range(max_tokens):
            for state, pending in zip(states, pending_lists):
                if i < len(pending):
                    token, count = pending[i]
                    coros.append(query_all_for_token(client, state, token, count, pbar))

        last_status_time = time.monotonic()
        for future in asyncio.as_completed(coros):
            await future
            now = time.monotonic()
            if now - last_status_time >= 5.0:
                log_status(states)
                last_status_time = now

    with tqdm(total=total_requests, desc="Requests") as pbar:
        await run_with_status(pbar)

    log_status(states)

    # Print incomplete tokens per model (due to errors like 404)
    for state in states:
        incomplete = len(state.input_tokens) - state.get_completed_tokens()
        if incomplete > 0:
            logger.info(f"{state.endpoint}: {incomplete} incomplete tokens")

    logger.info("Phase 1 complete")


if __name__ == "__main__":
    TEMPERATURE = 0.0
    asyncio.run(main(TEMPERATURE))
