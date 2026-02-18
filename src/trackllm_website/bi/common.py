"""Common utilities for BI (border input) experiments."""

import asyncio
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import orjson
from aiolimiter import AsyncLimiter
from beartype.typing import Callable
from tqdm import tqdm

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.download_tokenizers import (
    get_best_single_token_strings,
    load_existing_index,
    load_tokenizer_vocab,
)
from trackllm_website.config import Endpoint, logger
from trackllm_website.util import slugify

SAVE_INTERVAL = 5
_file_semaphore: asyncio.Semaphore | None = None


def _get_file_semaphore() -> asyncio.Semaphore:
    """Get or create the global file semaphore (max 50 concurrent file ops)."""
    global _file_semaphore
    if _file_semaphore is None:
        _file_semaphore = asyncio.Semaphore(50)
    return _file_semaphore


def get_input_tokens(
    endpoint: Endpoint,
    tokenizer_index: dict[str, str],
    fallback_tokens: list[str],
    num_tokens: int,
) -> list[str]:
    """Get input tokens for an endpoint, using its tokenizer if known, in a fixed random order."""
    if endpoint.model in tokenizer_index:
        vocab = load_tokenizer_vocab(tokenizer_index[endpoint.model], shuffle=True)
        return vocab[:num_tokens]
    else:
        return fallback_tokens[:num_tokens]


def load_existing_results(path: Path) -> dict[int, dict[str, list[str]]]:
    """Load existing results from JSON file."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    # Convert string keys to int (JSON only supports string keys)
    return {int(k): v for k, v in data.items()}


async def save_results(path: Path, results: dict[int, dict[str, list[str]]]) -> None:
    """Save results to JSON file."""
    async with _get_file_semaphore():
        path.parent.mkdir(parents=True, exist_ok=True)
        results_serializable = {str(k): v for k, v in results.items()}

        # Write to a temporary file in the same directory, then atomically replace
        with tempfile.NamedTemporaryFile(
            "wb", delete=False, dir=path.parent
        ) as tmp_file:
            tmp_file.write(orjson.dumps(results_serializable))
            temp_name = tmp_file.name
        os.replace(temp_name, path)


def load_tokenizers() -> tuple[dict[str, str], list[str]]:
    """Load tokenizer index and fallback tokens. Returns (tokenizer_index, fallback_tokens)."""
    logger.info("Loading tokenizer index and computing fallback tokens...")
    tokenizer_index = load_existing_index()
    fallback_tokens = get_best_single_token_strings()
    logger.info(
        f"Loaded {len(tokenizer_index)} tokenizers, {len(fallback_tokens)} fallback tokens"
    )
    return tokenizer_index, fallback_tokens


def get_output_path(endpoint: Endpoint, temperature: float, base_dir: Path) -> Path:
    """Get the output JSON path for an endpoint at a specific temperature."""
    filename = f"{slugify(f'{endpoint.model}#{endpoint.provider}')}.json"
    return base_dir / f"T={temperature:g}" / filename


@dataclass
class TemperatureResults:
    """Results for a single temperature."""

    temperature: float
    output_path: Path
    results: dict[int, dict[str, list[str]]] = field(default_factory=dict)
    _prompt_query_counts: dict[str, int] = field(default_factory=dict)
    _prompt_unique_outputs: dict[str, set[str]] = field(default_factory=dict)
    _unsaved_count: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.results = load_existing_results(self.output_path)
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        self._prompt_query_counts = {}
        self._prompt_unique_outputs = {}
        for token_results in self.results.values():
            for token, outputs in token_results.items():
                self._prompt_query_counts[token] = len(outputs)
                self._prompt_unique_outputs[token] = set(outputs)

    async def record_result(
        self, token: str, content: str | None, num_input_tokens: int
    ) -> None:
        if content is not None:
            async with self._lock:
                if num_input_tokens not in self.results:
                    self.results[num_input_tokens] = {}
                results = self.results[num_input_tokens]
                if token not in results:
                    results[token] = []
                results[token].append(content)
                self._prompt_query_counts[token] = (
                    self._prompt_query_counts.get(token, 0) + 1
                )
                if token not in self._prompt_unique_outputs:
                    self._prompt_unique_outputs[token] = set()
                self._prompt_unique_outputs[token].add(content)
                self._unsaved_count += 1
                if self._unsaved_count >= SAVE_INTERVAL:
                    await self._flush_unlocked()

    async def _flush_unlocked(self) -> None:
        """Flush without acquiring lock (caller must hold lock)."""
        if self._unsaved_count > 0:
            self._unsaved_count = 0
            await save_results(self.output_path, self.results)

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_unlocked()


@dataclass
class EndpointState:
    """Tracks state for a single endpoint across one or more temperatures."""

    endpoint: Endpoint
    input_tokens: list[str]
    temperatures: list[float]
    base_dir: Path
    rate_limiter: AsyncLimiter
    concurrency_semaphore: asyncio.Semaphore
    pending_before_new_semaphore: asyncio.Semaphore
    queries_per_token: int
    request_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    rate_limit_timestamps: deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    completed_queries: int = 0
    total_queries: int = 0
    got_404: bool = False
    recent_costs: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    _temp_results: dict[float, TemperatureResults] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for temp in self.temperatures:
            output_path = get_output_path(self.endpoint, temp, self.base_dir)
            self._temp_results[temp] = TemperatureResults(
                temperature=temp, output_path=output_path
            )

    def get_temp_results(self, temperature: float) -> TemperatureResults:
        return self._temp_results[temperature]

    def get_requests_per_second(self) -> float:
        """Calculate actual requests per second over the last few seconds."""
        if len(self.request_timestamps) < 2:
            return 0.0
        now = time.monotonic()
        cutoff = now - 5.0
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
        """Count (token, temperature) tuples that have all queries completed."""
        return sum(
            1
            for token in self.input_tokens
            for temp in self.temperatures
            if self._temp_results[temp]._prompt_query_counts.get(token, 0)
            >= self.queries_per_token
        )

    def get_border_tokens_count(self) -> int:
        """Count tokens that have at least two different outputs (at any temperature)."""
        return len(self.get_border_tokens())

    def get_border_tokens(self) -> list[str]:
        """Get list of border inputs (at any temperature)."""
        return list(
            {
                token
                for token in self.input_tokens
                for temp in self.temperatures
                if len(
                    self._temp_results[temp]._prompt_unique_outputs.get(token, set())
                )
                >= 2
            }
        )

    def get_pending_queries_per_temp(self, prompt: str) -> dict[float, int]:
        """Return number of queries still needed for this prompt at each temperature."""
        return {
            temp: max(
                0,
                self.queries_per_token
                - self._temp_results[temp]._prompt_query_counts.get(prompt, 0),
            )
            for temp in self.temperatures
        }

    def get_unfinished_prompts(self) -> list[tuple[str, dict[float, int]]]:
        """Get list of (prompt, {temp: pending_count}) for prompts that still need queries."""
        result = []
        for t in self.input_tokens:
            pending_per_temp = self.get_pending_queries_per_temp(t)
            if any(p > 0 for p in pending_per_temp.values()):
                result.append((t, pending_per_temp))
        return result

    async def record_result(
        self, temperature: float, token: str, content: str | None, num_input_tokens: int
    ) -> None:
        await self._temp_results[temperature].record_result(
            token, content, num_input_tokens
        )

    async def flush(self) -> None:
        for temp_results in self._temp_results.values():
            await temp_results.flush()


def log_status(states: list[EndpointState]) -> None:
    """Log status for each endpoint in a dynamic table format."""
    total_estimated_cost = 0.0

    if states:
        max_name_len = max(len(str(s.endpoint)) for s in states)
        col_width = max(max_name_len + 2, 20)
    else:
        col_width = 20

    fmt = f"{{:<{col_width}}} {{:>20}} {{:>18}} {{:>10}} {{:>10}} {{:>12}}"
    headers = ["Endpoint", "Tokens", "Border (BI%)", "RPS", "429s", "Est. Cost"]
    separator_len = col_width + 75

    separator = "-" * separator_len

    logger.info(separator)
    logger.info(fmt.format(*headers))
    logger.info(separator)

    for state in states:
        completed_tokens = state.get_completed_tokens()
        border_tokens = state.get_border_tokens_count()
        total_tokens = len(state.input_tokens) * len(state.temperatures)
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

        token_str = f"{completed_tokens}/{total_tokens}"
        border_str = f"{border_tokens} ({bi_pct:.1%})"
        rps_str = f"{rps:.1f}"
        cost_str = f"${estimated_cost:.4f}"

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

    logger.info(separator)
    logger.info(f"Total estimated cost: ${total_estimated_cost:.4f}")


async def query_single(
    client: OpenRouterClient,
    state: EndpointState,
    token: str,
    temperature: float,
) -> bool:
    """Execute a single query. Returns False if a 404 error is encountered."""
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
        temperature=temperature,
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
    await state.record_result(
        temperature, token, response.content, response.input_tokens
    )
    return True


async def query_all_for_token(
    client: OpenRouterClient,
    state: EndpointState,
    token: str,
    temp_pending: dict[float, int],
    pbar: tqdm,
    request_delay_seconds: float,
    stop_early: Callable[[EndpointState], bool] | None = None,
) -> None:
    """Query endpoint for all pending queries for a single token, with delays between requests.

    Interleaves queries across temperatures: for each query round, queries all temperatures
    that still need queries before moving to the next round.
    """
    total_pending = sum(temp_pending.values())
    if stop_early and stop_early(state):
        pbar.update(total_pending)
        return

    # Copy so we can mutate
    temp_pending = dict(temp_pending)
    max_rounds = max(temp_pending.values()) if temp_pending else 0

    async with state.pending_before_new_semaphore:
        for i in range(max_rounds):
            if state.got_404:
                return
            for temp in state.temperatures:
                if state.got_404:
                    return
                if temp_pending[temp] <= 0:
                    continue
                async with state.concurrency_semaphore:
                    success = await query_single(client, state, token, temp)
                temp_pending[temp] -= 1
                pbar.update(1)
                if not success:
                    return
                if stop_early and stop_early(state):
                    return
            if i < max_rounds - 1:
                await asyncio.sleep(request_delay_seconds)


async def run_queries(
    states: list[EndpointState],
    pending_lists: list[list[tuple[str, dict[float, int]]]],
    request_delay_seconds: float,
    stop_early: Callable[[EndpointState], bool] | None = None,
) -> None:
    """Run queries for all states with interleaved scheduling."""
    for state, pending in zip(states, pending_lists):
        state.total_queries = sum(
            sum(temp_pending.values()) for _, temp_pending in pending
        )

    client = OpenRouterClient()
    total_requests = sum(s.total_queries for s in states)
    logger.info(f"Total requests to process: {total_requests}")

    try:

        async def run_with_status(pbar: tqdm) -> None:
            coros = []
            max_tokens = max(len(p) for p in pending_lists) if pending_lists else 0
            for i in range(max_tokens):
                for state, pending in zip(states, pending_lists):
                    if i < len(pending):
                        token, temp_pending = pending[i]
                        coros.append(
                            query_all_for_token(
                                client,
                                state,
                                token,
                                temp_pending,
                                pbar,
                                request_delay_seconds,
                                stop_early,
                            )
                        )

            last_status_time = time.monotonic()
            for future in asyncio.as_completed(coros):
                await future
                now = time.monotonic()
                if now - last_status_time >= 5.0:
                    log_status(states)
                    last_status_time = now

        with tqdm(total=total_requests, desc="Requests") as pbar:
            await run_with_status(pbar)
    finally:
        for state in states:
            await state.flush()
        await client.close()

    log_status(states)

    for state in states:
        incomplete = len(state.input_tokens) - state.get_completed_tokens()
        if incomplete > 0:
            logger.info(f"{state.endpoint}: {incomplete} incomplete tokens")
