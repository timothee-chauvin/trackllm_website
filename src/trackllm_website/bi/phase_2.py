"""Phase 2: Repeatedly sample border inputs to detect model changes."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NewType

import orjson
from aiolimiter import AsyncLimiter

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.common import (
    PlainStrategy,
    QueryStrategy,
    extract_first_token,
    resolve_strategies,
    strategy_to_query_args,
)
from trackllm_website.bi.state import load_all_states
from trackllm_website.config import Endpoint, config, logger
from trackllm_website.util import atomic_write_bytes, slugify

Prompt = NewType("Prompt", str)
Timestamp = NewType("Timestamp", str)
ResponseToken = NewType("ResponseToken", str)

Results = dict[Prompt, dict[Timestamp, list[tuple[Timestamp, ResponseToken]]]]


def get_output_path(endpoint: Endpoint, year_month: str) -> Path:
    """Get the output JSON path for an endpoint."""
    endpoint_dir = config.bi.phase_2_dir / slugify(
        f"{endpoint.model}#{endpoint.provider}"
    )
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    return endpoint_dir / f"{year_month}.json"


def load_existing_results(path: Path) -> Results:
    """Load existing results from JSON file, restoring sample tuples.

    JSON round-trips tuples into lists; without the conversion, re-saving
    loaded results violates the `Results` type hint.
    """
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        raw = orjson.loads(f.read())
    return {
        prompt: {ts: [tuple(s) for s in samples] for ts, samples in batches.items()}
        for prompt, batches in raw.items()
    }


def save_results(
    path: Path,
    results: Results,
) -> None:
    """Save results to JSON file."""
    atomic_write_bytes(path, orjson.dumps(results))


def load_border_inputs(temperature: float) -> dict[str, list[Prompt]]:
    """Load border inputs from phase_1b output."""
    phase_1_dir = config.bi.get_phase_1_dir(temperature)
    border_inputs_path = phase_1_dir / "border_inputs.json"
    if not border_inputs_path.exists():
        raise FileNotFoundError(
            f"Border inputs file not found: {border_inputs_path}. Run phase_1b first."
        )
    with open(border_inputs_path, "rb") as f:
        return orjson.loads(f.read())


@dataclass
class EndpointState:
    """Tracks state for a single endpoint."""

    endpoint: Endpoint
    border_inputs: list[Prompt]
    output_path: Path
    rate_limiter: AsyncLimiter
    concurrency_semaphore: asyncio.Semaphore
    start_timestamp: Timestamp
    query_strategy: QueryStrategy = field(default_factory=PlainStrategy)

    results: Results = field(default_factory=dict)
    _current_batch: dict[Prompt, list[tuple[Timestamp, ResponseToken]]] = field(
        default_factory=dict
    )
    _total_queries: int = 0
    _error_count: int = 0
    abandoned: bool = False

    def __post_init__(self) -> None:
        self.results = load_existing_results(self.output_path)
        self._current_batch = {prompt: [] for prompt in self.border_inputs}

    def record_response(
        self,
        prompt: Prompt,
        content: ResponseToken | None,
        timestamp: Timestamp,
        *,
        error: bool,
    ) -> None:
        self._total_queries += 1
        if error:
            self._record_error()
        else:
            self._record_success(prompt, content, timestamp)
        if self._total_queries % 20 == 0:
            self.flush()

    def _record_error(self) -> None:
        self._error_count += 1
        abandon_after = config.bi.phase_2.abandon_this_run_after
        if (
            not self.abandoned
            and self._total_queries >= abandon_after
            and self._error_count == self._total_queries
        ):
            self.abandoned = True
            logger.warning(
                f"Abandoning {self.endpoint}: all first {self._total_queries} queries were errors"
            )

    def _record_success(
        self, prompt: Prompt, content: ResponseToken | None, timestamp: Timestamp
    ) -> None:
        if content is not None:
            self._current_batch[prompt].append((timestamp, content))

    def flush(self) -> None:
        """Merge current batch into results and save to disk."""
        batch_key = self.start_timestamp
        for prompt, timestamped_responses in self._current_batch.items():
            if prompt not in self.results:
                self.results[prompt] = {}
            self.results[prompt][batch_key] = timestamped_responses
        save_results(self.output_path, self.results)


async def query_single(
    client: OpenRouterClient,
    state: EndpointState,
    prompt: Prompt,
) -> None:
    """Execute a single query."""
    if state.abandoned:
        return
    await state.rate_limiter.acquire()

    now = datetime.now(tz=timezone.utc).replace(microsecond=0)
    timestamp = Timestamp(now.isoformat())
    response = await client.query(
        state.endpoint,
        prompt,
        temperature=0.0,
        logprobs=False,
        **strategy_to_query_args(state.query_strategy),
    )
    if response.error:
        logger.warning(
            f"Error for {state.endpoint}: {prompt!r}: {response.error.message}"
        )
        state.record_response(prompt, None, timestamp, error=True)
        return
    first_tok = extract_first_token(response)
    content = ResponseToken(first_tok) if first_tok else None
    state.record_response(prompt, content, timestamp, error=False)


async def query_all_for_prompt(
    client: OpenRouterClient,
    state: EndpointState,
    prompt: Prompt,
) -> None:
    """Query endpoint for all pending queries for a single prompt, with delays between requests."""
    n_queries = config.bi.phase_2.queries_per_token
    for i in range(n_queries):
        if state.abandoned:
            return
        async with state.concurrency_semaphore:
            await query_single(client, state, prompt)
        if i < n_queries - 1:
            await asyncio.sleep(config.bi.phase_2.request_delay_seconds)


async def phase_2() -> None:
    """Phase 2: Repeatedly sample border inputs to detect model changes."""
    phase_2_dir = config.bi.phase_2_dir
    phase_2_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running phase 2 (T=0)")
    logger.info("Loading border inputs from phase_1b...")
    border_inputs_by_endpoint = load_border_inputs(temperature=0.0)
    logger.info(f"Loaded border inputs for {len(border_inputs_by_endpoint)} endpoints")

    endpoints = [
        s.endpoint
        for s in load_all_states(config.bi.state_dir).values()
        if s.status == "monitoring"
    ]

    async with OpenRouterClient(timeout=60.0) as probe_client:
        strategies, _failed = await resolve_strategies(probe_client, endpoints)

    now = datetime.now(tz=timezone.utc).replace(microsecond=0)
    year_month = now.strftime("%Y-%m")
    start_timestamp = Timestamp(now.isoformat())

    requests_per_second = config.bi.phase_2.requests_per_second_per_endpoint
    max_concurrent_requests = config.bi.phase_2.max_concurrent_requests_per_endpoint

    states = []
    for ep in endpoints:
        endpoint_key = str(ep)
        if endpoint_key not in strategies:
            logger.warning(f"No strategy for {ep}, skipping")
            continue
        border_inputs = border_inputs_by_endpoint.get(endpoint_key, [])
        if not border_inputs:
            logger.warning(f"Empty border inputs for {ep}, skipping")
            continue

        states.append(
            EndpointState(
                endpoint=ep,
                border_inputs=border_inputs,
                output_path=get_output_path(ep, year_month),
                rate_limiter=AsyncLimiter(requests_per_second, 1),
                concurrency_semaphore=asyncio.Semaphore(max_concurrent_requests),
                start_timestamp=start_timestamp,
                query_strategy=strategies[endpoint_key],
            )
        )

    client = OpenRouterClient()
    tasks = [
        query_all_for_prompt(client, state, prompt)
        for state in states
        for prompt in state.border_inputs
    ]

    try:
        i = 0
        for future in asyncio.as_completed(tasks):
            await future
            i += 1
            if i % 100 == 0:
                logger.info(f"Processed {i} / {len(tasks)} queries")

    finally:
        for state in states:
            state.flush()
        await client.close()

    logger.info("Phase 2 complete")


if __name__ == "__main__":
    asyncio.run(phase_2())
