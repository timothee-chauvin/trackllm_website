"""Phase 2: Repeatedly sample border inputs to detect model changes."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NewType

import orjson
from aiolimiter import AsyncLimiter

from trackllm_website.api import OpenRouterClient
from trackllm_website.config import Endpoint, config, logger
from trackllm_website.util import slugify

Prompt = NewType("Prompt", str)
Timestamp = NewType("Timestamp", str)
ResponseToken = NewType("ResponseToken", str)


def get_output_path(endpoint: Endpoint, year_month: str) -> Path:
    """Get the output JSON path for an endpoint."""
    endpoint_dir = config.bi.phase_2_dir / slugify(
        f"{endpoint.model}#{endpoint.provider}"
    )
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    return endpoint_dir / f"{year_month}.json"


def load_existing_results(
    path: Path,
) -> dict[Prompt, dict[Timestamp, list[tuple[Timestamp, ResponseToken]]]]:
    """Load existing results from JSON file."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def save_results(
    path: Path,
    results: dict[Prompt, dict[Timestamp, list[tuple[Timestamp, ResponseToken]]]],
) -> None:
    """Save results to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))


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

    results: dict[Prompt, dict[Timestamp, list[tuple[Timestamp, ResponseToken]]]] = (
        field(default_factory=dict)
    )
    _current_batch: dict[Prompt, list[tuple[Timestamp, ResponseToken]]] = field(
        default_factory=dict
    )
    _query_count: int = 0

    def __post_init__(self) -> None:
        self.results = load_existing_results(self.output_path)
        self._current_batch = {prompt: [] for prompt in self.border_inputs}

    def record_result(
        self, prompt: Prompt, content: ResponseToken | None, timestamp: Timestamp
    ) -> None:
        """Record a query result into the current day's batch."""
        if content is not None:
            self._current_batch[prompt].append((timestamp, content))
        self._query_count += 1
        if self._query_count >= 20:
            self.flush()

    def flush(self) -> None:
        """Merge current batch into results and save to disk."""
        batch_key = self.start_timestamp
        for prompt, timestamped_responses in self._current_batch.items():
            if prompt not in self.results:
                self.results[prompt] = {}
            self.results[prompt][batch_key] = timestamped_responses
        save_results(self.output_path, self.results)
        self._query_count = 0


async def query_single(
    client: OpenRouterClient,
    state: EndpointState,
    prompt: Prompt,
) -> None:
    """Execute a single query."""
    await state.rate_limiter.acquire()

    now = datetime.now(tz=timezone.utc).replace(microsecond=0)
    timestamp = Timestamp(now.isoformat())
    response = await client.query(
        state.endpoint,
        prompt,
        temperature=0.0,
        logprobs=False,
    )
    if response.error:
        logger.warning(
            f"Error for {state.endpoint}: {prompt!r}: {response.error.message}"
        )
        return
    content = ResponseToken(response.content) if response.content else None
    state.record_result(prompt, content, timestamp)


async def query_all_for_prompt(
    client: OpenRouterClient,
    state: EndpointState,
    prompt: Prompt,
) -> None:
    """Query endpoint for all pending queries for a single prompt, with delays between requests."""
    n_queries = config.bi.phase_2.queries_per_token
    for i in range(n_queries):
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

    endpoints = config.endpoints_bi_phase_1
    now = datetime.now(tz=timezone.utc).replace(microsecond=0)
    year_month = now.strftime("%Y-%m")
    start_timestamp = Timestamp(now.isoformat())

    requests_per_second = config.bi.phase_2.requests_per_second_per_endpoint
    max_concurrent_requests = config.bi.phase_2.max_concurrent_requests_per_endpoint

    states = []
    for ep in endpoints:
        endpoint_key = str(ep)
        border_inputs = border_inputs_by_endpoint[endpoint_key]
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
