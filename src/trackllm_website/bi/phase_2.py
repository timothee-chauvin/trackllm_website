"""Phase 2: Repeatedly sample border inputs to detect model changes."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import orjson
from aiolimiter import AsyncLimiter

from trackllm_website.api import OpenRouterClient
from trackllm_website.config import Endpoint, config, logger
from trackllm_website.util import slugify


def get_output_path(endpoint: Endpoint, year_month: str) -> Path:
    """Get the output JSON path for an endpoint."""
    endpoint_dir = config.bi.phase_2_dir / slugify(
        f"{endpoint.model}#{endpoint.provider}"
    )
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    return endpoint_dir / f"{year_month}.json"


def load_existing_results(path: Path) -> dict[str, list[dict[str, str]]]:
    """Load existing results from JSON file."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def save_results(path: Path, results: dict[str, list[dict[str, str]]]) -> None:
    """Save results to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))


def load_border_inputs(temperature: float) -> dict[str, list[str]]:
    """Load border inputs from phase_1b output."""
    phase_1_dir = config.bi.get_phase_1_dir(temperature)
    border_inputs_path = phase_1_dir / "border_inputs.json"
    if not border_inputs_path.exists():
        raise FileNotFoundError(
            f"Border inputs file not found: {border_inputs_path}. Run phase_1b first."
        )
    with open(border_inputs_path, "rb") as f:
        return orjson.loads(f.read())


SAVE_INTERVAL = 20


@dataclass
class EndpointState:
    """Tracks state for a single endpoint."""

    endpoint: Endpoint
    border_inputs: list[str]
    output_path: Path
    rate_limiter: AsyncLimiter
    concurrency_semaphore: asyncio.Semaphore
    results: dict[str, list[list[dict[str, str]]]] = field(default_factory=dict)
    _unsaved_count: int = 0

    def __post_init__(self) -> None:
        self.results = load_existing_results(self.output_path)

    def record_result(self, prompt: str, content: str | None, timestamp: str) -> None:
        """Record a query result. Saves periodically."""
        if content is not None:
            if prompt not in self.results:
                self.results[prompt] = []
            self.results[prompt].append({timestamp: content})
            self._unsaved_count += 1
            if self._unsaved_count >= SAVE_INTERVAL:
                self.flush()

    def flush(self) -> None:
        """Save results to disk if there are unsaved changes."""
        if self._unsaved_count > 0:
            self._unsaved_count = 0
            save_results(self.output_path, self.results)


async def query_single(
    client: OpenRouterClient,
    state: EndpointState,
    prompt: str,
) -> None:
    """Execute a single query."""
    await state.rate_limiter.acquire()

    now = datetime.now(tz=timezone.utc).replace(microsecond=0)
    timestamp = now.isoformat()
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
    state.record_result(prompt, response.content, timestamp)


async def query_all_for_prompt(
    client: OpenRouterClient,
    state: EndpointState,
    prompt: str,
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
    year_month = datetime.now(tz=timezone.utc).strftime("%Y-%m")

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
