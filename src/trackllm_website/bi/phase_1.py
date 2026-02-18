"""Phase 1: Identify border inputs by querying endpoints with single-token inputs."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import orjson
from aiolimiter import AsyncLimiter

from trackllm_website.bi.common import (
    EndpointState,
    get_input_tokens,
    load_tokenizers,
    run_queries,
)
from trackllm_website.config import config, logger


@dataclass
class Phase1EndpointState(EndpointState):
    """Extended EndpointState with phase 1 early stopping logic."""

    reached_target: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.update_reached_target()

    def update_reached_target(self) -> None:
        if (
            self.get_border_tokens_count()
            >= config.bi.phase_1.border_input_candidate_ratio
            * config.bi.phase_1.target_border_inputs
        ):
            if self.reached_target is False:
                logger.info(f"Reached target border inputs for {self.endpoint}")
            self.reached_target = True

    def get_unfinished_border_inputs(self) -> list[tuple[str, int]]:
        """Get list of (prompt, pending_count) for border inputs that still need queries."""
        border_inputs = self.get_border_tokens()
        results = []
        for token in border_inputs:
            first_temp = self._temp_results[self.temperatures[0]]
            pending = (
                config.bi.phase_1.queries_per_candidate
                - first_temp._prompt_query_counts.get(token, 0)
            )
            if pending > 0:
                results.append((token, pending))
        return results


def stop_early_phase1(state: EndpointState) -> bool:
    """Check if we should stop early for phase 1 (reached target border inputs)."""
    if isinstance(state, Phase1EndpointState):
        state.update_reached_target()
        return state.reached_target
    return False


async def phase_1a(temperature: float, base_dir: Path | None = None) -> None:
    """Phase 1a: Identify candidate border inputs."""
    if base_dir is None:
        base_dir = config.bi.data_dir / "phase_1"
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running phase 1a with temperature={temperature:g}")
    tokenizer_index, fallback_tokens = load_tokenizers()

    endpoints = config.endpoints_bi_phase_1
    logger.info(f"Running phase 1a for {len(endpoints)} endpoints")

    requests_per_second = config.bi.phase_1.requests_per_second_per_endpoint
    max_concurrent_requests = config.bi.phase_1.max_concurrent_requests_per_endpoint
    max_concurrent_tokens = config.bi.phase_1.max_concurrent_tokens_per_endpoint

    states = [
        Phase1EndpointState(
            endpoint=ep,
            input_tokens=get_input_tokens(
                ep,
                tokenizer_index,
                fallback_tokens,
                config.bi.phase_1.tokens_per_endpoint,
            ),
            temperatures=[temperature],
            base_dir=base_dir,
            rate_limiter=AsyncLimiter(requests_per_second, 1),
            concurrency_semaphore=asyncio.Semaphore(max_concurrent_requests),
            pending_before_new_semaphore=asyncio.Semaphore(max_concurrent_tokens),
            queries_per_token=config.bi.phase_1.queries_per_token,
        )
        for ep in endpoints
    ]

    pending_lists = [s.get_unfinished_prompts() for s in states]

    await run_queries(
        states,
        pending_lists,
        config.bi.phase_1.request_delay_seconds,
        stop_early=stop_early_phase1,
    )

    logger.info("Phase 1a complete")


async def phase_1b(temperature: float, base_dir: Path | None = None) -> None:
    """Phase 1b: Perform additional sampling of the candidate border inputs and select the best ones."""
    if base_dir is None:
        base_dir = config.bi.data_dir / "phase_1"
    logger.info(f"Running phase 1b with temperature={temperature:g}")
    tokenizer_index, fallback_tokens = load_tokenizers()

    endpoints = config.endpoints_bi_phase_1
    logger.info(f"Running phase 1b for {len(endpoints)} endpoints")

    requests_per_second = config.bi.phase_1.requests_per_second_per_endpoint
    max_concurrent_requests = config.bi.phase_1.max_concurrent_requests_per_endpoint
    max_concurrent_tokens = config.bi.phase_1.max_concurrent_tokens_per_endpoint

    states = [
        Phase1EndpointState(
            endpoint=ep,
            input_tokens=get_input_tokens(
                ep,
                tokenizer_index,
                fallback_tokens,
                config.bi.phase_1.tokens_per_endpoint,
            ),
            temperatures=[temperature],
            base_dir=base_dir,
            rate_limiter=AsyncLimiter(requests_per_second, 1),
            concurrency_semaphore=asyncio.Semaphore(max_concurrent_requests),
            pending_before_new_semaphore=asyncio.Semaphore(max_concurrent_tokens),
            queries_per_token=config.bi.phase_1.queries_per_token,
        )
        for ep in endpoints
    ]

    results = {}
    for state in states:
        # TODO do the additional sampling, this is just a mock
        border_inputs = state.get_border_tokens()
        results[str(state.endpoint)] = border_inputs

    output_path = config.bi.get_phase_1_dir(temperature, base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "border_inputs.json", "wb") as f:
        f.write(orjson.dumps(results))


if __name__ == "__main__":
    TEMPERATURE = 0.0
    asyncio.run(phase_1a(TEMPERATURE))
