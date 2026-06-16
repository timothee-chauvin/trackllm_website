"""Phase 1: Identify border inputs by querying endpoints with single-token inputs."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import orjson
from aiolimiter import AsyncLimiter

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.common import (
    EndpointState,
    QueryStrategy,
    get_input_tokens,
    load_tokenizers,
    resolve_strategies,
    run_queries,
)
from trackllm_website.bi.sampling import sample_prompts
from trackllm_website.config import Endpoint, config, logger


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


async def phase_1a(
    endpoints: list[Endpoint], temperature: float, base_dir: Path | None
) -> None:
    """Phase 1a: Identify candidate border inputs."""
    if base_dir is None:
        base_dir = config.bi.data_dir / "phase_1"
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running phase 1a with temperature={temperature:g}")
    tokenizer_index, fallback_tokens = load_tokenizers()

    async with OpenRouterClient(timeout=60.0) as probe_client:
        strategies, _failed = await resolve_strategies(probe_client, endpoints)

    valid_endpoints = [ep for ep in endpoints if str(ep) in strategies]
    skipped = len(endpoints) - len(valid_endpoints)
    if skipped:
        logger.info(
            f"Skipped {skipped} endpoints (no working strategy / hidden reasoning)"
        )
    logger.info(f"Running phase 1a for {len(valid_endpoints)} endpoints")

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
            query_strategy=strategies[str(ep)],
        )
        for ep in valid_endpoints
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

    endpoints = config.endpoints_bi
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


def temperature_is_ignored(
    t0_distinct: dict[str, int], t1_distinct: dict[str, int]
) -> bool:
    """True if raising temperature 0->1 does NOT broaden the output distribution.

    Honored endpoints show strictly more diversity at T=1 on at least some prompts;
    if T=1 never exceeds T=0, temperature is a no-op.
    """
    return all(t1_distinct.get(p, 0) <= n for p, n in t0_distinct.items())


async def check_temperature(
    client,
    endpoint: Endpoint,
    strategy: QueryStrategy | None,
    border_prompts: list[str],
) -> bool:
    """Re-sample border prompts at T=0 and T=1; True if temperature is ignored.

    A high border-input prevalence is only meaningful if T=0 actually pins the
    output. Endpoints that ignore temperature (e.g. some reasoning models) produce
    fake border inputs with no detection power, so this gate excludes them.
    """
    # Worst case config.bi.temperature_gate.check_prompts * check_samples * 2 queries
    # (T=0 and T=1). For reasoning endpoints each query bills the full reasoning
    # budget as output, so this one-time onboarding probe can be the expensive path
    # for exactly the endpoints it targets (reasoning models that ignore temperature).
    gate = config.bi.temperature_gate
    prompts = border_prompts[: gate.check_prompts]

    async def distinct_at(temperature: float) -> dict[str, int]:
        samples, _ = await sample_prompts(
            client, endpoint, strategy, prompts, gate.check_samples, temperature
        )
        return {p: len({tok for _, tok in samples[p]}) for p in samples}

    t0_distinct = await distinct_at(0.0)
    t1_distinct = await distinct_at(1.0)
    ignored = temperature_is_ignored(t0_distinct, t1_distinct)
    if ignored:
        logger.warning(
            f"{endpoint}: temperature ignored (T=0 {t0_distinct} >= T=1 {t1_distinct})"
        )
    return ignored


if __name__ == "__main__":
    TEMPERATURE = 0.0
    asyncio.run(phase_1a(config.endpoints_bi, TEMPERATURE, None))
