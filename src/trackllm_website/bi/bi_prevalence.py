"""BI prevalence study: Query endpoints at multiple temperatures to measure border input prevalence."""

import asyncio
from pathlib import Path

import yaml
from aiolimiter import AsyncLimiter

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.common import (
    EndpointState,
    PlainStrategy,
    ReasoningBudgetStrategy,
    ReasoningDisabledStrategy,
    get_input_tokens,
    load_strategies,
    load_tokenizers,
    log_status,
    report_cost_to_target,
    resolve_strategies,
    run_queries,
    save_strategies,
)
from trackllm_website.bi.generate_bi_prevalence_endpoints import (
    PREVALENCE_MAX_AVG_COST,
    PREVALENCE_N,
    get_cheap_endpoints,
)
from trackllm_website.config import Endpoint, config, logger, root

TARGET_BORDER_INPUTS = 5


def _stop_early_5_bis(state: EndpointState) -> bool:
    return state.get_border_tokens_count() >= TARGET_BORDER_INPUTS


TRANSIENT_ERROR_CODES = {"429", "0", "500", "502", "503", "504"}


def _has_transient_error(errors: list[str]) -> bool:
    """True if any error looks transient (rate limit, timeout, server error)."""
    for e in errors:
        # Error format: "label: CODE message"
        parts = e.split(": ", 1)
        if len(parts) == 2:
            code = parts[1].split(" ", 1)[0]
            if code in TRANSIENT_ERROR_CODES:
                return True
    return False


def _save_prevalence_endpoints(endpoints: list[Endpoint]) -> None:
    output_data = {
        "endpoints_bi_prevalence": [
            {
                "api": e.api,
                "model": e.model,
                "provider": e.provider,
                "cost": list(e.cost),
            }
            for e in endpoints
        ]
    }
    output_path = root / "endpoints_bi_prevalence.yaml"
    with open(output_path, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved {len(endpoints)} endpoints to {output_path}")


async def refresh_endpoints() -> None:
    """Regenerate endpoints_bi_prevalence.yaml from endpoints_bi.yaml, dropping dead (404) ones."""
    endpoints_bi_path = root / "endpoints_bi.yaml"
    with open(endpoints_bi_path) as f:
        data = yaml.safe_load(f)
    all_endpoints = [Endpoint(**e) for e in data["endpoints_bi"]]
    candidates = get_cheap_endpoints(
        all_endpoints, n=PREVALENCE_N, max_avg_cost=PREVALENCE_MAX_AVG_COST
    )
    logger.info(
        f"Refresh: {len(all_endpoints)} total -> {len(candidates)} cheap candidates"
    )

    async with OpenRouterClient(timeout=60.0) as probe_client:
        _strategies, failed = await resolve_strategies(probe_client, candidates)

    kept_transient = {
        k: errors for k, errors in failed.items() if _has_transient_error(errors)
    }
    dead = {k for k in failed if k not in kept_transient}
    live_endpoints = [ep for ep in candidates if str(ep) not in dead]
    logger.info(
        f"Refresh: {len(dead)} dead, {len(kept_transient)} temporarily failing (kept), "
        f"{len(live_endpoints)} live endpoints written"
    )
    for ep_str, errors in kept_transient.items():
        logger.info(f"  kept (transient): {ep_str} — {errors}")
    _save_prevalence_endpoints(live_endpoints)

    # Prune dead entries from strategy cache
    cached = load_strategies()
    pruned = {k: v for k, v in cached.items() if k not in dead}
    if len(pruned) < len(cached):
        save_strategies(pruned)
        logger.info(
            f"Pruned {len(cached) - len(pruned)} dead entries from strategy cache"
        )

    # Strategy breakdown
    hidden_reasoning = sum(
        1 for errors in failed.values() if any("hidden reasoning" in e for e in errors)
    )
    counts = {
        "plain": 0,
        "reasoning disabled": 0,
        "reasoning budget": 0,
        "hidden reasoning": hidden_reasoning,
        "other failure": 0,
    }
    for ep in live_endpoints:
        key = str(ep)
        s = _strategies.get(key)
        if isinstance(s, PlainStrategy):
            counts["plain"] += 1
        elif isinstance(s, ReasoningDisabledStrategy):
            counts["reasoning disabled"] += 1
        elif isinstance(s, ReasoningBudgetStrategy):
            counts["reasoning budget"] += 1
        elif key not in failed:
            counts["other failure"] += 1
    total = len(live_endpoints) + hidden_reasoning
    logger.info(f"Strategy breakdown ({total} valid endpoints):")
    for name, n in counts.items():
        if n:
            logger.info(f"  {name}: {n} ({n / total:.0%})")


def _build_states(
    temperatures: list[float],
    base_dir: Path,
) -> tuple[list[EndpointState], list[Endpoint]]:
    """Build EndpointState objects for all prevalence endpoints, loading existing data from disk."""
    tokenizer_index, fallback_tokens = load_tokenizers()
    endpoints = config.endpoints_bi_prevalence

    requests_per_second = config.bi.phase_1.requests_per_second_per_endpoint
    max_concurrent_requests = config.bi.phase_1.max_concurrent_requests_per_endpoint
    max_concurrent_tokens = config.bi.phase_1.max_concurrent_tokens_per_endpoint

    states = [
        EndpointState(
            endpoint=ep,
            input_tokens=get_input_tokens(
                ep,
                tokenizer_index,
                fallback_tokens,
                config.bi.prevalence.tokens_per_endpoint,
            ),
            temperatures=temperatures,
            base_dir=base_dir,
            rate_limiter=AsyncLimiter(requests_per_second, 1),
            concurrency_semaphore=asyncio.Semaphore(max_concurrent_requests),
            pending_before_new_semaphore=asyncio.Semaphore(max_concurrent_tokens),
            queries_per_token=config.bi.prevalence.queries_per_token,
        )
        for ep in endpoints
    ]
    return states, endpoints


def _apply_cached_strategies(states: list[EndpointState]) -> None:
    """Apply cached strategies to states (for status display without API calls)."""
    from trackllm_website.bi.common import _raw_to_strategy, load_strategies

    cached = load_strategies()
    for state in states:
        key = str(state.endpoint)
        if key in cached:
            raw = cached[key]
            if isinstance(raw, dict) and "skip" in raw:
                continue
            state.query_strategy = _raw_to_strategy(raw)


def _count_dropped_candidates() -> int:
    """Count endpoints that were candidates but dropped during refresh (hidden reasoning, etc.)."""
    endpoints_bi_path = root / "endpoints_bi.yaml"
    if not endpoints_bi_path.exists():
        return 0
    with open(endpoints_bi_path) as f:
        data = yaml.safe_load(f)
    all_endpoints = [Endpoint(**e) for e in data["endpoints_bi"]]
    candidates = get_cheap_endpoints(
        all_endpoints, n=PREVALENCE_N, max_avg_cost=PREVALENCE_MAX_AVG_COST
    )
    prev_set = {str(e) for e in config.endpoints_bi_prevalence}
    return sum(1 for e in candidates if str(e) not in prev_set)


def show_status(
    temperatures: list[float],
    base_dir: Path | None = None,
    extra_failed: int | None = None,
) -> None:
    if base_dir is None:
        base_dir = config.bi.data_dir / "bi_prevalence"
    states, endpoints = _build_states(temperatures, base_dir)
    _apply_cached_strategies(states)
    if extra_failed is None:
        extra_failed = _count_dropped_candidates()
    if extra_failed:
        logger.info(
            f"{extra_failed} additional endpoints failed (hidden reasoning / dead)"
        )
    log_status(states, target_bis=TARGET_BORDER_INPUTS, extra_failed=extra_failed)
    total = len(endpoints) + extra_failed
    endpoints_with_bis = sum(1 for s in states if s.get_border_tokens_count() > 0)
    logger.info(
        f"Endpoints with BIs: {endpoints_with_bis}/{total} valid endpoints "
        f"({endpoints_with_bis / total:.0%})"
    )
    report_cost_to_target(states, TARGET_BORDER_INPUTS)


async def run_bi_prevalence(
    temperatures: list[float],
    base_dir: Path | None = None,
) -> None:
    """Run BI prevalence study across multiple temperatures.

    This is essentially the same as phase_1a, but:
    - Runs across multiple temperatures (shared rate limiters per endpoint)
    - Uses config.endpoints_bi_prevalence instead of config.endpoints_bi
    - Stops early per endpoint once 5 border inputs are found
    - Supports reasoning models via strategy resolution
    """
    if base_dir is None:
        base_dir = config.bi.data_dir / "bi_prevalence"

    logger.info(f"Running BI prevalence with temperatures={temperatures}")
    states, endpoints = _build_states(temperatures, base_dir)

    async with OpenRouterClient(timeout=60.0) as probe_client:
        strategies, _failed = await resolve_strategies(probe_client, endpoints)

    valid_endpoints = [ep for ep in endpoints if str(ep) in strategies]
    skipped = len(endpoints) - len(valid_endpoints)
    if skipped:
        logger.info(
            f"Skipped {skipped} endpoints (no working strategy / hidden reasoning)"
        )
    logger.info(
        f"Running for {len(valid_endpoints)} endpoints across temperatures: {temperatures}"
    )

    # Assign strategies to matching states
    valid_states = []
    for s in states:
        key = str(s.endpoint)
        if key in strategies:
            s.query_strategy = strategies[key]
            valid_states.append(s)

    pending_lists = [s.get_unfinished_prompts() for s in valid_states]

    await run_queries(
        valid_states,
        pending_lists,
        config.bi.phase_1.request_delay_seconds,
        stop_early=_stop_early_5_bis,
        target_bis=TARGET_BORDER_INPUTS,
    )

    # Cost report
    log_status(valid_states, target_bis=TARGET_BORDER_INPUTS)
    endpoints_with_bis = sum(1 for s in valid_states if s.get_border_tokens_count() > 0)
    logger.info(
        f"Endpoints with BIs: {endpoints_with_bis}/{len(endpoints)} valid endpoints "
        f"({endpoints_with_bis / len(endpoints):.0%})"
    )
    logger.info("BI prevalence study complete")


if __name__ == "__main__":
    import fire

    TEMPERATURES = [0.0]  # , 1e-10, 1e-5, 1e-3, 1e-2, 1e-1, 0.5, 1.5]

    def run(
        refresh: bool = False,
        status: bool = False,
        extra_failed: int | None = None,
    ) -> None:
        if refresh:
            asyncio.run(refresh_endpoints())
        elif status:
            show_status(TEMPERATURES, extra_failed=extra_failed)
        else:
            asyncio.run(run_bi_prevalence(TEMPERATURES))

    fire.Fire(run)
