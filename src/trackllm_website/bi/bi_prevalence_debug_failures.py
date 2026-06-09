"""Debug zero-BI endpoints: analyze why they have no border inputs, and optionally
probe them with more output tokens to check if BIs appear beyond the first token."""

import asyncio
from collections import Counter
from pathlib import Path

from aiolimiter import AsyncLimiter

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.bi_prevalence import TARGET_BORDER_INPUTS, _build_states
from trackllm_website.bi.common import (
    EndpointState,
    QueryStrategy,
    ReasoningBudgetStrategy,
    _raw_to_strategy,
    load_strategies,
    log_status,
    resolve_strategies,
    run_queries,
)
from trackllm_website.config import config, logger
from trackllm_website.storage import Response

SOURCE_TEMPERATURES = [
    0.0
]  # temperature of the main experiment (to find zero-BI endpoints)
PROBE_TEMPERATURES = [0.0]  # temperature to use when probing
MIN_SAMPLES = 450
COLLAPSED_THRESHOLD = 0.8
PROBE_PROMPTS = 200
MAX_DISPLAY_LEN = 120


def _full_content_extractor(response: Response) -> str:
    """Return full reasoning + content as a single string."""
    parts = []
    if response.reasoning_content:
        parts.append(response.reasoning_content)
    if response.content:
        parts.append(response.content)
    return " ".join(parts)


# --- Zero-BI analysis (no API calls) ---


def _flatten_outputs(
    results: dict[int, dict[str, list[str]]],
) -> dict[str, list[str]]:
    """Merge all input-token groups into {token: [outputs]}."""
    merged: dict[str, list[str]] = {}
    for group in results.values():
        for token, outputs in group.items():
            merged.setdefault(token, []).extend(outputs)
    return merged


def _get_zero_bi_endpoints(base_dir: Path) -> list[dict]:
    """Return info dicts for endpoints with 0 BIs and >= MIN_TOKENS tokens."""
    states, _endpoints = _build_states(SOURCE_TEMPERATURES, base_dir)
    zero_bi = []
    for state in states:
        if state.get_border_tokens_count() > 0:
            continue
        if state.get_completed_tokens() * state.queries_per_token < MIN_SAMPLES:
            continue
        for tr in state._temp_results.values():
            token_outputs = _flatten_outputs(tr.results)
            all_outputs = [o for outputs in token_outputs.values() for o in outputs]
            total_queries = len(all_outputs)
            output_counts = Counter(o for o in all_outputs if o)
            top_output, top_count = (
                output_counts.most_common(1)[0] if output_counts else ("", 0)
            )
            top_frac = top_count / total_queries if total_queries else 0
            zero_bi.append(
                {
                    "endpoint": state.endpoint,
                    "input_tokens": state.input_tokens,
                    "tokens": len(token_outputs),
                    "queries": total_queries,
                    "unique": len(output_counts),
                    "top_output": top_output,
                    "top_frac": top_frac,
                    "top5": output_counts.most_common(5),
                }
            )
    return zero_bi


def show_zero_bi(base_dir: Path) -> None:
    """Print tables of zero-BI endpoints."""
    zero_bi = _get_zero_bi_endpoints(base_dir)
    collapsed = [r for r in zero_bi if r["top_frac"] >= COLLAPSED_THRESHOLD]
    deterministic = [r for r in zero_bi if r["top_frac"] < COLLAPSED_THRESHOLD]

    logger.info(
        f"=== Collapsed endpoints ({len(collapsed)}) — dominated by a single output ==="
    )
    logger.info(f"{'Endpoint':<75} {'Top output':<15} {'Fraction':>10} {'Tokens':>8}")
    logger.info("-" * 115)
    for r in sorted(collapsed, key=lambda r: -r["top_frac"]):
        logger.info(
            f"{str(r['endpoint']):<75} {repr(r['top_output']):<15} "
            f"{r['top_frac']:>9.1%} {r['tokens']:>8}"
        )

    logger.info("")
    logger.info(
        f"=== Deterministic endpoints ({len(deterministic)}) — varied outputs, no per-token variation ==="
    )
    logger.info(f"{'Endpoint':<75} {'Unique':>8} {'Tokens':>8}   Top outputs")
    logger.info("-" * 115)
    for r in sorted(deterministic, key=lambda r: -r["unique"]):
        top5_str = ", ".join(f"{repr(o)}: {n}" for o, n in r["top5"])
        logger.info(
            f"{str(r['endpoint']):<75} {r['unique']:>8} {r['tokens']:>8}   {top5_str}"
        )


# --- Probing with extended output tokens ---


def _stop_early_5_bis(state: EndpointState) -> bool:
    return state.get_border_tokens_count() >= TARGET_BORDER_INPUTS


def _build_probe_states(
    zero_bi: list[dict],
    strategies: dict[str, QueryStrategy],
    output_tokens: int,
    probe_dir: Path,
) -> list[EndpointState]:
    """Build EndpointState objects for probing, using same config as the main experiment."""
    requests_per_second = config.bi.phase_1.requests_per_second_per_endpoint
    max_concurrent_requests = config.bi.phase_1.max_concurrent_requests_per_endpoint
    max_concurrent_tokens = config.bi.phase_1.max_concurrent_tokens_per_endpoint
    queries_per_token = config.bi.prevalence.queries_per_token

    states = []
    for info in zero_bi:
        ep = info["endpoint"]
        key = str(ep)
        if key not in strategies:
            continue
        states.append(
            EndpointState(
                endpoint=ep,
                input_tokens=info["input_tokens"][:PROBE_PROMPTS],
                temperatures=PROBE_TEMPERATURES,
                base_dir=probe_dir,
                rate_limiter=AsyncLimiter(requests_per_second, 1),
                concurrency_semaphore=asyncio.Semaphore(max_concurrent_requests),
                pending_before_new_semaphore=asyncio.Semaphore(max_concurrent_tokens),
                queries_per_token=queries_per_token,
                query_strategy=strategies[key],
                extra_output_tokens=output_tokens - 1,
                content_extractor=_full_content_extractor,
            )
        )
    return states


def _get_bis(state: EndpointState, n_words: int | None) -> list[tuple[str, list[str]]]:
    """Return (token, unique_outputs) for tokens with >1 unique output."""
    all_outputs: dict[str, list[str]] = {}
    for tr in state._temp_results.values():
        for group in tr.results.values():
            for tok, outputs in group.items():
                all_outputs.setdefault(tok, []).extend(outputs)
    bis = []
    for tok, outputs in all_outputs.items():
        if n_words is not None:
            processed = [" ".join(o.split()[:n_words]) for o in outputs if o]
        else:
            processed = [o for o in outputs if o]
        unique = sorted(set(processed))
        if len(unique) > 1:
            bis.append((tok, unique))
    return bis


def _count_bis(state: EndpointState, n_words: int | None) -> int:
    return len(_get_bis(state, n_words))


def _truncate(s: str, max_len: int = MAX_DISPLAY_LEN) -> str:
    return s[:max_len] + ("..." if len(s) > max_len else "")


def _report_probe_results(states: list[EndpointState], output_tokens: int) -> None:
    """Report two-level BI analysis and show BI contents."""
    reasoning_states = [
        s for s in states if isinstance(s.query_strategy, ReasoningBudgetStrategy)
    ]
    non_reasoning_states = [s for s in states if s not in reasoning_states]

    # Show BI contents
    for state in states:
        is_reasoning = isinstance(state.query_strategy, ReasoningBudgetStrategy)
        if is_reasoning:
            output_bis = _get_bis(state, output_tokens)
            reasoning_bis = _get_bis(state, None)
        else:
            output_bis = _get_bis(state, None)
            reasoning_bis = []

        if not output_bis and not reasoning_bis:
            continue
        logger.info("")
        logger.info(f"  {state.endpoint}:")
        for tok, variants in output_bis:
            label = "output BI" if is_reasoning else "BI"
            logger.info(f"    {label} for {tok!r}:")
            for v in variants:
                logger.info(f"      {_truncate(v)!r}")
        output_bi_tokens = {t for t, _ in output_bis}
        for tok, variants in reasoning_bis:
            if tok in output_bi_tokens:
                continue
            logger.info(f"    reasoning-only BI for {tok!r}:")
            for v in variants:
                logger.info(f"      {_truncate(v)!r}")

    logger.info("")
    if non_reasoning_states:
        with_bi = sum(1 for s in non_reasoning_states if _count_bis(s, None) > 0)
        logger.info(
            f"Non-reasoning: {with_bi}/{len(non_reasoning_states)} "
            f"({with_bi / len(non_reasoning_states):.0%}) have ≥1 BI"
        )

    if reasoning_states:
        with_output_bi = sum(
            1 for s in reasoning_states if _count_bis(s, output_tokens) > 0
        )
        with_reasoning_bi = sum(1 for s in reasoning_states if _count_bis(s, None) > 0)
        logger.info(
            f"Reasoning output: {with_output_bi}/{len(reasoning_states)} "
            f"({with_output_bi / len(reasoning_states):.0%}) have ≥1 output BI "
            f"(first {output_tokens} words)"
        )
        logger.info(
            f"Reasoning full: {with_reasoning_bi}/{len(reasoning_states)} "
            f"({with_reasoning_bi / len(reasoning_states):.0%}) have ≥1 reasoning+output BI"
        )


async def run_probe(output_tokens: int, base_dir: Path) -> None:
    """Probe zero-BI endpoints with extended output tokens."""
    zero_bi = _get_zero_bi_endpoints(base_dir)
    if not zero_bi:
        logger.info("No zero-BI endpoints found.")
        return

    endpoints = [r["endpoint"] for r in zero_bi]
    logger.info(
        f"Probing {len(endpoints)} zero-BI endpoints with {output_tokens} output tokens"
    )

    async with OpenRouterClient(timeout=60.0) as client:
        strategies, failed = await resolve_strategies(client, endpoints)

    for key, errors in failed.items():
        logger.info(f"Skipping {key} (probe failed: {errors})")

    probe_dir = config.bi.data_dir / "bi_prevalence_probe" / f"N={output_tokens}"
    states = _build_probe_states(zero_bi, strategies, output_tokens, probe_dir)

    logger.info(f"Running probes for {len(states)} endpoints")
    pending_lists = [s.get_unfinished_prompts() for s in states]

    await run_queries(
        states,
        pending_lists,
        config.bi.phase_1.request_delay_seconds,
        stop_early=_stop_early_5_bis,
        target_bis=TARGET_BORDER_INPUTS,
    )

    log_status(states, target_bis=TARGET_BORDER_INPUTS)
    _report_probe_results(states, output_tokens)


def show_probe_status(output_tokens: int, base_dir: Path) -> None:
    """Show status and BI contents for a previous probe run, no API calls."""
    zero_bi = _get_zero_bi_endpoints(base_dir)
    probe_dir = config.bi.data_dir / "bi_prevalence_probe" / f"N={output_tokens}"
    cached = load_strategies()

    states = _build_probe_states(
        zero_bi,
        {
            str(info["endpoint"]): _raw_to_strategy(cached.get(str(info["endpoint"])))
            for info in zero_bi
            if str(info["endpoint"]) in cached
            and not (
                isinstance(cached.get(str(info["endpoint"])), dict)
                and "skip" in cached[str(info["endpoint"])]
            )
        },
        output_tokens,
        probe_dir,
    )
    # Filter to only states that have data on disk
    states = [s for s in states if any(tr.results for tr in s._temp_results.values())]

    if not states:
        logger.info(f"No probe data found in {probe_dir}")
        return

    log_status(states, target_bis=TARGET_BORDER_INPUTS)
    _report_probe_results(states, output_tokens)


if __name__ == "__main__":
    import fire

    def main(
        probe: int | None = None, status: bool = False, analyze: bool = False
    ) -> None:
        base_dir = config.bi.data_dir / "bi_prevalence"
        if analyze:
            show_zero_bi(base_dir)
        elif status:
            if probe is None:
                logger.error("--status requires --probe N")
                return
            show_probe_status(probe, base_dir)
        elif probe is not None:
            asyncio.run(run_probe(probe, base_dir))
        else:
            show_zero_bi(base_dir)

    fire.Fire(main)
