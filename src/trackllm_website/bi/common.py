"""Common utilities for BI (border input) experiments."""

import asyncio
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
from trackllm_website.bi.selection import SelectionPolicy, exceeds_ceiling
from trackllm_website.config import Endpoint, config, logger, root
from trackllm_website.storage import Response
from trackllm_website.util import atomic_write_bytes, slugify

SAVE_INTERVAL = 5

# Marker placed first in a discover_strategy error list when an endpoint is
# short-circuited for cost; callers route these to the too_expensive cache.
TOO_EXPENSIVE = "too_expensive"


# --- Query strategy types for reasoning-aware endpoints ---


@dataclass(frozen=True)
class PlainStrategy:
    pass


@dataclass(frozen=True)
class ReasoningDisabledStrategy:
    pass


@dataclass(frozen=True)
class ReasoningBudgetStrategy:
    budget: int


QueryStrategy = PlainStrategy | ReasoningDisabledStrategy | ReasoningBudgetStrategy


def strategy_to_query_args(strategy: QueryStrategy) -> dict:
    """Convert a QueryStrategy to kwargs for client.query()."""
    if isinstance(strategy, PlainStrategy):
        return {}
    elif isinstance(strategy, ReasoningDisabledStrategy):
        return {"reasoning": {"effort": "none"}}
    elif isinstance(strategy, ReasoningBudgetStrategy):
        return {
            "output_tokens": strategy.budget + 1,
            "reasoning": {"max_tokens": strategy.budget},
        }


def extract_first_token(response: Response) -> str | None:
    """Concatenate reasoning + content, return first whitespace-delimited token."""
    parts = []
    if response.reasoning_content:
        parts.append(response.reasoning_content)
    if response.content:
        parts.append(response.content)
    text = " ".join(parts)
    tokens = text.split()
    return tokens[0] if tokens else None


# --- Strategy discovery and caching ---


def load_strategies() -> dict[str, dict | None]:
    if not (root / config.bi.probe.strategies_path).exists():
        return {}
    with open(root / config.bi.probe.strategies_path, "rb") as f:
        return orjson.loads(f.read())


def save_strategies(strategies: dict[str, dict | None]) -> None:
    atomic_write_bytes(
        root / config.bi.probe.strategies_path,
        orjson.dumps(strategies, option=orjson.OPT_SORT_KEYS),
    )


def _raw_to_strategy(raw: dict | None) -> QueryStrategy:
    """Convert a cached raw strategy dict to a QueryStrategy."""
    if raw is None:
        return PlainStrategy()
    if "effort" in raw:
        return ReasoningDisabledStrategy()
    return ReasoningBudgetStrategy(budget=raw["max_tokens"] * 2)


def _strategy_to_raw(strategy: QueryStrategy) -> dict | None:
    """Convert a QueryStrategy to a raw dict for caching (stores the discovered budget, not doubled)."""
    if isinstance(strategy, PlainStrategy):
        return None
    elif isinstance(strategy, ReasoningDisabledStrategy):
        return {"effort": "none"}
    elif isinstance(strategy, ReasoningBudgetStrategy):
        # Store the original discovered budget (not doubled) for cache compat with test_reasoning.py
        return {"max_tokens": strategy.budget // 2}


async def discover_strategy(
    client: OpenRouterClient,
    endpoint: Endpoint,
    policy: SelectionPolicy | None = None,
) -> tuple[QueryStrategy, None] | tuple[None, list[str]]:
    """Probe an endpoint to find a working query strategy.

    Returns (strategy, None) on success, or (None, errors) on failure.
    For budget strategies, returns budget = discovered_budget * 2 (headroom).

    If `policy` is given, short-circuit as soon as a probe proves the endpoint will
    be too expensive: we monitor at 2x the discovered budget for headroom, so if
    twice a probe's advertised (token-math) price already exceeds the per-request
    ceiling, every real request will too. Bails with errors=[TOO_EXPENSIVE, ...]
    instead of escalating (which, for e.g. image models, means more expensive probes).

    This is a token-math pre-filter on `response.cost` (advertised price x usage),
    not the billed-cost backstop: an endpoint that bills far above its token math
    (a "liar", e.g. per-image billing with trivial token usage) is still caught
    later by vet_endpoint's measured get_generation_cost.
    """
    errors: list[str] = []
    TRANSIENT_CODES = {429, 0}

    def _record_error(r: Response, label: str) -> None:
        if r.error:
            errors.append(f"{label}: {r.error.http_code} {r.error.message}")
        else:
            errors.append(f"{label}: empty response")

    def _too_expensive(r: Response) -> list[str] | None:
        """The 2x-buffered advertised cost of this probe, vs the selection ceiling."""
        if policy is None or r.error:
            return None
        if exceeds_ceiling(2 * r.cost, endpoint.model, endpoint.provider, policy):
            return [
                TOO_EXPENSIVE,
                f"2x ${r.cost:.4f}/req exceeds ${policy.max_endpoint_cost}/mo ceiling",
            ]
        return None

    # Try plain
    r = await client.query(
        endpoint,
        config.bi.probe.prompt,
        logprobs=False,
        max_retries=config.bi.probe.max_retries,
    )
    if (te := _too_expensive(r)) is not None:
        return None, te
    if not r.error and extract_first_token(r):
        return PlainStrategy(), None
    _record_error(r, "plain")
    if r.error and r.error.http_code in TRANSIENT_CODES:
        return None, errors

    # Try effort=none
    r = await client.query(
        endpoint,
        config.bi.probe.prompt,
        logprobs=False,
        reasoning={"effort": "none"},
        max_retries=config.bi.probe.max_retries,
    )
    if (te := _too_expensive(r)) is not None:
        return None, te
    if not r.error and extract_first_token(r):
        return ReasoningDisabledStrategy(), None
    _record_error(r, "effort=none")

    # Try escalating budgets
    budget = 1
    while budget <= config.bi.probe.max_budget:
        r = await client.query(
            endpoint,
            config.bi.probe.prompt,
            logprobs=False,
            output_tokens=budget + 1,
            reasoning={"max_tokens": budget},
            max_retries=config.bi.probe.max_retries,
        )
        if (te := _too_expensive(r)) is not None:
            return None, te
        if not r.error and extract_first_token(r):
            if not r.reasoning_content:
                errors.append(f"budget={budget}: hidden reasoning")
                return None, errors
            return ReasoningBudgetStrategy(budget=budget * 2), None
        _record_error(r, f"budget={budget}")
        budget *= 2

    return None, errors


async def resolve_strategies(
    client: OpenRouterClient,
    endpoints: list[Endpoint],
    policy: SelectionPolicy | None = None,
) -> tuple[dict[str, QueryStrategy], dict[str, list[str]]]:
    """Resolve strategies for all endpoints, using cache where possible.

    Returns (strategies, failed) where:
    - strategies: mapping from endpoint str -> QueryStrategy for working endpoints
    - failed: mapping from endpoint str -> error list for endpoints that failed probing

    When `policy` is given, an endpoint proven too expensive mid-probe is reported in
    `failed` with TOO_EXPENSIVE as the first error (see discover_strategy).
    """
    cached_raw = load_strategies()
    result: dict[str, QueryStrategy] = {}
    failed: dict[str, list[str]] = {}
    to_probe: list[Endpoint] = []

    for ep in endpoints:
        key = str(ep)
        if key in cached_raw:
            raw = cached_raw[key]
            if isinstance(raw, dict) and "skip" in raw:
                failed[key] = [f"cached: {raw['skip']}"]
                logger.info(f"{ep}: skipped ({raw['skip']})")
                continue
            result[key] = _raw_to_strategy(raw)
        else:
            to_probe.append(ep)

    if not to_probe:
        return result, failed

    logger.info(f"Probing {len(to_probe)} endpoints for reasoning strategy...")

    async def _probe_one(
        ep: Endpoint,
    ) -> tuple[str, QueryStrategy | None, list[str] | None]:
        key = str(ep)
        strategy, errors = await discover_strategy(client, ep, policy=policy)
        if strategy is None:
            logger.warning(f"Skipping {ep} — probe errors: {errors}")
            return key, None, errors
        logger.info(f"{ep}: discovered {strategy}")
        return key, strategy, None

    probe_results = await asyncio.gather(*[_probe_one(ep) for ep in to_probe])

    updated = False
    for key, strategy, errors in probe_results:
        if strategy is None:
            failed[key] = errors
            if any("hidden reasoning" in e for e in errors):
                cached_raw[key] = {"skip": "hidden reasoning"}
                updated = True
        else:
            result[key] = strategy
            if key not in cached_raw or _raw_to_strategy(cached_raw[key]) != strategy:
                cached_raw[key] = _strategy_to_raw(strategy)
                updated = True

    if updated:
        save_strategies(cached_raw)
        logger.info(f"Updated strategy cache ({len(cached_raw)} entries)")

    return result, failed

    return result


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


META_KEY = "_meta"


QueryMeta = list[int]  # [input_tokens, output_tokens, reasoning_tokens]


def load_existing_results(
    path: Path,
) -> tuple[dict[int, dict[str, list[str]]], dict[int, dict[str, list[QueryMeta]]]]:
    """Load existing results and metadata from JSON file."""
    if not path.exists():
        return {}, {}
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    meta_raw = data.pop(META_KEY, {})
    results = {int(k): v for k, v in data.items()}
    meta = {int(k): v for k, v in meta_raw.items()}
    return results, meta


async def save_results(
    path: Path,
    results: dict[int, dict[str, list[str]]],
    meta: dict[int, dict[str, list[QueryMeta]]],
) -> None:
    """Save results and metadata to JSON file."""
    async with _get_file_semaphore():
        serializable = {str(k): v for k, v in results.items()}
        if meta:
            serializable[META_KEY] = {str(k): v for k, v in meta.items()}
        atomic_write_bytes(path, orjson.dumps(serializable))


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
    meta: dict[int, dict[str, list[QueryMeta]]] = field(default_factory=dict)
    _prompt_query_counts: dict[str, int] = field(default_factory=dict)
    _prompt_unique_outputs: dict[str, set[str]] = field(default_factory=dict)
    _unsaved_count: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.results, self.meta = load_existing_results(self.output_path)
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        self._prompt_query_counts = {}
        self._prompt_unique_outputs = {}
        for token_results in self.results.values():
            for token, outputs in token_results.items():
                self._prompt_query_counts[token] = len(outputs)
                non_empty = {o for o in outputs if o}
                if non_empty:
                    self._prompt_unique_outputs[token] = non_empty

    async def record_result(
        self,
        token: str,
        content: str,
        num_input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
    ) -> None:
        async with self._lock:
            if num_input_tokens not in self.results:
                self.results[num_input_tokens] = {}
            results = self.results[num_input_tokens]
            if token not in results:
                results[token] = []
            results[token].append(content)

            if num_input_tokens not in self.meta:
                self.meta[num_input_tokens] = {}
            meta = self.meta[num_input_tokens]
            if token not in meta:
                meta[token] = []
            meta[token].append([num_input_tokens, output_tokens, reasoning_tokens])

            self._prompt_query_counts[token] = (
                self._prompt_query_counts.get(token, 0) + 1
            )
            if content:
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
            await save_results(self.output_path, self.results, self.meta)

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
    query_strategy: QueryStrategy = field(default_factory=PlainStrategy)
    extra_output_tokens: int = 0
    content_extractor: Callable[[Response], str] | None = None
    max_retries: int | None = None
    backoff_on_timeout: bool = True
    request_timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    rate_limit_timestamps: deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    abandon_after_timeouts: int | None = None
    completed_queries: int = 0
    total_queries: int = 0
    got_404: bool = False
    timeout_count: int = 0
    unresponsive: bool = False
    recent_costs: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    successful_queries: int = 0
    empty_responses: int = 0
    _temp_results: dict[float, TemperatureResults] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for temp in self.temperatures:
            output_path = get_output_path(self.endpoint, temp, self.base_dir)
            self._temp_results[temp] = TemperatureResults(
                temperature=temp, output_path=output_path
            )
        # Count empty responses from loaded data
        for tr in self._temp_results.values():
            for token_results in tr.results.values():
                for outputs in token_results.values():
                    self.empty_responses += sum(1 for o in outputs if not o)

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

    def get_border_tokens_count(self, max_tokens: int | None = None) -> int:
        """Count tokens that have at least two different outputs (at any temperature)."""
        return len(self.get_border_tokens(max_tokens))

    def get_border_tokens(self, max_tokens: int | None = None) -> list[str]:
        """Get list of border inputs (at any temperature).

        If max_tokens is set, only consider the first max_tokens input tokens.
        """
        tokens = self.input_tokens[:max_tokens] if max_tokens else self.input_tokens
        return list(
            {
                token
                for token in tokens
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
        self,
        temperature: float,
        token: str,
        content: str,
        num_input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
    ) -> None:
        await self._temp_results[temperature].record_result(
            token,
            content,
            num_input_tokens,
            output_tokens,
            reasoning_tokens,
        )

    async def flush(self) -> None:
        for temp_results in self._temp_results.values():
            await temp_results.flush()


def log_status(
    states: list[EndpointState],
    target_bis: int | None = None,
    extra_failed: int = 0,
) -> None:
    """Log status for each endpoint in a dynamic table format."""
    total_estimated_cost = 0.0
    total_spent_cost = 0.0

    def _compute_spent_cost(state: EndpointState) -> float:
        input_cost_per_mtok = state.endpoint.cost[0]
        output_cost_per_mtok = state.endpoint.cost[1]
        # Use meta if available (exact per-query token counts)
        total_in = 0
        total_out = 0
        has_meta = False
        for tr in state._temp_results.values():
            for token_metas in tr.meta.values():
                for metas in token_metas.values():
                    has_meta = True
                    for in_tok, out_tok, reas_tok in metas:
                        total_in += in_tok
                        total_out += out_tok + reas_tok
        # Use in-memory counters if available (live run)
        if not has_meta and state.successful_queries > 0:
            total_in = state.total_input_tokens
            total_out = state.total_output_tokens + state.total_reasoning_tokens
        # Last resort: estimate 1 input + 1 output token per query
        if total_in == 0 and total_out == 0:
            n_queries = sum(
                len(outputs)
                for tr in state._temp_results.values()
                for token_results in tr.results.values()
                for outputs in token_results.values()
            )
            total_in = n_queries
            total_out = n_queries
        return (
            total_in * input_cost_per_mtok + total_out * output_cost_per_mtok
        ) / 1_000_000

    if states:
        max_name_len = max(len(str(s.endpoint)) for s in states)
        col_width = max(max_name_len + 2, 20)
    else:
        col_width = 20

    fmt = f"{{:<{col_width}}} {{:>20}} {{:>18}} {{:>10}} {{:>10}} {{:>12}} {{:>12}} {{:>22}} {{:>8}}"
    headers = [
        "Endpoint",
        "Tokens",
        "Border (BI%)",
        "RPS",
        "429s",
        "Spent",
        "Est. Cost",
        "Avg Tok (in/out/reas)",
        "Empty%",
    ]
    separator_len = col_width + 119

    separator = "-" * separator_len

    logger.info(separator)
    logger.info(fmt.format(*headers))
    logger.info(separator)

    rows: list[tuple[float, list]] = []
    for state in states:
        completed_tokens = state.get_completed_tokens()
        border_tokens = state.get_border_tokens_count()
        total_tokens = len(state.input_tokens) * len(state.temperatures)
        rate_limits = state.get_recent_rate_limits()
        rps = state.get_requests_per_second()

        bi_pct = border_tokens / completed_tokens if completed_tokens else 0

        spent_cost = _compute_spent_cost(state)
        total_spent_cost += spent_cost
        completed_queries = sum(
            len(outputs)
            for tr in state._temp_results.values()
            for token_results in tr.results.values()
            for outputs in token_results.values()
        )
        expected_queries = state.total_queries or total_tokens * state.queries_per_token
        if completed_queries > 0:
            estimated_cost = spent_cost / completed_queries * expected_queries
        else:
            estimated_cost = 0.0
        total_estimated_cost += estimated_cost

        if state.successful_queries > 0:
            avg_in = state.total_input_tokens / state.successful_queries
            avg_out = state.total_output_tokens / state.successful_queries
            avg_reas = state.total_reasoning_tokens / state.successful_queries
            tok_str = f"{avg_in:.0f}/{avg_out:.0f}/{avg_reas:.0f}"
        else:
            tok_str = "-"

        total_non_error = state.successful_queries + state.empty_responses
        empty_str = (
            f"{state.empty_responses / total_non_error:.0%}" if total_non_error else "-"
        )

        rows.append(
            (
                estimated_cost,
                [
                    str(state.endpoint),
                    f"{completed_tokens}/{total_tokens}",
                    f"{border_tokens} ({bi_pct:.1%})",
                    f"{rps:.1f}",
                    rate_limits,
                    f"${spent_cost:.4f}",
                    f"${estimated_cost:.4f}",
                    tok_str,
                    empty_str,
                ],
            )
        )

    for _, cols in sorted(rows):
        logger.info(fmt.format(*cols))

    logger.info(separator)
    logger.info(
        f"Total spent: ${total_spent_cost:.4f}  |  Total estimated: ${total_estimated_cost:.4f}"
    )

    if target_bis is not None and states:
        min_samples = 450
        max_possible = max(len(s.input_tokens) * s.queries_per_token for s in states)
        if max_possible < min_samples:
            # Small experiment: "mature" = fully completed
            mature = [
                s
                for s in states
                if s.get_completed_tokens() >= len(s.input_tokens)
                or s.get_border_tokens_count() >= target_bis
            ]
            mature_label = "completed"
        else:
            mature = [
                s
                for s in states
                if s.get_completed_tokens() * s.queries_per_token >= min_samples
                or s.get_border_tokens_count() >= target_bis
            ]
            mature_label = f"≥{min_samples} samples"
        n_mature = len(mature) + extra_failed
        for k in range(1, target_bis + 1):
            # 1000 tokens, no reasoning
            k1_nr = sum(
                1
                for s in mature
                if s.get_border_tokens_count(max_tokens=1000) >= k
                and not isinstance(s.query_strategy, ReasoningBudgetStrategy)
            )
            # 2000 tokens, no reasoning
            k2_nr = sum(
                1
                for s in mature
                if s.get_border_tokens_count() >= k
                and not isinstance(s.query_strategy, ReasoningBudgetStrategy)
            )
            # 2000 tokens, with reasoning
            k2_r = sum(1 for s in mature if s.get_border_tokens_count() >= k)
            logger.info(
                f"≥{k} BIs ({mature_label}):  "
                f"1k tok: {k1_nr}/{n_mature} ({k1_nr / n_mature:.0%})  "
                f"2k tok: {k2_nr}/{n_mature} ({k2_nr / n_mature:.0%})  "
                f"+reasoning: {k2_r}/{n_mature} ({k2_r / n_mature:.0%})"
            )


def _queries_to_reach_target(state: EndpointState, target_bis: int) -> int | None:
    """Count queries needed to reach target_bis BIs, walking tokens in input order.

    Returns None if the endpoint hasn't reached the target.
    """
    if state.get_border_tokens_count() < target_bis:
        return None
    bis_found = 0
    queries_used = 0
    for tok in state.input_tokens:
        if bis_found >= target_bis:
            break
        for tr in state._temp_results.values():
            for token_results in tr.results.values():
                if tok in token_results:
                    queries_used += len(token_results[tok])
                    non_empty = {o for o in token_results[tok] if o}
                    if len(non_empty) > 1:
                        bis_found += 1
    return queries_used


def report_cost_to_target(states: list[EndpointState], target_bis: int) -> None:
    """Print per-endpoint and total cost to reach target_bis BIs."""
    rows: list[tuple[str, int, float]] = []
    for state in states:
        queries = _queries_to_reach_target(state, target_bis)
        if queries is None:
            continue
        input_cost_per_mtok, output_cost_per_mtok = state.endpoint.cost
        cost = queries * (input_cost_per_mtok + output_cost_per_mtok) / 1_000_000
        rows.append((str(state.endpoint), queries, cost))

    if not rows:
        logger.info(f"No endpoints reached {target_bis} BIs")
        return

    max_name = max(len(name) for name, _, _ in rows)
    col_w = max(max_name + 2, 20)
    fmt = f"{{:<{col_w}}} {{:>10}} {{:>12}}"
    sep = "-" * (col_w + 25)

    logger.info(sep)
    logger.info(fmt.format("Endpoint", "Queries", "Cost"))
    logger.info(sep)
    for name, queries, cost in sorted(rows, key=lambda r: r[2]):
        logger.info(fmt.format(name, queries, f"${cost:.4f}"))
    logger.info(sep)

    total_cost = sum(c for _, _, c in rows)
    total_queries = sum(q for _, q, _ in rows)
    avg_queries = total_queries / len(rows)
    avg_cost = total_cost / len(rows)
    logger.info(
        f"Total: ${total_cost:.4f} across {len(rows)} endpoints "
        f"(avg {avg_queries:.0f} queries, ${avg_cost:.4f}/endpoint)"
    )


async def query_single(
    client: OpenRouterClient,
    state: EndpointState,
    token: str,
    temperature: float,
) -> bool:
    """Execute a single query. Returns False if the endpoint should be abandoned."""
    if state.got_404 or state.unresponsive:
        return False

    await state.rate_limiter.acquire()

    if state.got_404 or state.unresponsive:
        return False

    state.request_timestamps.append(time.monotonic())

    def on_retry(status: int) -> None:
        if status == 429:
            state.rate_limit_timestamps.append(time.monotonic())

    query_kwargs = strategy_to_query_args(state.query_strategy)
    if state.extra_output_tokens:
        query_kwargs["output_tokens"] = (
            query_kwargs.get("output_tokens", 1) + state.extra_output_tokens
        )

    response = await client.query(
        state.endpoint,
        token,
        temperature=temperature,
        logprobs=False,
        on_retry=on_retry,
        max_retries=state.max_retries,
        backoff_on_timeout=state.backoff_on_timeout,
        **query_kwargs,
    )
    state.completed_queries += 1
    if response.error:
        if response.error.http_code == 404:
            if not state.got_404:
                state.got_404 = True
                logger.warning(f"Got 404 for {state.endpoint}, abandoning endpoint")
            return False
        if response.error.http_code == 0 and response.error.message.startswith(
            "Timeout"
        ):
            state.timeout_count += 1
            # Abandon a "produced nothing" endpoint: enough timeouts and zero
            # successes. Gating on successful_queries==0 (rather than requiring
            # EVERY request to be a timeout) means a stray non-timeout error can't
            # keep a dead endpoint querying for the whole run.
            if (
                state.abandon_after_timeouts is not None
                and state.timeout_count >= state.abandon_after_timeouts
                and state.successful_queries == 0
            ):
                state.unresponsive = True
                logger.warning(
                    f"{state.endpoint} unresponsive ({state.timeout_count} timeouts, 0 successes), abandoning for this run"
                )
                return False
        logger.warning(
            f"Error for {state.endpoint}: {token!r}: {response.error.message}"
        )
        return True
    state.recent_costs.append(response.cost)
    state.total_input_tokens += response.input_tokens
    state.total_output_tokens += response.output_tokens
    state.total_reasoning_tokens += response.reasoning_tokens
    content = (
        state.content_extractor(response)
        if state.content_extractor
        else (extract_first_token(response) or "")
    )
    if not content:
        state.empty_responses += 1
    state.successful_queries += 1
    await state.record_result(
        temperature,
        token,
        content,
        response.input_tokens,
        response.output_tokens,
        response.reasoning_tokens,
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
    full_pending = len(state.temperatures) * state.queries_per_token
    is_resuming_partial = total_pending < full_pending
    if not is_resuming_partial and stop_early and stop_early(state):
        pbar.update(total_pending)
        return

    # Copy so we can mutate
    temp_pending = dict(temp_pending)
    max_rounds = max(temp_pending.values()) if temp_pending else 0

    async with state.pending_before_new_semaphore:
        for i in range(max_rounds):
            if state.got_404 or state.unresponsive:
                return
            for temp in state.temperatures:
                if state.got_404 or state.unresponsive:
                    return
                if temp_pending[temp] <= 0:
                    continue
                async with state.concurrency_semaphore:
                    success = await query_single(client, state, token, temp)
                temp_pending[temp] -= 1
                pbar.update(1)
                if not success:
                    return
            if i < max_rounds - 1:
                await asyncio.sleep(request_delay_seconds)


async def run_queries(
    states: list[EndpointState],
    pending_lists: list[list[tuple[str, dict[float, int]]]],
    request_delay_seconds: float,
    stop_early: Callable[[EndpointState], bool] | None = None,
    target_bis: int | None = None,
) -> None:
    """Run queries for all states with interleaved scheduling."""
    # Filter out tokens that would be immediately skipped by stop_early,
    # so they don't inflate the tqdm total/rate on resume.
    actual_pending_lists = []
    for state, pending in zip(states, pending_lists):
        if stop_early and stop_early(state):
            # Only keep partial tokens (must be completed for clean stats)
            full = len(state.temperatures) * state.queries_per_token
            kept = [(tok, tp) for tok, tp in pending if sum(tp.values()) < full]
        else:
            kept = pending
        actual_pending_lists.append(kept)
        state.total_queries = sum(sum(tp.values()) for _, tp in kept)

    client = OpenRouterClient()
    total_requests = sum(s.total_queries for s in states)
    logger.info(f"Total requests to process: {total_requests}")

    try:

        async def run_with_status(pbar: tqdm) -> None:
            coros = []
            max_tokens = (
                max(len(p) for p in actual_pending_lists) if actual_pending_lists else 0
            )
            for i in range(max_tokens):
                for state, pending in zip(states, actual_pending_lists):
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
                    log_status(states, target_bis=target_bis)
                    last_status_time = now

        with tqdm(total=total_requests, desc="Requests") as pbar:
            await run_with_status(pbar)
    finally:
        for state in states:
            await state.flush()
        await client.close()

    log_status(states, target_bis=target_bis)

    for state in states:
        incomplete = len(state.input_tokens) - state.get_completed_tokens()
        if incomplete > 0:
            logger.info(f"{state.endpoint}: {incomplete} incomplete tokens")
