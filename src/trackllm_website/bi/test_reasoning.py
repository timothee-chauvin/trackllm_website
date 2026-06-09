"""Test which endpoints return content with limited output tokens.

Reasoning models need extra tokens for chain-of-thought before producing visible output,
so requesting only 1 output token may result in empty content.
"""

import asyncio
import logging
from datetime import datetime, timezone

import orjson
import yaml
from tqdm import tqdm

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.generate_bi_prevalence_endpoints import (
    main as regenerate_endpoints,
)
from trackllm_website.config import Endpoint, config, logger, root
from trackllm_website.storage import Response, ResponseError

PROMPT = "a"
OUTPUT_TOKENS = 1
REASONING_BUDGET = 1024
TIMEOUT = 60.0
BAD_ENDPOINTS_PATH = root / "bad_endpoints_test_reasoning.json"
STRATEGIES_PATH = root / config.bi.probe.strategies_path
HISTORY_PATH = root / "history_test_reasoning.json"
SKIP_HTTP_CODES = {0, 404, 429}


def _load_json(path) -> dict | list | set:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def _save_json(path, data) -> None:
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_SORT_KEYS))


def load_bad_endpoints() -> set[str]:
    data = _load_json(BAD_ENDPOINTS_PATH)
    return set(data) if data else set()


def save_bad_endpoints(bad: set[str]) -> None:
    _save_json(BAD_ENDPOINTS_PATH, sorted(bad))


def load_known_strategies() -> dict[str, dict]:
    """Load previously discovered strategies. Maps endpoint str -> reasoning arg dict."""
    return _load_json(STRATEGIES_PATH) or {}


def save_known_strategies(strategies: dict[str, dict]) -> None:
    _save_json(STRATEGIES_PATH, strategies)


def load_prevalence_endpoints() -> list[Endpoint]:
    regenerate_endpoints()
    with open(root / "endpoints_bi_prevalence.yaml") as f:
        data = yaml.safe_load(f)
    all_eps = [Endpoint(**e) for e in data["endpoints_bi_prevalence"]]
    bad = load_bad_endpoints()
    filtered = [e for e in all_eps if str(e) not in bad]
    if skipped := len(all_eps) - len(filtered):
        logger.info(f"Skipped {skipped} known-bad endpoints")
    return filtered


async def run(
    output_tokens: int = OUTPUT_TOKENS,
    reset: bool = False,
    freeze_blacklist: bool = True,
) -> None:
    """Test which endpoints return content with limited output tokens.

    For each endpoint, tries three strategies in order:
    1. reasoning.max_tokens=1024 + output_tokens (cheap, works for reasoning models)
    2. Plain output_tokens, no reasoning param (works for non-reasoning models)
    3. reasoning.effort="none" to disable reasoning (last resort)

    Args:
        output_tokens: Number of output tokens to request (on top of reasoning budget).
        reset: Clear the blacklist before running (new failures will still be added).
        freeze_blacklist: Keep existing blacklist but don't add new entries.
    """
    logger.setLevel(logging.DEBUG)
    if reset:
        for p in [BAD_ENDPOINTS_PATH, STRATEGIES_PATH, HISTORY_PATH]:
            if p.exists():
                p.unlink()
        logger.info("Reset blacklist, strategies, and history")
    endpoints = load_prevalence_endpoints()
    logger.info(
        f"Testing {len(endpoints)} endpoints with output_tokens={output_tokens}"
    )

    results["output_tokens"] = output_tokens
    results["total"] = len(endpoints)
    pbar = tqdm(total=len(endpoints), desc="Querying")
    new_bad: set[str] = set()

    async def _try_request(client: OpenRouterClient, ep, output_tok, reasoning_arg):
        try:
            return await client._make_request(
                ep,
                PROMPT,
                logprobs=False,
                output_tokens=output_tok,
                reasoning=reasoning_arg,
            )
        except Exception as e:
            import aiohttp

            if isinstance(e, aiohttp.ClientResponseError):
                return e.status, str(e)
            elif isinstance(e, asyncio.TimeoutError):
                return 0, f"Timeout after {TIMEOUT}s"
            else:
                return 0, repr(e)

    def _make_error_response(ep, http_code, message):
        return Response(
            date=datetime.now(tz=timezone.utc),
            endpoint=ep,
            prompt=PROMPT,
            logprobs=None,
            cost=0.0,
            error=ResponseError(http_code=http_code, message=message),
        )

    def _strategy_label(reasoning_arg) -> str:
        if reasoning_arg is None:
            return "plain"
        if "effort" in reasoning_arg:
            return f"effort={reasoning_arg['effort']}"
        return f"budget={reasoning_arg['max_tokens']}"

    def _make_strategies(
        output_tokens: int,
        start_budget: int = 1,
    ) -> list[tuple[int, dict | None]]:
        strategies: list[tuple[int, dict | None]] = [
            (output_tokens, None),
            (output_tokens, {"effort": "none"}),
        ]
        budget = start_budget
        while budget <= 2048:
            strategies.append((budget + output_tokens, {"max_tokens": budget}))
            budget *= 2
        return strategies

    known_strategies = load_known_strategies()
    new_strategies: dict[str, dict | None] = {}
    # History: endpoint -> {"output_tokens": [...], "first_token": [...], "reasoning": []}
    history: dict[str, dict[str, list]] = _load_json(HISTORY_PATH) or {}

    async def _search_strategies(client, ep, strategies):
        r = None
        winning_strategy = "none"
        for output_tok, reasoning_arg in strategies:
            logger.info(f"{ep}: trying {_strategy_label(reasoning_arg)}")
            r = await _try_request(client, ep, output_tok, reasoning_arg)
            if isinstance(r, Response) and not r.error and r.content:
                winning_strategy = _strategy_label(reasoning_arg)
                new_strategies[str(ep)] = reasoning_arg
                break
            if isinstance(r, tuple) and r[0] in SKIP_HTTP_CODES:
                break
        if isinstance(r, tuple):
            r = _make_error_response(ep, *r)
        return r, winning_strategy

    async def query_one(client: OpenRouterClient, ep) -> None:
        ep_key = str(ep)
        known = known_strategies.get(ep_key)
        if known is not None and isinstance(known, dict) and "max_tokens" in known:
            # Start from the known budget, escalate if it fails
            start = known["max_tokens"]
            r, winning_strategy = await _search_strategies(
                client,
                ep,
                _make_strategies(output_tokens, start_budget=start),
            )
            if winning_strategy != "none":
                winning_strategy += (
                    " (cached)" if known == new_strategies.get(ep_key) else ""
                )
        elif known is not None:
            # plain or effort=none: just use it directly
            r = await _try_request(client, ep, output_tokens, known)
            winning_strategy = _strategy_label(known) + " (cached)"
            if isinstance(r, tuple):
                r = _make_error_response(ep, *r)
        else:
            r, winning_strategy = await _search_strategies(
                client,
                ep,
                _make_strategies(output_tokens),
            )

        entry = {"response": r, "strategy": winning_strategy}
        if r.error and r.error.http_code in SKIP_HTTP_CODES:
            new_bad.add(ep_key)
            results["skipped"] += 1
        elif r.error:
            results["errored"].append(entry)
        elif not r.content:
            results["empty"].append(entry)
        else:
            results["ok"].append(entry)
            h = history.setdefault(
                ep_key,
                {
                    "output_tokens": [],
                    "first_token": [],
                    "reasoning": [],
                },
            )
            h["output_tokens"].append(r.output_tokens)
            h["first_token"].append(r.content.split()[0] if r.content else "")
            h["reasoning"].append(r.reasoning_content or "")
        pbar.update()

    async with OpenRouterClient(timeout=TIMEOUT) as client:
        await asyncio.gather(*[query_one(client, ep) for ep in endpoints])
    pbar.close()

    if new_bad and not freeze_blacklist:
        all_bad = load_bad_endpoints() | new_bad
        save_bad_endpoints(all_bad)
        logger.info(f"Added {len(new_bad)} newly-bad endpoints ({len(all_bad)} total)")

    if new_strategies:
        all_strategies = {**known_strategies, **new_strategies}
        save_known_strategies(all_strategies)
        logger.info(
            f"Saved {len(new_strategies)} new strategies ({len(all_strategies)} total)"
        )

    _save_json(HISTORY_PATH, history)
    results["history"] = history


results: dict = {
    "ok": [],
    "empty": [],
    "errored": [],
    "skipped": 0,
    "output_tokens": 0,
    "total": 0,
}


def print_results() -> None:
    ok, empty, errored = results["ok"], results["empty"], results["errored"]
    completed = len(ok) + len(empty) + len(errored)
    skipped = results["skipped"]
    print(f"\n{'=' * 60}")
    print(
        f"output_tokens={results['output_tokens']}  |  {completed}/{results['total']} endpoints"
        f"  ({skipped} skipped as 404/429)"
    )
    print(f"  OK (got content):  {len(ok)}")
    print(f"  Empty content:     {len(empty)}")
    print(f"  Errored:           {len(errored)}")
    print(f"{'=' * 60}")

    history = results.get("history", {})

    if ok:
        print("\nOK:")
        for e in sorted(ok, key=lambda e: str(e["response"].endpoint)):
            r = e["response"]
            parts = [
                f"  {r.endpoint}  strategy={e['strategy']}  output_tokens={r.output_tokens}"
            ]
            if r.reasoning_tokens:
                parts.append(f"reasoning_tokens={r.reasoning_tokens}")
            if r.reasoning_content:
                preview = r.reasoning_content[:80].replace("\n", " ")
                parts.append(f"reasoning={preview!r}")
            print("  ".join(parts))

    if empty:
        print("\nEmpty content:")
        for e in sorted(empty, key=lambda e: str(e["response"].endpoint)):
            print(f"  {e['response'].endpoint}  strategy={e['strategy']}")

    if errored:
        print("\nErrored:")
        for e in sorted(errored, key=lambda e: str(e["response"].endpoint)):
            r = e["response"]
            print(f"  {r.endpoint}: {r.error.message}")

    def _changed(vals: list) -> bool:
        return len(vals) > 1 and len(set(vals)) > 1

    # Report changes across runs
    changes = []
    for ep_key, h in sorted(history.items()):
        tokens = h.get("output_tokens", [])
        first_tokens = h.get("first_token", [])
        reasonings = h.get("reasoning", [])
        tok_changed = _changed(tokens)
        ft_changed = _changed(first_tokens)
        r_changed = _changed(reasonings)
        if tok_changed or ft_changed or r_changed:
            lines = [f"  {ep_key}"]
            if ft_changed:
                lines.append(f"    first_token: {first_tokens}")
            if tok_changed:
                lines.append(f"    output_tokens: {tokens}")
            if r_changed:
                previews = [r[:60].replace("\n", " ") for r in reasonings]
                lines.append(f"    reasoning: {previews}")
            changes.append("\n".join(lines))

    # Endpoints that used a reasoning budget but didn't return reasoning content
    hidden_reasoning = [
        e
        for e in ok
        if "budget=" in e["strategy"] and not e["response"].reasoning_content
    ]
    if hidden_reasoning:
        print(
            f"\nHidden reasoning ({len(hidden_reasoning)} endpoints — used budget but no reasoning content):"
        )
        for e in sorted(hidden_reasoning, key=lambda e: str(e["response"].endpoint)):
            r = e["response"]
            print(
                f"  {r.endpoint}  strategy={e['strategy']}  reasoning_tokens={r.reasoning_tokens}"
            )

    if changes:
        print(f"\n{'=' * 60}")
        print(f"CHANGES ACROSS RUNS ({len(changes)} endpoints):")
        print(f"{'=' * 60}")
        for line in changes:
            print(line)


if __name__ == "__main__":
    import fire

    try:
        fire.Fire(run)
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    finally:
        print_results()
