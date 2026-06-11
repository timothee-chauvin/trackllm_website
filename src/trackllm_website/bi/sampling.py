"""Reusable BI sampler: query an endpoint's prompts n times each at T=0."""

import asyncio
from datetime import datetime, timezone

from aiolimiter import AsyncLimiter

from trackllm_website.bi.common import (
    QueryStrategy,
    extract_first_token,
    strategy_to_query_args,
)
from trackllm_website.config import Endpoint, config, logger


async def sample_prompts(
    client,
    endpoint: Endpoint,
    strategy: QueryStrategy,
    prompts: list[str],
    n_per_prompt: int,
) -> tuple[dict[str, list[tuple[str, str]]], int]:
    """Returns ({prompt: [(timestamp, token), ...]}, n_errors).

    Respects the phase 2 rate limits from config. Empty responses count as
    successes with no sample (consistent with phase_2.py).
    """
    cfg = config.bi.phase_2
    limiter = AsyncLimiter(cfg.requests_per_second_per_endpoint, 1)
    semaphore = asyncio.Semaphore(cfg.max_concurrent_requests_per_endpoint)
    samples: dict[str, list[tuple[str, str]]] = {p: [] for p in prompts}
    n_errors = 0

    async def one(prompt: str) -> None:
        nonlocal n_errors
        for i in range(n_per_prompt):
            async with semaphore:
                await limiter.acquire()
                ts = (
                    datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
                )
                response = await client.query(
                    endpoint,
                    prompt,
                    temperature=0.0,
                    logprobs=False,
                    **strategy_to_query_args(strategy),
                )
            if response.error:
                logger.warning(
                    f"Error for {endpoint}: {prompt!r}: {response.error.message}"
                )
                n_errors += 1
                continue
            tok = extract_first_token(response)
            if tok:
                samples[prompt].append((ts, tok))
            if i < n_per_prompt - 1:
                await asyncio.sleep(cfg.request_delay_seconds)

    await asyncio.gather(*(one(p) for p in prompts))
    return samples, n_errors
