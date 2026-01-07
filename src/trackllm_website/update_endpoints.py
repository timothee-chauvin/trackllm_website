import asyncio
from collections.abc import Iterable
from decimal import Decimal
from typing import Literal

import aiohttp
import requests
import yaml
from pydantic import BaseModel

from trackllm_website.api import OpenRouterClient
from trackllm_website.config import Endpoint, config, logger, root
from trackllm_website.storage import ResultsStorage
from trackllm_website.util import (
    gather_with_concurrency,
    gather_with_concurrency_streaming,
)

BAD_ENDPOINTS_BI_PATH = root / "bad_endpoints_bi.yaml"


class BadEndpointReason(BaseModel):
    token_usage: list[int] | None = None


class BadEndpoint(BaseModel):
    api: Literal["openrouter"]
    model: str
    provider: str | None = None
    reason: BadEndpointReason

    @classmethod
    def from_endpoint(
        cls, endpoint: Endpoint, reason: BadEndpointReason
    ) -> "BadEndpoint":
        return cls(
            api=endpoint.api,
            model=endpoint.model,
            provider=endpoint.provider,
            reason=reason,
        )

    def matches(self, endpoint: Endpoint) -> bool:
        return (
            self.api == endpoint.api
            and self.model == endpoint.model
            and self.provider == endpoint.provider
        )


def load_bad_endpoints_bi() -> list[BadEndpoint]:
    """Load the set of known bad BI endpoints from disk."""
    if not BAD_ENDPOINTS_BI_PATH.exists():
        return []
    with open(BAD_ENDPOINTS_BI_PATH) as f:
        data = yaml.safe_load(f) or {}
    return [
        BadEndpoint(
            api=e["api"],
            model=e["model"],
            provider=e.get("provider"),
            reason=BadEndpointReason(**e.get("reason", {})),
        )
        for e in data.get("bad_endpoints_bi", [])
    ]


def save_bad_endpoints_bi(bad_endpoints: Iterable[BadEndpoint]) -> None:
    """Save the list of bad BI endpoints to disk."""
    sorted_bad = sorted(
        bad_endpoints,
        key=lambda b: (b.api, b.model, b.provider or ""),
    )
    output_data = {
        "bad_endpoints_bi": [
            {
                "api": b.api,
                "model": b.model,
                "provider": b.provider,
                "reason": b.reason.model_dump(exclude_none=True),
            }
            for b in sorted_bad
        ]
    }
    with open(BAD_ENDPOINTS_BI_PATH, "w") as f:
        # default_flow_style=False for consistent dumping
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)


async def get_model_endpoints(
    session, model_id, logprob_filter: bool = False
) -> list[Endpoint]:
    """Fetch endpoints for a model.

    Args:
        session: aiohttp session
        model_id: OpenRouter model ID
        logprob_filter: If True, only return endpoints that claim to support logprobs
    """
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    try:
        async with session.get(url) as response:
            data = await response.json()
            model_endpoints = data["data"]["endpoints"]
            filtered_endpoints = []
            for endpoint in model_endpoints:
                if logprob_filter and not (
                    "logprobs" in endpoint["supported_parameters"]
                    and "top_logprobs" in endpoint["supported_parameters"]
                ):
                    continue
                endpoint_data = Endpoint(
                    api="openrouter",
                    model=model_id,
                    provider=endpoint["tag"],
                    cost=(
                        float(
                            (
                                Decimal(endpoint["pricing"]["prompt"]) * 1_000_000
                            ).normalize()
                        ),
                        float(
                            (
                                Decimal(endpoint["pricing"]["completion"]) * 1_000_000
                            ).normalize()
                        ),
                    ),
                )
                filtered_endpoints.append(endpoint_data)
            return filtered_endpoints
    except Exception as e:
        logger.error(f"Error fetching endpoints for {model_id}: {e}")
        return []


async def get_endpoints(logprob_filter: bool = False) -> list[Endpoint]:
    """Get all endpoints for all models.

    Args:
        logprob_filter: If True, only return endpoints that claim to support logprobs
    """
    response = requests.get("https://openrouter.ai/api/v1/models")
    model_ids = [model["id"] for model in response.json()["data"]]
    all_endpoints = []
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_model_endpoints(session, model_id, logprob_filter=logprob_filter)
            for model_id in model_ids
        ]
        async for result in gather_with_concurrency_streaming(
            config.api.max_workers, *tasks
        ):
            all_endpoints.extend(result)

    filtered_endpoints = [
        e for e in all_endpoints if e.cost[0] + e.cost[1] < config.api.max_cost_mtok
    ]

    if config.api.openrouter_avoid_free_endpoints:
        filtered_endpoints = [
            e for e in filtered_endpoints if e.cost[0] + e.cost[1] > 0
        ]

    log_msg = (
        f"Found {len(all_endpoints)} {'endpoints claiming logprobs support' if logprob_filter else 'total endpoints'}, "
        f"keeping {len(filtered_endpoints)} within max cost of ${config.api.max_cost_mtok}/Mtok"
    )
    if config.api.openrouter_avoid_free_endpoints:
        log_msg += " and excluding free endpoints"
    logger.info(log_msg)

    return filtered_endpoints


async def test_endpoint_token_usage(
    endpoint: Endpoint, max_input_tokens: int = 10, max_output_tokens: int = 1
) -> tuple[Endpoint | None, BadEndpoint | None]:
    """Test if an endpoint uses acceptable token counts.

    Returns:
        (valid_endpoint, bad_endpoint) - at least one will be None.
    """
    client = OpenRouterClient()
    try:
        response = await client.query(endpoint, "a", logprobs=False, temperature=0)
        if response.error:
            logger.info(f"{endpoint} token test: ❌ (error: {response.error.message})")
            return None, None  # Don't mark as bad on error, might be transient
        if (
            response.input_tokens > max_input_tokens
            or response.output_tokens > max_output_tokens
        ):
            logger.info(
                f"{endpoint} token test: ❌ "
                f"(input={response.input_tokens}, output={response.output_tokens})"
            )
            bad = BadEndpoint.from_endpoint(
                endpoint,
                BadEndpointReason(
                    token_usage=(response.input_tokens, response.output_tokens)
                ),
            )
            return None, bad
        logger.info(
            f"{endpoint} token test: ✅ "
            f"(input={response.input_tokens}, output={response.output_tokens})"
        )
        return endpoint, None
    except Exception:
        logger.exception(f"Error testing token usage for {endpoint}")
        return None, None  # Don't mark as bad on exception, might be transient


async def test_endpoint_logprobs(endpoint: Endpoint) -> Endpoint | None:
    """Test if an endpoint actually returns logprobs when queried with 'x'"""
    client = OpenRouterClient()
    try:
        response = await client.query(endpoint, "x")
        if response.error or len(
            response.logprobs.logprobs
        ) != endpoint.get_max_logprobs(config):
            log_msg = f"{endpoint} logprob support: ❌"
            if response.error:
                log_msg += f"\n{response.error}"
            else:
                log_msg += f"\n{len(response.logprobs.logprobs)} logprobs, expected {endpoint.get_max_logprobs(config)}"
            logger.info(log_msg)
            return None
        logger.info(f"{endpoint} logprob support: ✅")
        return endpoint
    except Exception:
        logger.exception(f"Error testing logprobs for {endpoint}")
        return None


async def test_endpoints_logprobs(endpoints: Iterable[Endpoint]) -> list[Endpoint]:
    """Return the endpoints that actually return logprobs when queried with 'x'"""
    tasks = [test_endpoint_logprobs(e) for e in endpoints]
    return [
        e for e in await gather_with_concurrency(config.api.max_workers, *tasks) if e
    ]


async def update_endpoints_lt():
    """Update the LT endpoints by removing stalled endpoints and adding new ones."""
    storage = ResultsStorage(config.data_dir)
    current_endpoints = config.endpoints_lt
    endpoints_to_keep = set([e for e in current_endpoints if not storage.is_stalled(e)])
    logger.info(
        f"Keeping {len(endpoints_to_keep)}/{len(current_endpoints)} non-stalled endpoints"
    )
    endpoints_claiming_logprobs = await get_endpoints(logprob_filter=True)

    # Update costs with latest values
    for endpoint in endpoints_to_keep:
        if endpoint in endpoints_claiming_logprobs:
            updated_endpoint = endpoints_claiming_logprobs[
                endpoints_claiming_logprobs.index(endpoint)
            ]
            if updated_endpoint.cost != endpoint.cost:
                logger.info(
                    f"Updating cost for {endpoint}: {endpoint.cost} -> {updated_endpoint.cost}"
                )
                endpoint.cost = updated_endpoint.cost

    endpoints_to_test = set(endpoints_claiming_logprobs) - endpoints_to_keep
    valid_endpoints = await test_endpoints_logprobs(endpoints_to_test)
    logger.info(f"Found {len(valid_endpoints)} valid new endpoints")
    final_endpoints = endpoints_to_keep | set(valid_endpoints)
    # Sort endpoints by total cost, then api, model and provider
    sorted_endpoints = sorted(
        final_endpoints, key=lambda e: (sum(e.cost), e.api, e.model, e.provider)
    )

    output_data = {"endpoints_lt": []}
    for e in sorted_endpoints:
        e_data = {
            "api": e.api,
            "model": e.model,
            "provider": e.provider,
            "cost": list(e.cost),
        }
        output_data["endpoints_lt"].append(e_data)

    with open(config.endpoints_yaml_path_lt, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Updated {config.endpoints_yaml_path_lt}")


async def update_endpoints_bi():
    """Update the BI endpoints with all providers for all models.

    Filters out endpoints that use too many tokens (>10 input or >1 output).
    Bad endpoints are stored in bad_endpoints_bi.yaml to avoid re-testing.
    Good endpoints are stored in endpoints_bi.yaml and also skipped on re-runs.
    Only new endpoints (not in good or bad lists) are tested.
    """
    all_endpoints = await get_endpoints(logprob_filter=False)
    all_endpoints_set = set(all_endpoints)

    known_bad = load_bad_endpoints_bi()
    known_good = set(config.endpoints_bi)
    logger.info(
        f"Loaded {len(known_good)} known good endpoints, "
        f"{len(known_bad)} known bad endpoints"
    )

    # Update costs for known good endpoints if they're still in the API response
    still_good = []
    for good_endpoint in known_good:
        if good_endpoint in all_endpoints_set:
            updated = next(e for e in all_endpoints if e == good_endpoint)
            still_good.append(updated)
        else:
            still_good.append(good_endpoint)
    logger.info(f"Keeping all {len(still_good)} known good endpoints")

    # Filter out known bad and known good endpoints - only test new ones
    endpoints_to_test = [
        e
        for e in all_endpoints
        if not any(b.matches(e) for b in known_bad) and e not in known_good
    ]
    skipped_bad = len(
        [e for e in all_endpoints if any(b.matches(e) for b in known_bad)]
    )
    skipped_good = len([e for e in all_endpoints if e in known_good])
    logger.info(
        f"Testing {len(endpoints_to_test)} new endpoints "
        f"(skipping {skipped_bad} known bad, {skipped_good} known good)"
    )

    # Test token usage for each new endpoint
    tasks = [test_endpoint_token_usage(e) for e in endpoints_to_test]
    results = await gather_with_concurrency(config.api.max_workers, *tasks)

    newly_valid = [r[0] for r in results if r[0] is not None]
    new_bad = [r[1] for r in results if r[1] is not None]

    logger.info(
        f"Token test results: {len(newly_valid)} valid, "
        f"{len(new_bad)} new bad endpoints"
    )

    # Update bad endpoints list
    all_bad = known_bad + new_bad
    save_bad_endpoints_bi(all_bad)
    logger.info(f"Saved {len(all_bad)} total bad endpoints to {BAD_ENDPOINTS_BI_PATH}")

    # Combine still-good and newly-valid endpoints
    all_good = still_good + newly_valid

    # Sort and save valid endpoints to endpoints_bi.yaml
    sorted_endpoints = sorted(
        all_good, key=lambda e: (sum(e.cost), e.api, e.model, e.provider)
    )

    output_data = {"endpoints_bi": []}
    for e in sorted_endpoints:
        output_data["endpoints_bi"].append(
            {
                "api": e.api,
                "model": e.model,
                "provider": e.provider,
                "cost": list(e.cost),
            }
        )

    with open(config.endpoints_yaml_path_bi, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    logger.info(
        f"Updated {config.endpoints_yaml_path_bi} with {len(sorted_endpoints)} endpoints"
    )


async def main():
    await update_endpoints_lt()
    await update_endpoints_bi()


if __name__ == "__main__":
    asyncio.run(main())
