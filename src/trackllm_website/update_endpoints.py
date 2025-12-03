import asyncio
from collections.abc import Iterable
from decimal import Decimal

import aiohttp
import requests
import yaml

from trackllm_website.api import OpenRouterClient
from trackllm_website.config import Endpoint, config, logger
from trackllm_website.storage import ResultsStorage
from trackllm_website.util import (
    gather_with_concurrency,
    gather_with_concurrency_streaming,
)


async def get_model_endpoints(session, model_id) -> list[Endpoint]:
    """Fetch endpoints for a model and return a list of endpoints that claim to support logprobs"""
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    try:
        async with session.get(url) as response:
            data = await response.json()
            model_endpoints = data["data"]["endpoints"]
            model_endpoints_with_logprobs = []
            for endpoint in model_endpoints:
                if not (
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
                model_endpoints_with_logprobs.append(endpoint_data)
            return model_endpoints_with_logprobs
    except Exception as e:
        logger.error(f"Error fetching endpoints for {model_id}: {e}")
        return []


async def get_endpoints_claiming_logprobs() -> list[Endpoint]:
    """Get all endpoints that claim to support logprobs"""
    response = requests.get("https://openrouter.ai/api/v1/models")
    model_ids = [model["id"] for model in response.json()["data"]]
    endpoints_claiming_logprobs = []
    async with aiohttp.ClientSession() as session:
        tasks = [get_model_endpoints(session, model_id) for model_id in model_ids]
        async for result in gather_with_concurrency_streaming(
            config.api.max_workers, *tasks
        ):
            endpoints_claiming_logprobs.extend(result)
    return endpoints_claiming_logprobs


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


async def update_endpoints_yaml():
    """Update endpoints.yaml by removing stalled endpoints and adding new ones."""
    storage = ResultsStorage(config.data_dir)
    current_endpoints = config.endpoints
    endpoints_to_keep = set([e for e in current_endpoints if not storage.is_stalled(e)])
    logger.info(
        f"Keeping {len(endpoints_to_keep)}/{len(current_endpoints)} non-stalled endpoints"
    )
    endpoints_claiming_logprobs_unfiltered = await get_endpoints_claiming_logprobs()
    endpoints_claiming_logprobs = [
        e
        for e in endpoints_claiming_logprobs_unfiltered
        if e.cost[0] + e.cost[1] < config.api.max_cost_mtok
    ]
    logger.info(
        f"Found {len(endpoints_claiming_logprobs_unfiltered)} endpoints claiming logprobs support, keeping {len(endpoints_claiming_logprobs)} within max cost of ${config.api.max_cost_mtok}/Mtok"
    )

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

    output_data = {"endpoints": []}
    for e in sorted_endpoints:
        e_data = {
            "api": e.api,
            "model": e.model,
            "provider": e.provider,
            "cost": list(e.cost),
        }
        output_data["endpoints"].append(e_data)

    with open(config.endpoints_yaml_path, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Updated {config.endpoints_yaml_path}")


async def main():
    await update_endpoints_yaml()


if __name__ == "__main__":
    asyncio.run(main())
