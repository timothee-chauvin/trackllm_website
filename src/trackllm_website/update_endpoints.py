from decimal import Decimal

import aiohttp
import requests

from trackllm_website.api import OpenRouterClient
from trackllm_website.config import Endpoint, config, logger
from trackllm_website.util import gather_with_concurrency_streaming


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


async def test_endpoint_logprobs(endpoint: Endpoint) -> bool:
    """Test if an endpoint actually returns logprobs when queried with 'x'"""
    client = OpenRouterClient()
    try:
        logger.info(f"Testing logprobs for {endpoint}...")
        response = await client.query(endpoint, "x")
        if response.error:
            return False
        return len(response.logprobs) == endpoint.get_max_logprobs()
    except Exception:
        return False
