import asyncio
import random
from datetime import datetime
from typing import Any, Awaitable, Callable

import aiohttp
import numpy as np
from pydantic import BaseModel

from trackllm_website.config import Endpoint, config, logger


class LogprobVector(BaseModel, arbitrary_types_allowed=True):
    """A vector of returned logprobs and the corresponding tokens. May be returned to multiple queries if non-determinism is low."""

    tokens: list[str]
    logprobs: list[np.float32]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogprobVector):
            return False
        return self.tokens == other.tokens and self.logprobs == other.logprobs


class LogprobResponse(BaseModel):
    """A logprob vector returned to a specific query."""

    date: datetime
    logprob_vector: LogprobVector


class Response(BaseModel):
    endpoint: Endpoint
    prompt: str
    logprobs: LogprobResponse | None
    cost: float | int
    error: str | None = None


class OpenRouterClient:
    async def _make_request(
        self, endpoint: Endpoint, prompt: str, temperature: float = 0.0
    ) -> Response:
        request_data = {
            "model": endpoint.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 1,
            # "logprobs": True,
            # "top_logprobs": endpoint.get_max_logprobs(cfg=config),
            "temperature": temperature,
            "provider": {
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
        if endpoint.provider:
            request_data["provider"]["only"] = [endpoint.provider]

        print(request_data)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {config.openrouter_api_key}"},
                json=request_data,
            ) as resp:
                if not resp.ok:
                    error_text = await resp.text()
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=resp.status,
                        message=f"HTTP {resp.status}: {error_text}",
                    )
                response = await resp.json()
                print(response)
                # Sometimes we get a 200 OK response with a JSON that says "Internal Server Error" 500...
                if set(response.keys()) == {"error", "user_id"}:
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=int(response["error"]["code"]),
                        message=f"HTTP {response['error']['code']}: {response['error']['message']}",
                    )

        cost = compute_cost(response["usage"], endpoint)

        # Extract logprobs for the first token
        if response["choices"] and response["choices"][0]["logprobs"]:
            logprobs = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            tokens = [logprob["token"] for logprob in logprobs]
            probs = [logprob["logprob"] for logprob in logprobs]
            return Response(
                endpoint,
                prompt,
                tokens,
                probs,
                cost,
                response.get("system_fingerprint", None),
            )

        logger.error(f"No logprobs returned for {endpoint}")
        return Response(endpoint, prompt, [], [], cost, error="No logprobs returned")

    async def query(
        self, endpoint: Endpoint, prompt: str, temperature: float = 0.0
    ) -> Response:
        try:
            return await retry_with_exponential_backoff(
                self._make_request,
                endpoint,
                prompt,
                temperature,
                max_retries=config.api.max_retries,
            )
        except Exception as e:
            # TODO didn't always try config.api.max_retries times.
            logger.error(
                f"Error querying {endpoint} after {config.api.max_retries} retries: {e}"
            )
            return Response(
                endpoint=endpoint, prompt=prompt, logprobs=None, cost=0.0, error=str(e)
            )


def compute_cost(usage: dict, endpoint: Endpoint) -> float:
    return (
        usage["prompt_tokens"] * endpoint.cost[0] / 1e6
        + usage["completion_tokens"] * endpoint.cost[1] / 1e6
    )


async def retry_with_exponential_backoff(
    func: Callable[..., Awaitable[Any]],
    *args,
    max_retries: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (
        aiohttp.ClientError,
        asyncio.TimeoutError,
    ),
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504),
    **kwargs,
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: The async function to retry
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay between retries
        jitter: Whether to add random jitter to delay
        retryable_exceptions: Tuple of exception types that should trigger retries
        retryable_status_codes: HTTP status codes that should trigger retries
        **kwargs: Keyword arguments for the function
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            return result
        except aiohttp.ClientResponseError as e:
            # Check if this is a retryable HTTP status code
            if e.status in retryable_status_codes and attempt < max_retries:
                wait_time = min(max_delay, (base_delay * (2**attempt)))
                if jitter:
                    wait_time *= random.uniform(0.9, 1.1)

                logger.warning(
                    f"HTTP {e.status} error. Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries + 1})"
                )
                await asyncio.sleep(wait_time)
                last_exception = e
                continue
            else:
                raise e
        except retryable_exceptions as e:
            if attempt < max_retries:
                wait_time = min(max_delay, (base_delay * (2**attempt)))
                if jitter:
                    wait_time *= random.uniform(0.9, 1.1)

                logger.warning(
                    f"Retryable error: {e}. Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries + 1})"
                )
                await asyncio.sleep(wait_time)
                last_exception = e
                continue
            else:
                raise e
        except Exception as e:
            # Non-retryable exceptions are re-raised immediately
            logger.error(f"Non-retryable error: {e}")
            raise e

    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected end of retry loop")
