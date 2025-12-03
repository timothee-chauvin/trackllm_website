import asyncio
import random
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import aiohttp
import numpy as np
import orjson

from trackllm_website.config import Endpoint, config, logger
from trackllm_website.storage import (
    Response,
    ResponseError,
    ResponseLogprobs,
)


class OpenRouterClient:
    async def _make_request(
        self, endpoint: Endpoint, prompt: str, temperature: float = 0.0
    ) -> Response:
        request_data = {
            "model": endpoint.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": 1,
            "logprobs": True,
            "top_logprobs": endpoint.get_max_logprobs(cfg=config),
            "temperature": temperature,
            "provider": {
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
        if endpoint.provider:
            request_data["provider"]["only"] = [endpoint.provider]

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
                        message=error_text,
                    )
                response = await resp.json()
                # Sometimes we get a 200 OK response with a JSON that says "Internal Server Error" 500...
                if set(response.keys()) == {"error", "user_id"}:
                    raise aiohttp.ClientResponseError(
                        request_info=resp.request_info,
                        history=resp.history,
                        status=int(response["error"]["code"]),
                        message=response["error"]["message"],
                    )

        cost = compute_cost(response["usage"], endpoint)

        # Extract logprobs for the first token
        if response["choices"] and response["choices"][0]["logprobs"]:
            logprobs_data = response["choices"][0]["logprobs"]["content"][0][
                "top_logprobs"
            ]
            tokens = [logprob["token"] for logprob in logprobs_data]
            probs = [np.float32(logprob["logprob"]) for logprob in logprobs_data]

            logprobs = ResponseLogprobs(tokens=tokens, logprobs=probs)

            return Response(
                date=datetime.now(tz=timezone.utc),
                endpoint=endpoint,
                prompt=prompt,
                logprobs=logprobs,
                cost=cost,
                error=None,
            )

        logger.error(f"No logprobs returned for {endpoint}")
        return Response(
            date=datetime.now(tz=timezone.utc),
            endpoint=endpoint,
            prompt=prompt,
            logprobs=None,
            cost=cost,
            error=ResponseError(
                http_code=resp.status,
                message="No logprobs returned",
            ),
        )

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
            if isinstance(e, aiohttp.ClientResponseError):
                http_code, message_json = e.status, e.message
                try:
                    message = orjson.dumps(
                        orjson.loads(message_json.encode()).get("error", e)
                    ).decode()
                except orjson.JSONDecodeError | orjson.JSONEncodeError:
                    message = str(e)
                logger.error(f"Error querying {endpoint}: {http_code} {message}")
            else:
                http_code, message = 0, str(e)
            return Response(
                date=datetime.now(tz=timezone.utc),
                endpoint=endpoint,
                prompt=prompt,
                logprobs=None,
                cost=0.0,
                error=ResponseError(http_code=http_code, message=message),
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
    """Retry an async function with exponential backoff."""

    def calc_delay(attempt: int) -> float:
        delay = min(max_delay, base_delay * (2**attempt))
        return delay * random.uniform(0.9, 1.1) if jitter else delay

    def is_retryable(e: Exception) -> tuple[bool, str]:
        if (
            isinstance(e, aiohttp.ClientResponseError)
            and e.status in retryable_status_codes
        ):
            return True, f"HTTP {e.status}"
        if isinstance(e, retryable_exceptions):
            return True, str(e)
        return False, ""

    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            retryable, msg = is_retryable(e)
            if not retryable or attempt >= max_retries:
                raise
            wait_time = calc_delay(attempt)
            logger.warning(
                f"{msg}. Retrying in {wait_time:.2f}s ({attempt + 1}/{max_retries + 1})"
            )
            await asyncio.sleep(wait_time)
            last_exception = e

    raise last_exception or RuntimeError("Unexpected end of retry loop")
