import asyncio
import random
from datetime import datetime, timezone
from typing import Any, Awaitable

import aiohttp
import numpy as np
import orjson
from beartype.typing import Callable

from trackllm_website.config import Endpoint, config, logger
from trackllm_website.spend import record_query
from trackllm_website.storage import (
    Response,
    ResponseError,
    ResponseLogprobs,
)
from trackllm_website.util import trim_to_length


class OpenRouterClient:
    def __init__(self, timeout: float | None = None):
        self.connector = aiohttp.TCPConnector(limit=600)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            headers={"Authorization": f"Bearer {config.require_openrouter_api_key()}"},
            timeout=aiohttp.ClientTimeout(total=timeout or config.api.timeout),
        )

    async def close(self):
        """Must be called when the client is done."""
        await self.session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def get_generation_cost(
        self, generation_id: str, session: aiohttp.ClientSession | None = None
    ) -> float | int | None:
        """Fetch actual cost from OpenRouter generation endpoint."""
        should_close = session is None
        session = session or aiohttp.ClientSession()
        try:
            for delay in (5, 10, 20, 40):
                await asyncio.sleep(delay)
                try:
                    async with session.get(
                        url="https://openrouter.ai/api/v1/generation",
                        headers={
                            "Authorization": f"Bearer {config.require_openrouter_api_key()}"
                        },
                        params={"id": generation_id},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if not resp.ok:
                            continue
                        data = await resp.json()
                        cost = data.get("data", {}).get("total_cost")
                        if cost is not None:
                            return cost
                except Exception:
                    continue
            return None
        finally:
            if should_close:
                await session.close()

    async def _make_request(
        self,
        endpoint: Endpoint,
        prompt: str,
        temperature: float | int = config.api.temperature,
        logprobs: bool = True,
        output_tokens: int | None = None,
        reasoning: dict | None = None,
    ) -> Response:
        request_data = {
            "model": endpoint.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": output_tokens or 1,
            "temperature": temperature,
            "provider": {
                "allow_fallbacks": False,
                # require_parameters would filter providers on max_completion_tokens vs
                # max_tokens naming; with `only` pinning a single provider it only causes
                # spurious 404s. Actual logprob support is checked by inspecting the response.
                "require_parameters": False,
            },
            "top_p": 1.0,
        }
        if logprobs:
            request_data["logprobs"] = True
            request_data["top_logprobs"] = endpoint.get_max_logprobs(cfg=config)
        if endpoint.provider:
            request_data["provider"]["only"] = [endpoint.provider]
        if reasoning is not None:
            request_data["reasoning"] = reasoning

        async with self.session.post(
            url="https://openrouter.ai/api/v1/chat/completions",
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

        # Some providers return 200 OK with no usage at all (e.g. io.net): nothing
        # to bill or count, so surface a clear error instead of a bare KeyError.
        if "usage" not in response:
            return Response(
                date=datetime.now(tz=timezone.utc),
                endpoint=endpoint,
                prompt=prompt,
                logprobs=None,
                cost=0.0,
                error=ResponseError(
                    http_code=resp.status,
                    message="No usage in response; body: "
                    f"{trim_to_length(orjson.dumps(response).decode(), 500)}",
                ),
            )

        usage = response["usage"]
        input_tokens = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]
        reasoning_tokens = (
            usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
        )
        cost = compute_cost(usage, endpoint)
        generation_id = response.get("id")

        # Extract content and reasoning
        content = None
        reasoning_content = None
        if response["choices"]:
            message = response["choices"][0].get("message", {})
            content = message.get("content")
            reasoning_content = message.get("reasoning")

        # Extract logprobs for the first token (if requested)
        response_logprobs = None
        if logprobs and response["choices"] and response["choices"][0].get("logprobs"):
            logprobs_data = response["choices"][0]["logprobs"]["content"][0][
                "top_logprobs"
            ]
            tokens = [logprob["token"] for logprob in logprobs_data]
            probs = [np.float32(logprob["logprob"]) for logprob in logprobs_data]
            response_logprobs = ResponseLogprobs(tokens=tokens, logprobs=probs)

        if logprobs and response_logprobs is None:
            return Response(
                date=datetime.now(tz=timezone.utc),
                endpoint=endpoint,
                prompt=prompt,
                content=content,
                logprobs=None,
                cost=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
                reasoning_content=reasoning_content,
                generation_id=generation_id,
                error=ResponseError(
                    http_code=resp.status,
                    message="No logprobs returned",
                ),
            )

        return Response(
            date=datetime.now(tz=timezone.utc),
            endpoint=endpoint,
            prompt=prompt,
            content=content,
            logprobs=response_logprobs,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            reasoning_content=reasoning_content,
            generation_id=generation_id,
            error=None,
        )

    async def query(
        self,
        endpoint: Endpoint,
        prompt: str,
        temperature: float | int = config.api.temperature,
        logprobs: bool = True,
        on_retry: Callable[[int], None] | None = None,
        output_tokens: int | None = None,
        reasoning: dict | None = None,
        max_retries: int | None = None,
        backoff_on_timeout: bool = True,
    ) -> Response:
        try:
            response = await retry_with_exponential_backoff(
                self._make_request,
                endpoint,
                prompt,
                temperature,
                logprobs,
                output_tokens,
                reasoning,
                max_retries=max_retries
                if max_retries is not None
                else config.api.max_retries,
                on_retry=on_retry,
                backoff_on_timeout=backoff_on_timeout,
            )
        except Exception as e:
            if isinstance(e, aiohttp.ClientResponseError):
                http_code, message_json = e.status, e.message
                try:
                    message = orjson.dumps(
                        orjson.loads(message_json.encode()).get("error", e)
                    ).decode()
                # For some reason, orjson.JSONDecodeError | orjson.JSONEncodeError fails with:
                # "catching classes that do not inherit from BaseException is not allowed"
                except (orjson.JSONDecodeError, orjson.JSONEncodeError):
                    message = str(e)
            elif isinstance(e, asyncio.TimeoutError):
                http_code, message = 0, f"Timeout after {config.api.timeout}s"
            else:
                http_code, message = 0, str(e)
            response = Response(
                date=datetime.now(tz=timezone.utc),
                endpoint=endpoint,
                prompt=prompt,
                logprobs=None,
                cost=0.0,
                error=ResponseError(http_code=http_code, message=message),
            )
        record_query(response.cost, response.error is not None)
        return response


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
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504),
    on_retry: Callable[[int], None] | None = None,
    backoff_on_timeout: bool = True,
    **kwargs,
) -> Any:
    """Retry an async function with exponential backoff.

    Args:
        on_retry: Optional callback called with HTTP status code when retrying.
        backoff_on_timeout: When False, timeout retries skip the sleep entirely.
    """

    def calc_delay(attempt: int) -> float:
        delay = min(max_delay, base_delay * (2**attempt))
        return delay * random.uniform(0.9, 1.1) if jitter else delay

    def is_retryable(e: Exception) -> tuple[bool, int | None, str]:
        if (
            isinstance(e, aiohttp.ClientResponseError)
            and e.status in retryable_status_codes
        ):
            return True, e.status, f"HTTP {e.status}"
        elif isinstance(e, asyncio.TimeoutError):
            return True, None, f"Timeout after {config.api.timeout}s"
        return False, None, ""

    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            retryable, status, msg = is_retryable(e)
            if not retryable or attempt >= max_retries:
                raise
            if on_retry and status:
                on_retry(status)
            if isinstance(e, asyncio.TimeoutError) and not backoff_on_timeout:
                wait_time = 0.0
            else:
                wait_time = calc_delay(attempt)
            logger.debug(
                f"{msg}. Retrying in {wait_time:.2f}s ({attempt + 1}/{max_retries + 1})"
            )
            if wait_time > 0.0:
                await asyncio.sleep(wait_time)
            last_exception = e

    raise last_exception or RuntimeError("Unexpected end of retry loop")
