"""Per-run discovery of an endpoint's top_logprobs cap.

Some providers reject top_logprobs above an undocumented cap with a 400 whose
message format varies (Novita says "[0, 5]", StreamLake gives no range at all),
so instead of parsing errors we probe descending values. The discovered cap is
stamped onto the endpoint in memory only: endpoints_lt.yaml is rewritten daily
without max_logprobs, and caps can change provider-side at any time.
"""

import asyncio

from beartype.typing import Awaitable, Callable

from trackllm_website.config import Endpoint, config
from trackllm_website.storage import Response

LOGPROB_LADDER = (20, 10, 8, 5, 3, 2, 1)

QueryFn = Callable[[Endpoint, str], Awaitable[Response]]


async def query_discovering_max_logprobs(
    query: QueryFn, endpoint: Endpoint, prompt: str
) -> Response:
    """Query once; on a 400, walk down LOGPROB_LADDER until a value is accepted.

    The first accepted value is stored in endpoint.max_logprobs so the run's
    subsequent queries use it. If every rung 400s, the error was not about
    top_logprobs: the endpoint's original depth is restored.
    """
    response = await query(endpoint, prompt)
    original = endpoint.max_logprobs
    initial = endpoint.get_max_logprobs(cfg=config)
    for n in [v for v in LOGPROB_LADDER if v < initial]:
        if response.error is None or response.error.http_code != 400:
            return response
        endpoint.max_logprobs = n
        response = await query(endpoint, prompt)
    if response.error is not None and response.error.http_code == 400:
        endpoint.max_logprobs = original
    return response


async def query_endpoint(
    query: QueryFn, endpoint: Endpoint, prompts: list[str]
) -> list[Response]:
    """Query all prompts for one endpoint, first one alone to discover the
    top_logprobs cap before the others run concurrently."""
    first, *rest = prompts
    responses = [await query_discovering_max_logprobs(query, endpoint, first)]
    responses.extend(await asyncio.gather(*(query(endpoint, p) for p in rest)))
    return responses
