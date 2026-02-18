import asyncio
import hashlib
from collections.abc import AsyncIterator, Coroutine
from typing import Any

from trackllm_website.config import Endpoint


async def gather_with_concurrency(
    n: int, *coros: Coroutine[Any, Any, Any]
) -> list[Any]:
    # Taken from https://stackoverflow.com/a/61478547
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))


async def gather_with_concurrency_streaming(
    n: int, *coros: Coroutine[Any, Any, Any]
) -> AsyncIterator[Any]:
    """Run coroutines with limited concurrency, yielding results as they complete."""
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro: Coroutine[Any, Any, Any]) -> Any:
        async with semaphore:
            return await coro

    # Create the semaphore-wrapped coroutines
    sem_coros = [sem_coro(c) for c in coros]

    # Use as_completed to get results as they finish
    for coro in asyncio.as_completed(sem_coros):
        yield await coro


def trim_to_length(s: str, length: int) -> str:
    """Trim a string to a maximum length, adding ellipsis if truncated."""
    return s[:length] + "..." if len(s) > length else s


def slugify(s: str, max_length: int = 1000, hash_length: int = 0) -> str:
    """
    Convert a string to a slugified version suitable for Linux and MacOS filenames.

    Special characters are hex-encoded to preserve information while keeping
    the filename safe. For example, "|" becomes "7c".

    Args:
        s: The input string to slugify
        max_length: Maximum length of the output without the hash
        hash_length: Length of the hash to append to the output

    Returns:
        A slugified string safe for use as a Linux or MacOS filename
    """
    slug = ""

    for char in s:
        if char.isalnum() or char in "._-+=@~,":
            slug += char
        elif char == " ":
            slug += "-"
        else:
            slug += f"{ord(char):02x}"

    slug = slug[:max_length]

    if hash_length > 0:
        string_hash = hashlib.md5(s.encode("utf-8")).hexdigest()[:hash_length]
        slug += "_" + string_hash

    return slug


def endpoint_from_slug(slug: str) -> Endpoint:
    """Find an Endpoint whose str representation slugifies to the given slug."""
    from trackllm_website.config import config

    all_endpoints = (
        # config.endpoints_lt
        config.endpoints_bi
        # + config.endpoints_bi_phase_1
        # + config.endpoints_bi_prevalence
    )
    for endpoint in all_endpoints:
        if slugify(f"{endpoint.model}#{endpoint.provider}") == slug:
            return endpoint
    raise ValueError(f"No endpoint found for slug: {slug}")
