import asyncio
import hashlib
import os
import tempfile
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from typing import Any

from trackllm_website.config import Endpoint


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to path atomically via a tempfile in the same directory + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as tmp_file:
        tmp_file.write(data)
        temp_name = tmp_file.name
    os.replace(temp_name, path)


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
    """Run coroutines with limited concurrency, yielding results as they complete.

    Tasks are started in order as slots become available, preserving submission order.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    coro_iter = iter(coros)
    active_tasks: set[asyncio.Task[None]] = set()
    done_count = 0
    total = len(coros)

    async def worker(coro: Coroutine[Any, Any, Any]) -> None:
        result = await coro
        await queue.put(result)

    def start_next() -> bool:
        """Start the next coroutine if available. Returns True if one was started."""
        try:
            coro = next(coro_iter)
            task = asyncio.create_task(worker(coro))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)
            return True
        except StopIteration:
            return False

    # Start initial batch
    for _ in range(min(n, total)):
        start_next()

    # Yield results and start new tasks as slots free up
    while done_count < total:
        result = await queue.get()
        done_count += 1
        start_next()
        yield result


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

    all_endpoints = config.endpoints_bi + config.endpoints_bi_prevalence
    for endpoint in all_endpoints:
        if slugify(f"{endpoint.model}#{endpoint.provider}") == slug:
            return endpoint

    # Fall back to the state files, which form the registry of historically
    # monitored endpoints that may no longer be in the live OpenRouter catalog
    # (and hence dropped from the config lists). Lazy-imported to avoid a cycle:
    # state.py imports slugify from this module.
    from trackllm_website.bi.state import load_all_states

    state = load_all_states(config.bi.state_dir).get(slug)
    if state is not None:
        return state.endpoint

    raise ValueError(f"No endpoint found for slug: {slug}")
