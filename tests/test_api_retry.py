import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from trackllm_website import api
from trackllm_website.api import get_json_with_retry, retry_with_exponential_backoff


@pytest.fixture
def no_backoff(monkeypatch):
    async def instant(_delay):
        pass

    monkeypatch.setattr(api.asyncio, "sleep", instant)


def test_retries_connection_errors(no_backoff):
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise aiohttp.ClientPayloadError("Response payload is not completed")
        if calls["n"] == 2:
            raise aiohttp.ServerDisconnectedError()
        return "ok"

    assert asyncio.run(retry_with_exponential_backoff(flaky, max_retries=3)) == "ok"
    assert calls["n"] == 3


def test_does_not_retry_other_errors(no_backoff):
    calls = {"n": 0}

    async def broken():
        calls["n"] += 1
        raise ValueError("bad")

    with pytest.raises(ValueError):
        asyncio.run(retry_with_exponential_backoff(broken, max_retries=3))
    assert calls["n"] == 1


def _flaky_app(calls: dict) -> web.Application:
    """1st request: chunked body cut mid-stream (the CI failure); 2nd: 502; then OK."""

    async def handler(request: web.Request) -> web.StreamResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            resp = web.StreamResponse()
            await resp.prepare(request)
            await resp.write(b'{"data": [')
            request.transport.close()
            return resp
        if calls["n"] == 2:
            raise web.HTTPBadGateway()
        return web.json_response({"data": [1, 2]})

    app = web.Application()
    app.router.add_get("/models", handler)
    return app


def test_get_json_with_retry_survives_truncation_and_5xx(no_backoff):
    calls = {"n": 0}

    async def run():
        async with TestServer(_flaky_app(calls)) as server:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            ) as session:
                return await get_json_with_retry(
                    session, str(server.make_url("/models"))
                )

    assert asyncio.run(run()) == {"data": [1, 2]}
    assert calls["n"] == 3


def test_get_json_with_retry_raises_on_404(no_backoff):
    calls = {"n": 0}

    async def run():
        async with TestServer(_flaky_app(calls)) as server:
            async with aiohttp.ClientSession() as session:
                await get_json_with_retry(session, str(server.make_url("/missing")))

    with pytest.raises(aiohttp.ClientResponseError) as exc:
        asyncio.run(run())
    assert exc.value.status == 404
    assert calls["n"] == 0
