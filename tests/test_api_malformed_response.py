import asyncio

from trackllm_website.api import OpenRouterClient
from trackllm_website.config import Endpoint


class _FakeResp:
    ok = True
    status = 200
    request_info = None
    history = ()

    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload

    def post(self, url, json):
        return _FakeResp(self._payload)


def _query(payload):
    async def run():
        client = OpenRouterClient()
        real_session = client.session
        client.session = _FakeSession(payload)
        try:
            return await client.query(
                Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1)),
                "x",
                logprobs=False,
            )
        finally:
            client.session = real_session
            await client.close()

    return asyncio.run(run())


def test_missing_usage_yields_clear_error_not_keyerror():
    # Some providers (e.g. io.net) return 200 OK without a usage field; this used
    # to surface as the bare KeyError string "'usage'".
    payload = {
        "id": "gen-1",
        "choices": [{"message": {"content": "hi"}}],
    }
    r = _query(payload)
    assert r.error is not None
    assert r.error.http_code == 200
    assert "usage" in r.error.message and r.error.message != "'usage'"
    assert r.cost == 0.0
