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


class _FakeErrorResp:
    ok = False
    request_info = None
    history = ()

    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp

    def post(self, url, json):
        return self._resp


def _query(resp):
    async def run():
        client = OpenRouterClient()
        real_session = client.session
        client.session = _FakeSession(resp)
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
    r = _query(_FakeResp(payload))
    assert r.error is not None
    assert r.error.http_code == 200
    assert "usage" in r.error.message and r.error.message != "'usage'"
    assert r.cost == 0.0


def test_missing_usage_error_includes_body_excerpt():
    # Some providers wrap the real error in a 200 body (e.g. mara's
    # {"error": ...}); the message must surface it, not just the key names.
    payload = {"error": {"message": "User is locked. Reason: Exhausted balance"}}
    r = _query(_FakeResp(payload))
    assert r.error is not None
    assert "User is locked" in r.error.message


def test_missing_usage_error_body_excerpt_is_trimmed():
    payload = {"error": {"message": "x" * 5000}}
    r = _query(_FakeResp(payload))
    assert r.error is not None
    assert len(r.error.message) < 1000


# Error bodies that are valid JSON but not a dict used to raise AttributeError
# on .get("error", ...), escaping query's error normalization entirely.


def test_json_string_error_body():
    r = _query(_FakeErrorResp(400, '"Internal Server Error"'))
    assert r.error is not None
    assert r.error.http_code == 400
    assert "Internal Server Error" in r.error.message


def test_json_null_error_body():
    r = _query(_FakeErrorResp(400, "null"))
    assert r.error is not None
    assert r.error.http_code == 400
    assert r.error.message


def test_json_list_error_body():
    r = _query(_FakeErrorResp(400, "[]"))
    assert r.error is not None
    assert r.error.http_code == 400
    assert r.error.message


def test_dict_error_body_still_extracts_error_field():
    r = _query(_FakeErrorResp(404, '{"error": {"message": "No allowed providers"}}'))
    assert r.error is not None
    assert r.error.http_code == 404
    assert "No allowed providers" in r.error.message


def test_get_generation_cost_logs_last_failure(monkeypatch, caplog):
    """Exhausting the retries must leave a log trail: a systematic failure (auth,
    schema change) would otherwise stall every caller with zero log lines."""
    from trackllm_website import api as api_mod

    async def no_sleep(_delay):
        pass

    monkeypatch.setattr(api_mod.asyncio, "sleep", no_sleep)

    def boom_get(*args, **kwargs):
        raise RuntimeError("boom")

    async def run():
        client = OpenRouterClient()
        monkeypatch.setattr(client.session, "get", boom_get)
        try:
            return await client.get_generation_cost("gen-123", session=client.session)
        finally:
            await client.close()

    with caplog.at_level("WARNING", logger="trackllm-website"):
        assert asyncio.run(run()) is None
    assert any("gen-123" in r.message and "boom" in r.message for r in caplog.records)
