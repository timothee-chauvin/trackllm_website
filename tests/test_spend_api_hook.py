import asyncio

from trackllm_website.api import OpenRouterClient
from trackllm_website.config import Endpoint
from trackllm_website.spend import track
from trackllm_website.storage import Response, ResponseError


def _ep():
    return Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))


def _resp(cost, error=None):
    return Response(
        date=__import__("datetime").datetime.now(
            tz=__import__("datetime").timezone.utc
        ),
        endpoint=_ep(),
        prompt="x",
        content="y",
        logprobs=None,
        cost=cost,
        input_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        reasoning_content=None,
        generation_id="g",
        error=error,
    )


def test_query_records_cost_and_errors_into_bucket(monkeypatch):
    seq = [
        _resp(0.10),
        _resp(0.0, ResponseError(http_code=500, message="boom")),
        _resp(0.20),
    ]

    async def fake_make_request(self, *a, **k):
        return seq.pop(0)

    monkeypatch.setattr(OpenRouterClient, "_make_request", fake_make_request)

    async def run():
        client = OpenRouterClient()
        try:
            with track() as s:
                await asyncio.gather(
                    client.query(_ep(), "x"),
                    client.query(_ep(), "x"),
                    client.query(_ep(), "x"),
                )
            return s
        finally:
            await client.close()

    s = asyncio.run(run())
    assert s.n_queries == 3
    assert s.n_errors == 1
    assert abs(s.cost - 0.30) < 1e-9


def test_query_noop_without_bucket(monkeypatch):
    async def fake_make_request(self, *a, **k):
        return _resp(0.10)

    monkeypatch.setattr(OpenRouterClient, "_make_request", fake_make_request)

    async def run():
        client = OpenRouterClient()
        try:
            return await client.query(_ep(), "x")  # no bucket → must not raise
        finally:
            await client.close()

    assert asyncio.run(run()).cost == 0.10
