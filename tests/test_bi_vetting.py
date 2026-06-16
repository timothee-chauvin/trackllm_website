import asyncio
from datetime import datetime, timezone

from trackllm_website.bi.common import PlainStrategy
from trackllm_website.bi.vetting import EndpointCache, vet_endpoint
from trackllm_website.config import Endpoint
from trackllm_website.storage import Response, ResponseError

EP = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1.0, 2.0))


class FakeClient:
    def __init__(self, response, gen_cost):
        self._response, self._gen_cost = response, gen_cost

    async def query(self, endpoint, prompt, **kwargs):
        return self._response

    async def get_generation_cost(self, gen_id, session=None):
        return self._gen_cost


def ok_response(cost, gen_id="g1"):
    # A real Response carrying content + a billed cost.
    return Response(
        date=datetime.now(tz=timezone.utc),
        endpoint=EP,
        prompt="a",
        content="Hello",
        cost=cost,
        input_tokens=7,
        output_tokens=1,
        reasoning_tokens=0,
        generation_id=gen_id,
        error=None,
    )


def test_vet_records_measured_cost():
    # expected (response.cost) = 0.00001; actual billed matches => candidate,
    # and cost_per_request is the ACTUAL billed cost.
    client = FakeClient(ok_response(cost=0.00001), gen_cost=0.00001)
    res = asyncio.run(vet_endpoint(client, EP, PlainStrategy()))
    assert res.bucket == "candidate"
    assert abs(res.cost_per_request - 0.00001) < 1e-12


def test_vet_flags_pricing_liar():
    # expected = 0.00001 but provider bills 0.0002 (20x) => liar.
    client = FakeClient(ok_response(cost=0.00001), gen_cost=0.0002)
    res = asyncio.run(vet_endpoint(client, EP, PlainStrategy()))
    assert res.bucket == "liar"


def test_vet_transient_when_no_generation_cost():
    # Honest response but the actual billed cost can't be fetched => transient.
    client = FakeClient(ok_response(cost=0.00001), gen_cost=None)
    res = asyncio.run(vet_endpoint(client, EP, PlainStrategy()))
    assert res.bucket == "transient"


def test_vet_transient_error_is_not_cached():
    resp = Response(
        date=datetime.now(tz=timezone.utc),
        endpoint=EP,
        prompt="a",
        content=None,
        error=ResponseError(http_code=500, message="boom"),
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cost=0.0,
        generation_id=None,
    )
    client = FakeClient(resp, gen_cost=None)
    res = asyncio.run(vet_endpoint(client, EP, PlainStrategy()))
    assert res.bucket == "transient"  # don't cache; retry next run


def test_cache_round_trip(tmp_path):
    cache = EndpointCache(liars=[], too_expensive=[], bad_temperature=[])
    cache.add_liar(EP)
    path = tmp_path / "endpoints_cache_bi.yaml"
    cache.save(path)
    loaded = EndpointCache.load(path)
    assert loaded.is_cached(EP)
    assert loaded.bucket_of(EP) == "liar"
