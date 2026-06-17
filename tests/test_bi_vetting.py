import asyncio
from datetime import datetime, timedelta, timezone

from trackllm_website.bi.common import PlainStrategy
from trackllm_website.bi.selection import Rule, SelectionPolicy
from trackllm_website.bi.vetting import EndpointCache, should_recheck, vet_endpoint
from trackllm_website.config import Endpoint
from trackllm_website.storage import Response, ResponseError
from trackllm_website.update_endpoints import exceeds_ceiling, merge_goods

EP = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1.0, 2.0))


class FakeClient:
    def __init__(self, response, gen_cost):
        self._response, self._gen_cost = response, gen_cost
        self.session = None

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


def _ceiling_policy(ceiling, flagships):
    return SelectionPolicy(
        budget_per_month=10,
        max_endpoint_cost=ceiling,
        exclude=[],
        rules=[
            Rule(
                name="flagships",
                kind="models",
                patterns=flagships,
                providers_per_model=1,
                flagship=True,
            )
        ],
    )


def test_non_flagship_above_ceiling_is_too_expensive():
    pol = _ceiling_policy(0.5, ["m/flag"])
    # 0.0001 * 6000 = 0.6 > 0.5 ceiling
    assert exceeds_ceiling(0.0001, "m/other", "p", pol) is True


def test_flagship_above_ceiling_is_kept():
    pol = _ceiling_policy(0.5, ["m/flag"])
    assert exceeds_ceiling(0.0001, "m/flag", "p", pol) is False


def test_merge_goods_carries_forward_transient_flakes():
    # A re-measured this run (fresh cpr), B cached as a liar, C flaked transiently
    # (not freshly good, not cached) => carried forward with its prior cpr.
    def mk(model, cpr):
        e = Endpoint(api="openrouter", model=model, provider="p", cost=(1, 1))
        e.cost_per_request = cpr
        return e

    prior = [mk("m/a", 0.1), mk("m/b", 0.2), mk("m/c", 0.3)]
    freshly_good = [mk("m/a", 0.15)]
    cache = EndpointCache(liars=[], too_expensive=[], bad_temperature=[])
    cache.add_liar(mk("m/b", 0.2))

    result = merge_goods(prior, freshly_good, cache)
    by_model = {e.model: e for e in result}

    assert set(by_model) == {"m/a", "m/c"}  # B (cached liar) excluded
    assert by_model["m/a"].cost_per_request == 0.15  # fresh measurement wins
    assert by_model["m/c"].cost_per_request == 0.3  # carried with prior cpr


def test_merge_goods_drops_carried_without_cost():
    # D flaked transiently (not freshly good, not cached) but has no prior
    # cost_per_request => can't be priced/selected, so it's dropped.
    def mk(model, cpr):
        e = Endpoint(api="openrouter", model=model, provider="p", cost=(1, 1))
        e.cost_per_request = cpr
        return e

    prior = [mk("m/a", 0.1), mk("m/d", None)]
    cache = EndpointCache(liars=[], too_expensive=[], bad_temperature=[])

    result = merge_goods(prior, freshly_good=[], cache=cache)
    assert {e.model for e in result} == {"m/a"}


def test_cache_round_trip(tmp_path):
    cache = EndpointCache(liars=[], too_expensive=[], bad_temperature=[])
    cache.add_liar(EP)
    path = tmp_path / "endpoints_cache_bi.yaml"
    cache.save(path)
    loaded = EndpointCache.load(path)
    assert loaded.is_cached(EP)
    assert loaded.bucket_of(EP) == "liar"


def test_cache_round_trips_last_recheck(tmp_path):
    when = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    cache = EndpointCache(
        liars=[], too_expensive=[], bad_temperature=[], last_recheck=when
    )
    path = tmp_path / "endpoints_cache_bi.yaml"
    cache.save(path)
    loaded = EndpointCache.load(path)
    assert loaded.last_recheck == when


def test_cache_round_trips_none_last_recheck(tmp_path):
    cache = EndpointCache(liars=[], too_expensive=[], bad_temperature=[])
    path = tmp_path / "endpoints_cache_bi.yaml"
    cache.save(path)
    assert EndpointCache.load(path).last_recheck is None


NOW = datetime(2026, 6, 16, tzinfo=timezone.utc)


def test_should_recheck_when_never_rechecked():
    cache = EndpointCache(liars=[], too_expensive=[], bad_temperature=[])
    assert should_recheck(cache, NOW, recheck_days=14) is True


def test_should_recheck_when_interval_elapsed():
    cache = EndpointCache(
        liars=[],
        too_expensive=[],
        bad_temperature=[],
        last_recheck=NOW - timedelta(days=14),
    )
    assert should_recheck(cache, NOW, recheck_days=14) is True


def test_should_not_recheck_when_recent():
    cache = EndpointCache(
        liars=[],
        too_expensive=[],
        bad_temperature=[],
        last_recheck=NOW - timedelta(days=2),
    )
    assert should_recheck(cache, NOW, recheck_days=14) is False
