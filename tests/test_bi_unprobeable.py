"""The unprobeable bucket: endpoints whose vetting probes persistently fail
(or provably cannot succeed) stop being re-probed daily."""

from trackllm_website.bi.selection import SelectionPolicy
from trackllm_website.bi.vetting import (
    EndpointCache,
    FailureStreak,
    UnprobeableEntry,
    VetResult,
)
from trackllm_website.config import Endpoint
from trackllm_website.update_endpoints import (
    partition_batch,
    record_probe_failures,
    route_vet_result,
)

THRESHOLD = 3


def ep(model="m", provider="p"):
    return Endpoint(api="openrouter", model=model, provider=provider, cost=(1, 1))


def empty_cache():
    return EndpointCache(liars=[], too_expensive=[], bad_temperature=[])


def test_failures_below_threshold_keep_endpoint_probeable():
    cache = empty_cache()
    for _ in range(THRESHOLD - 1):
        cache.record_failure(ep(), "boom", threshold=THRESHOLD)
    assert not cache.is_cached(ep())


def test_reaching_threshold_moves_endpoint_to_unprobeable():
    cache = empty_cache()
    for _ in range(THRESHOLD):
        cache.record_failure(ep(), "timeout contacting provider", threshold=THRESHOLD)
    assert cache.bucket_of(ep()) == "unprobeable"
    (entry,) = cache.unprobeable
    assert entry.reason == "flaky"
    assert entry.detail == "timeout contacting provider"
    assert str(ep()) not in cache.failure_streaks


def test_success_resets_streak():
    cache = empty_cache()
    for _ in range(THRESHOLD - 1):
        cache.record_failure(ep(), "boom", threshold=THRESHOLD)
    cache.record_success(ep())
    for _ in range(THRESHOLD - 1):
        cache.record_failure(ep(), "boom", threshold=THRESHOLD)
    assert not cache.is_cached(ep())


def test_add_unprobeable_deduplicates():
    cache = empty_cache()
    cache.add_unprobeable(ep(), reason="batch", detail=None)
    cache.add_unprobeable(ep(), reason="batch", detail=None)
    assert len(cache.unprobeable) == 1


def test_save_load_roundtrip(tmp_path):
    cache = empty_cache()
    cache.add_unprobeable(ep("a"), reason="batch", detail=None)
    cache.record_failure(ep("b"), "boom", threshold=THRESHOLD)
    path = tmp_path / "cache.yaml"
    cache.save(path)
    loaded = EndpointCache.load(path)
    assert loaded.unprobeable == [
        UnprobeableEntry(endpoint=ep("a"), reason="batch", detail=None)
    ]
    assert loaded.failure_streaks[str(ep("b"))].count == 1
    assert loaded.failure_streaks[str(ep("b"))].last_error == "boom"


def test_recheck_clears_unprobeable_and_streaks():
    from trackllm_website.bi.vetting import clear_recheckable

    cache = empty_cache()
    cache.too_expensive.append(ep("a"))
    cache.bad_temperature.append(ep("b"))
    cache.add_unprobeable(ep("c"), reason="flaky", detail="boom")
    cache.record_failure(ep("d"), "boom", threshold=THRESHOLD)
    assert clear_recheckable(cache) == 3
    assert cache.too_expensive == []
    assert cache.bad_temperature == []
    assert cache.unprobeable == []
    assert cache.failure_streaks == {}


def test_record_probe_failures_skips_too_expensive():
    from trackllm_website.bi.common import TOO_EXPENSIVE

    cache = empty_cache()
    a, b = ep("a"), ep("b")
    failed = {str(a): [TOO_EXPENSIVE, "other"], str(b): ["no strategy worked"]}
    record_probe_failures(failed, [a, b], cache, threshold=THRESHOLD, exempt=set())
    assert str(a) not in cache.failure_streaks  # cost-rejected, not flaky
    assert cache.failure_streaks[str(b)].last_error == "no strategy worked"


def test_record_failure_trims_long_errors():
    cache = empty_cache()
    cache.record_failure(ep(), "x" * 1000, threshold=1)
    (entry,) = cache.unprobeable
    assert len(entry.detail) < 250


def test_definitive_buckets_end_the_streak():
    for add in ("add_liar", "add_too_expensive", "add_bad_temperature"):
        cache = empty_cache()
        cache.record_failure(ep(), "boom", threshold=THRESHOLD)
        getattr(cache, add)(ep())
        assert cache.failure_streaks == {}, add


def test_record_probe_failures_exempts_prior_goods():
    cache = empty_cache()
    a, b = ep("a"), ep("b")
    failed = {str(a): ["err"], str(b): ["err"]}
    record_probe_failures(failed, [a, b], cache, threshold=THRESHOLD, exempt={a})
    assert str(a) not in cache.failure_streaks
    assert str(b) in cache.failure_streaks


class TestRouteVetResult:
    def route(self, res, endpoint, cache, prior_good):
        policy = SelectionPolicy(
            budget_per_month=10, max_endpoint_cost=0.5, exclude=[], rules=[]
        )
        return route_vet_result(
            res, endpoint, cache, policy, threshold=1, prior_good=prior_good
        )

    def test_transient_accrues_streak_for_new_endpoint(self):
        cache = empty_cache()
        res = VetResult(bucket="transient", detail="boom")
        assert self.route(res, ep(), cache, prior_good=False) is None
        assert cache.bucket_of(ep()) == "unprobeable"  # threshold=1

    def test_transient_is_exempt_for_prior_goods(self):
        cache = empty_cache()
        res = VetResult(bucket="transient", detail="boom")
        assert self.route(res, ep(), cache, prior_good=True) is None
        assert not cache.is_cached(ep())  # merge_goods carry-forward applies

    def cheap(self):
        # exceeds_ceiling compares cost_per_request * samples_per_month
        from trackllm_website.config import config

        return 0.5 / config.bi.samples_per_month / 2

    def test_candidate_resets_streak_and_returns_endpoint(self):
        cache = empty_cache()
        cache.failure_streaks[str(ep())] = FailureStreak(count=1, last_error="old")
        res = VetResult(bucket="candidate", cost_per_request=self.cheap())
        kept = self.route(res, ep(), cache, prior_good=False)
        assert kept is not None and kept.cost_per_request == self.cheap()
        assert cache.failure_streaks == {}

    def test_candidate_over_ceiling_is_too_expensive(self):
        cache = empty_cache()
        res = VetResult(bucket="candidate", cost_per_request=self.cheap() * 4)
        assert self.route(res, ep(), cache, prior_good=False) is None
        assert cache.bucket_of(ep()) == "too_expensive"

    def test_liar_is_cached(self):
        cache = empty_cache()
        res = VetResult(bucket="liar")
        assert self.route(res, ep(), cache, prior_good=False) is None
        assert cache.bucket_of(ep()) == "liar"


def test_partition_batch_skips_batch_models():
    eps = [ep("org/m"), ep("org/m:batch"), ep("org/m2:batch", provider="q")]
    probe, skip = partition_batch(eps)
    assert [e.model for e in probe] == ["org/m"]
    assert [e.model for e in skip] == ["org/m:batch", "org/m2:batch"]


def test_load_of_legacy_cache_without_new_keys(tmp_path):
    cache = empty_cache()
    cache.save(tmp_path / "cache.yaml")
    # simulate a pre-unprobeable file by dropping the new keys
    text = (tmp_path / "cache.yaml").read_text()
    legacy = "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith(("unprobeable", "failure_streaks"))
    )
    (tmp_path / "legacy.yaml").write_text(legacy)
    loaded = EndpointCache.load(tmp_path / "legacy.yaml")
    assert loaded.unprobeable == []
    assert loaded.failure_streaks == {}
