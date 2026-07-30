"""The unprobeable bucket: endpoints whose vetting probes persistently fail
(or provably cannot succeed) stop being re-probed daily."""

from trackllm_website.bi.vetting import EndpointCache, UnprobeableEntry
from trackllm_website.config import Endpoint

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
    from trackllm_website.update_endpoints import record_probe_failures

    cache = empty_cache()
    a, b = ep("a"), ep("b")
    failed = {str(a): [TOO_EXPENSIVE, "other"], str(b): ["no strategy worked"]}
    record_probe_failures(failed, [a, b], cache, threshold=THRESHOLD)
    assert str(a) not in cache.failure_streaks  # cost-rejected, not flaky
    assert cache.failure_streaks[str(b)].last_error == "no strategy worked"


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
