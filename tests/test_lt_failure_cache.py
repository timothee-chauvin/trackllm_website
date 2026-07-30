from datetime import datetime, timezone

import yaml

from trackllm_website.config import Endpoint
from trackllm_website.update_endpoints import LTFailureCache, update_lt_failure_cache

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
LATER = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)


def ep(model, provider):
    return Endpoint(api="openrouter", model=model, provider=provider, cost=(1, 2))


def test_record_and_clear():
    cache = LTFailureCache(failures=[])
    e = ep("org/m", "p")
    cache.record(e, "error: boom", NOW)
    assert [(f.model, f.provider, f.reason, f.last_seen) for f in cache.failures] == [
        ("org/m", "p", "error: boom", NOW)
    ]
    cache.clear(e)
    assert cache.failures == []


def test_record_updates_existing_entry():
    cache = LTFailureCache(failures=[])
    e = ep("org/m", "p")
    cache.record(e, "returned 5 logprobs, expected 20", NOW)
    cache.record(e, "error: boom", LATER)
    assert len(cache.failures) == 1
    assert cache.failures[0].reason == "error: boom"
    assert cache.failures[0].last_seen == LATER


def test_clear_leaves_other_endpoints():
    cache = LTFailureCache(failures=[])
    cache.record(ep("org/m", "p"), "error: boom", NOW)
    cache.record(ep("org/m", "q"), "error: boom", NOW)
    cache.clear(ep("org/m", "p"))
    assert [(f.model, f.provider) for f in cache.failures] == [("org/m", "q")]


def test_persist_round_trip(tmp_path):
    path = tmp_path / "endpoints_cache_lt.yaml"
    cache = LTFailureCache(failures=[])
    cache.record(ep("org/b", "p"), "error: boom", NOW)
    cache.record(ep("org/a", "p"), "returned 5 logprobs, expected 20", NOW)
    cache.save(path)

    raw = yaml.safe_load(path.read_text())
    assert [(f["model"], f["provider"]) for f in raw["failures"]] == [
        ("org/a", "p"),
        ("org/b", "p"),
    ]

    loaded = LTFailureCache.load(path)
    assert sorted((f.model, f.provider, f.reason, f.last_seen) for f in loaded.failures) == sorted(
        (f.model, f.provider, f.reason, f.last_seen) for f in cache.failures
    )


def test_load_missing_file(tmp_path):
    assert LTFailureCache.load(tmp_path / "nope.yaml").failures == []


def test_load_empty_failures(tmp_path):
    path = tmp_path / "endpoints_cache_lt.yaml"
    path.write_text("failures: []\n")
    assert LTFailureCache.load(path).failures == []


def test_update_lt_failure_cache_records_and_clears():
    cache = LTFailureCache(failures=[])
    passes_now = ep("org/fixed", "p")
    cache.record(passes_now, "error: old failure", NOW)
    still_failing = ep("org/broken", "p")
    update_lt_failure_cache(
        cache, [passes_now], {still_failing: "returned 5 logprobs, expected 20"}, LATER
    )
    assert [(f.model, f.provider, f.reason, f.last_seen) for f in cache.failures] == [
        ("org/broken", "p", "returned 5 logprobs, expected 20", LATER)
    ]
