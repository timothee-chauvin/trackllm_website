from pathlib import Path

import orjson

from trackllm_website.bi.scan import changepoint_scan, split_statistic
from trackllm_website.config import config

FIXTURES = Path("tests/fixtures/phase_2")


def load_fixture(slug: str) -> dict:
    return orjson.loads((FIXTURES / slug / "data.json").read_bytes())


def first_n_batches(results: dict, n: int) -> dict:
    all_ts = sorted({ts for batches in results.values() for ts in batches})
    keep = set(all_ts[:n])
    return {p: {ts: s for ts, s in b.items() if ts in keep} for p, b in results.items()}


def test_split_statistic_is_summed_squared_frequency_distance():
    pre = {"p1": {"a": 8, "b": 2}, "p2": {"x": 10}}
    post = {"p1": {"a": 2, "b": 8}, "p2": {"x": 5, "y": 5}}
    # p1: (0.8-0.2)^2 + (0.2-0.8)^2 = 0.72; p2: (1-0.5)^2 + (0-0.5)^2 = 0.5
    assert abs(split_statistic(pre, post) - 1.22) < 1e-9


def test_too_few_batches_no_scan():
    results = first_n_batches(load_fixture("openai2fgpt-4o-mini23azure"), 3)
    assert changepoint_scan(results) is None


def test_window_passed_no_scan():
    results = first_n_batches(
        load_fixture("openai2fgpt-4o-mini23azure"), config.bi.scan.max_batches + 5
    )
    assert changepoint_scan(results) is None


def test_stable_endpoint_no_event():
    results = first_n_batches(load_fixture("openai2fgpt-4o-mini23azure"), 20)
    assert changepoint_scan(results) is None


def test_unstable_endpoint_no_event():
    results = first_n_batches(
        load_fixture("qwen2fqwen3-235b-a22b-250723wandb2fbf16"), 20
    )
    assert changepoint_scan(results) is None


def test_detects_early_step():
    # hy3 @ atlas-cloud: upstream model swap at batch 3 (2026-07-20, TV -> 1.0),
    # invisible to the adaptive rule (too early, then absorbed into its baseline)
    results = load_fixture("tencent2fhy323atlas-cloud2ffp8")
    event = changepoint_scan(results)
    assert event is not None
    assert event.split_ts[:10] == "2026-07-20"
    assert event.p_value <= config.bi.scan.alpha


def test_change_before_first_monitoring_batch_is_out_of_reach():
    # glm-5.2 @ deepinfra changed between the reference burst and the first
    # daily batch, so the reference block is the lone outlier. Any batch
    # ordering that puts that block at either end yields the identical
    # partition and statistic, capping the permutation p-value at ~2/n
    # (0.147 here) — an exchangeability test cannot certify this case with a
    # single-burst reference. Documented limit, not a bug: it needs
    # multi-day references (or a reference-outlier flag) to become detectable.
    results = load_fixture("z-ai2fglm-5.223deepinfra2ffp4")
    assert changepoint_scan(results) is None
