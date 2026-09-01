from pathlib import Path

import orjson
import pytest

from trackllm_website.bi.detection import (
    adaptive_transitions,
    epoch_tv_series,
    is_unstable,
    select_top_bis,
    tv_shift,
)
from trackllm_website.config import DetectionConfig, config

FIXTURES = Path("tests/fixtures/phase_2")


def load_fixture(slug: str) -> dict:
    return orjson.loads((FIXTURES / slug / "data.json").read_bytes())


def reference_from_first_batch(results: dict) -> dict[str, list]:
    ref_ts = min(ts for batches in results.values() for ts in batches)
    return {p: b[ref_ts] for p, b in results.items() if ref_ts in b and b[ref_ts]}


def series(slug: str):
    results = load_fixture(slug)
    return epoch_tv_series(reference_from_first_batch(results), results)


def test_detects_hyperbolic_deepseek_change():
    tv = series("deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8")
    events = adaptive_transitions(tv)
    assert [e[:10] for e in events] == ["2026-01-24"]


def test_stable_endpoint_no_events():
    tv = series("openai2fgpt-4o-mini23azure")
    assert adaptive_transitions(tv) == []
    assert not is_unstable(load_fixture("openai2fgpt-4o-mini23azure"))


def test_unstable_endpoint_flagged_not_fired():
    tv = series("z-ai2fglm-5.223siliconflow2ffp8")
    assert adaptive_transitions(tv) == []
    assert is_unstable(load_fixture("z-ai2fglm-5.223siliconflow2ffp8"))


def test_wandb_qwen3_is_stale_reference_not_unstable():
    # Long treated as the canonical unstable endpoint (TV ~0.46 vs reference),
    # but its days agree with each other to 0.02-0.19: the distribution moved
    # right after the reference burst and the borders collapsed. Changed, not
    # unstable — and still must not fire the adaptive rule.
    tv = series("qwen2fqwen3-235b-a22b-250723wandb2fbf16")
    assert adaptive_transitions(tv) == []
    assert not is_unstable(load_fixture("qwen2fqwen3-235b-a22b-250723wandb2fbf16"))


def test_changed_endpoint_with_stale_reference_is_not_unstable():
    # Both sit far from their pre-change reference (hy3: TV 1.0) but their days
    # agree with each other — changed, not unstable. The badge measures the
    # endpoint's own day-to-day dispersion, not distance to the reference.
    assert not is_unstable(load_fixture("tencent2fhy323atlas-cloud2ffp8"))
    assert not is_unstable(load_fixture("z-ai2fglm-5.223deepinfra2ffp4"))


SYNTHETIC_DETECTION = DetectionConfig(
    window=5,
    exclusion=2,
    min_baseline=2,
    sigma_k=2.0,
    abs_delta=0.1,
    persistence=2,
    cooldown=10,
    instability_window=14,
    instability_threshold=0.35,
)


@pytest.fixture
def synthetic_detection(monkeypatch):
    monkeypatch.setattr(config.bi, "detection", SYNTHETIC_DETECTION)
    return SYNTHETIC_DETECTION


def tv_series(vals: list[float]) -> list[tuple[str, float]]:
    return [(f"2026-01-{i + 1:02d}T00:00:00", v) for i, v in enumerate(vals)]


def test_early_days_never_compared_to_future(synthetic_detection):
    # Two spikes at the start, flat afterwards. Days 0-1 have no trailing
    # baseline (i < exclusion + min_baseline): they must be skipped, not
    # evaluated against a negative-end slice vals[0:i-exclusion] that wraps
    # around to include the (flat) future and makes the spikes "deviate".
    vals = [0.6, 0.6] + [0.0] * 20
    assert adaptive_transitions(tv_series(vals)) == []


def test_streak_resets_across_skipped_days(synthetic_detection, monkeypatch):
    monkeypatch.setattr(
        config.bi,
        "detection",
        SYNTHETIC_DETECTION.model_copy(update={"exclusion": 3}),
    )
    # Spike at day 2 (only "deviating" via the wrapped negative-end slice) and
    # at day 5 (first legitimately evaluated day). Days 3-4 in between are
    # skipped for insufficient baseline; a streak carried across them would
    # reach persistence=2 and date an event at never-evaluated day 4.
    vals = [0.0, 0.0, 0.6, 0.0, 0.0, 0.6] + [0.0] * 16
    assert adaptive_transitions(tv_series(vals)) == []


def test_level_shift_fires_at_onset(synthetic_detection):
    vals = [0.05] * 8 + [0.5] * 4
    events = adaptive_transitions(tv_series(vals))
    assert events == ["2026-01-09T00:00:00"]


def test_cooldown_suppresses_nearby_second_event(synthetic_detection, monkeypatch):
    vals = [0.05] * 8 + [0.5, 0.5, 0.05, 0.05, 1.0, 1.0]
    # Second onset (day 12) is 4 days after the first (day 8).
    assert adaptive_transitions(tv_series(vals)) == ["2026-01-09T00:00:00"]
    monkeypatch.setattr(
        config.bi,
        "detection",
        SYNTHETIC_DETECTION.model_copy(update={"cooldown": 1}),
    )
    assert adaptive_transitions(tv_series(vals)) == [
        "2026-01-09T00:00:00",
        "2026-01-13T00:00:00",
    ]


def test_select_top_bis_by_balance():
    reference = {
        "balanced": [["t", "a"], ["t", "b"], ["t", "a"], ["t", "b"]],
        "skewed": [["t", "a"], ["t", "a"], ["t", "a"], ["t", "b"]],
        "dirac": [["t", "a"], ["t", "a"], ["t", "a"], ["t", "a"]],
    }
    assert select_top_bis(reference, 2) == ["balanced", "skewed"]


def test_tv_shift_is_mean_difference_across_split():
    tv = [("d1", 0.1), ("d2", 0.3), ("d3", 0.5), ("d4", 0.7)]
    assert tv_shift(tv, "d3") == pytest.approx(0.4)
    assert tv_shift(tv, "d1") is None
    assert tv_shift(tv, "d9") is None
