from pathlib import Path

import orjson

from trackllm_website.bi.detection import (
    adaptive_transitions,
    epoch_tv_series,
    is_unstable,
    select_top_bis,
)

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


def test_select_top_bis_by_balance():
    reference = {
        "balanced": [["t", "a"], ["t", "b"], ["t", "a"], ["t", "b"]],
        "skewed": [["t", "a"], ["t", "a"], ["t", "a"], ["t", "b"]],
        "dirac": [["t", "a"], ["t", "a"], ["t", "a"], ["t", "a"]],
    }
    assert select_top_bis(reference, 2) == ["balanced", "skewed"]
