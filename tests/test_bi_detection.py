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
    assert not is_unstable(tv)


def test_unstable_endpoint_flagged_not_fired():
    tv = series("qwen2fqwen3-235b-a22b-250723wandb2fbf16")
    assert adaptive_transitions(tv) == []
    assert is_unstable(tv)


def test_select_top_bis_by_balance():
    reference = {
        "balanced": [["t", "a"], ["t", "b"], ["t", "a"], ["t", "b"]],
        "skewed": [["t", "a"], ["t", "a"], ["t", "a"], ["t", "b"]],
        "dirac": [["t", "a"], ["t", "a"], ["t", "a"], ["t", "a"]],
    }
    assert select_top_bis(reference, 2) == ["balanced", "skewed"]
