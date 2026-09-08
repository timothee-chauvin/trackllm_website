import json
from datetime import datetime, timezone
from pathlib import Path

import orjson

from trackllm_website.generate_site.lt import EndpointInfo, LTData, PromptInfo
from trackllm_website.generate_site.lt_raw import (
    DAILY_FILENAME,
    TOP_TOKENS,
    build_daily,
    write_lt_daily,
)
from trackllm_website.lt_drift import compute_drift_series
from trackllm_website.lt_scores import N_PER_TEST, load_prompt_logprobs

TOKENS = [f"t{i}" for i in range(12)]


def _write_prompt(endpoint_dir: Path, name: str, text: str, days: int, per_day: int):
    """`days` days of `per_day` queries in 2026-01, each returning all of TOKENS
    with logprobs that fall with the token index and drift down over the days."""
    d = endpoint_dir / name
    d.mkdir(parents=True)
    (d / "info.json").write_text(
        json.dumps({"prompt": text, "endpoint": {"model": "m/a", "provider": "p"}})
    )
    md = d / "2026-01"
    md.mkdir()
    vectors, queries = [], []
    for day in range(days):
        for q in range(per_day):
            vectors.append(
                {
                    "tokens": list(range(len(TOKENS))),
                    "logprobs": [
                        -0.1 * (i + 1) - 0.01 * day for i in range(len(TOKENS))
                    ],
                }
            )
            queries.append([f"{day + 1:02d} {q:02d}:00:00", len(vectors) - 1])
    (md / "logprobs.json").write_text(
        json.dumps({"seen_tokens": TOKENS, "seen_logprobs": vectors})
    )
    (md / "errors.json").write_text(json.dumps({"seen_errors": []}))
    (md / "queries.json").write_text(json.dumps(queries))
    return d


def test_days_are_the_drift_series_days_and_tokens_the_top_by_mean(tmp_path):
    ep = tmp_path / "m2fa23p"
    _write_prompt(ep, "hi_1", "Hi", days=4, per_day=N_PER_TEST)
    daily = build_daily(ep, {"hi_1": "Hi"}, None)
    assert daily is not None and daily["drift_prompt"] == 0
    (p,) = daily["prompts"]
    assert p["text"] == "Hi"
    drift = compute_drift_series(load_prompt_logprobs(ep / "hi_1"), None)
    assert [d[0] for d in p["days"]] == [dt.date().isoformat() for dt, _ in drift]
    assert [d[1] for d in p["days"]] == [N_PER_TEST] * 4
    day0 = p["days"][0][2]
    assert len(day0) == TOP_TOKENS
    assert [p["tokens"][i] for i, _ in day0] == TOKENS[:TOP_TOKENS]
    assert [v for _, v in day0] == [round(-0.1 * (i + 1), 3) for i in range(TOP_TOKENS)]
    assert len(p["ref"]) == len(p["tokens"])
    # the reference is the mean of the whole (14-day-short) series: days 0..3
    assert p["ref"][0] == round(-0.1 - 0.015, 3)
    # the lowest logprob returned: the last token on the last day
    assert p["floor"] == round(-0.1 * len(TOKENS) - 0.03, 3)


def test_drift_prompt_is_the_longest_and_short_prompts_are_left_out(tmp_path):
    ep = tmp_path / "m2fa23p"
    _write_prompt(ep, "a_1", "a", days=3, per_day=N_PER_TEST)
    _write_prompt(ep, "b_2", "b", days=5, per_day=N_PER_TEST)
    _write_prompt(ep, "c_3", "c", days=1, per_day=N_PER_TEST)  # under 2*N_PER_TEST
    daily = build_daily(ep, {"a_1": "a", "b_2": "b", "c_3": "c"}, None)
    assert [p["text"] for p in daily["prompts"]] == ["a", "b"]
    assert daily["drift_prompt"] == 1


def test_no_prompt_with_enough_data_means_no_file(tmp_path):
    ep = tmp_path / "m2fa23p"
    _write_prompt(ep, "a_1", "a", days=1, per_day=N_PER_TEST)
    assert build_daily(ep, {"a_1": "a"}, None) is None


def test_reference_follows_the_first_change(tmp_path):
    """Anchored on the earliest changepoint like the drift lane, not on the first
    days: otherwise the two columns of the readout would not be the drift."""
    ep = tmp_path / "m2fa23p"
    _write_prompt(ep, "a_1", "a", days=6, per_day=N_PER_TEST)
    first_change = datetime(2026, 1, 3, tzinfo=timezone.utc)
    daily = build_daily(ep, {"a_1": "a"}, first_change)
    # reference = days 1 and 2 (indices 0 and 1): mean offset 0.005
    assert daily["prompts"][0]["ref"][0] == round(-0.1 - 0.005, 3)


def test_write_lt_daily_writes_and_prunes(tmp_path):
    ep = tmp_path / "m2fa23p"
    _write_prompt(ep, "a_1", "a", days=3, per_day=N_PER_TEST)
    stale = tmp_path / "gone" / DAILY_FILENAME
    stale.parent.mkdir()
    stale.write_text("{}")
    info = EndpointInfo(
        model="m/a",
        provider="p",
        slug="m2fa23p",
        prompts=[PromptInfo(slug="a_1", prompt="a", months=["2026-01"])],
    )

    lt = LTData(dates=[], scores=[], n_per_test=N_PER_TEST, changes=[], drift=[])
    write_lt_daily(tmp_path, [info], {"m2fa23p": lt})
    assert not stale.exists()
    out = orjson.loads((ep / DAILY_FILENAME).read_bytes())
    assert out["prompts"][0]["text"] == "a"
