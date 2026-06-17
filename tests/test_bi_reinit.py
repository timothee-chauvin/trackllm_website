import asyncio
from datetime import datetime, timezone
from pathlib import Path

import orjson

from trackllm_website.bi import reinit as reinit_mod
from trackllm_website.bi.reinit import parse_phase_1_results
from trackllm_website.config import Endpoint

ENDPOINT = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))
NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)


def fake_sampler(distributions):
    """distributions: prompt -> list of tokens to cycle through."""

    async def sample(client, endpoint, strategy, prompts, n, temperature):
        return (
            {
                p: [
                    (NOW.isoformat(), distributions[p][i % len(distributions[p])])
                    for i in range(n)
                ]
                for p in prompts
            },
            0,
        )

    return sample


def test_reinit_keeps_survivors_and_ranks(monkeypatch, tmp_path):
    # old BI "dead" collapsed to one token; "alive" still has two
    monkeypatch.setattr(
        reinit_mod,
        "sample_prompts",
        fake_sampler({"alive": ["a", "b"], "dead": ["a"], "new1": ["x", "y"]}),
    )

    async def fake_discover(endpoint, exclude):
        return ["new1"], 0.0

    monkeypatch.setattr(reinit_mod, "discover_candidates", fake_discover)
    monkeypatch.setattr(reinit_mod.config.bi.reinit, "top_k_bis", 2)
    monkeypatch.setattr(reinit_mod.config.bi.reinit, "min_bis", 2)
    phase2_path = tmp_path / "phase2.json"
    monkeypatch.setattr(reinit_mod, "get_output_path", lambda ep, ym: phase2_path)

    result = asyncio.run(
        reinit_mod.reinit(None, None, ENDPOINT, ["alive", "dead"], NOW)
    )
    epoch = result.epoch
    assert result.reason == "ok"
    assert epoch is not None
    assert sorted(epoch.border_inputs) == ["alive", "new1"]
    assert set(epoch.reference) == {"alive", "new1"}

    # Addendum 1: the reference batch is persisted to the phase-2 monthly file,
    # keyed by the microsecond-stripped epoch start, so detection won't skip the
    # first real day.
    persisted = orjson.loads(phase2_path.read_bytes())
    batch_key = NOW.replace(microsecond=0).isoformat()
    assert set(persisted) == {"alive", "new1"}
    assert all(batch_key in batches for batches in persisted.values())
    assert persisted["alive"][batch_key] == [list(s) for s in epoch.reference["alive"]]


def test_reinit_returns_none_below_min_bis(monkeypatch):
    monkeypatch.setattr(reinit_mod, "sample_prompts", fake_sampler({"dead": ["a"]}))

    async def fake_discover(endpoint, exclude):
        return [], 0.0

    monkeypatch.setattr(reinit_mod, "discover_candidates", fake_discover)
    result = asyncio.run(reinit_mod.reinit(None, None, ENDPOINT, ["dead"], NOW))
    assert result.epoch is None
    assert result.reason == "no_bis"


def test_reinit_bad_temperature_on_high_prevalence(monkeypatch):
    # onboarding (old_bis empty), discovery yields high prevalence, and the
    # temperature gate confirms T=0 is ignored -> bad_temperature, no epoch.
    async def fake_discover(endpoint, exclude):
        return ["p1", "p2", "p3"], 0.9  # above the 0.30 trigger

    async def fake_check(client, endpoint, strategy, prompts):
        return True

    monkeypatch.setattr(reinit_mod, "discover_candidates", fake_discover)
    monkeypatch.setattr(reinit_mod, "check_temperature", fake_check)

    result = asyncio.run(reinit_mod.reinit(None, None, ENDPOINT, [], NOW))
    assert result.epoch is None
    assert result.reason == "bad_temperature"


def test_reinit_no_gate_when_prevalence_low(monkeypatch, tmp_path):
    # Low prevalence -> gate never runs; check_temperature would raise if called.
    monkeypatch.setattr(reinit_mod, "sample_prompts", fake_sampler({"p1": ["a", "b"]}))
    monkeypatch.setattr(
        reinit_mod, "get_output_path", lambda ep, ym: tmp_path / "phase2.json"
    )

    async def fake_discover(endpoint, exclude):
        return ["p1"], 0.1  # below trigger

    async def boom(*a, **k):
        raise AssertionError("temperature gate should not run at low prevalence")

    monkeypatch.setattr(reinit_mod, "discover_candidates", fake_discover)
    monkeypatch.setattr(reinit_mod, "check_temperature", boom)
    monkeypatch.setattr(reinit_mod.config.bi.reinit, "top_k_bis", 1)
    monkeypatch.setattr(reinit_mod.config.bi.reinit, "min_bis", 1)

    result = asyncio.run(reinit_mod.reinit(None, None, ENDPOINT, [], NOW))
    assert result.reason == "ok"


def test_parse_phase_1_candidates_skips_meta_and_decoy(tmp_path: Path):
    (tmp_path / "endpoint.json").write_bytes(
        orjson.dumps(
            {
                "1": {"bi_prompt": ["a", "b", "a"], "boring": ["a", "a", "a"]},
                "_meta": {"1": {"bi_prompt": [[1, 1, 0]], "boring": [[1, 1, 0]]}},
            }
        )
    )
    (tmp_path / "border_inputs.json").write_bytes(orjson.dumps({"decoy": ["a", "b"]}))

    # Denominator counts distinct real prompts (bi_prompt, boring), not _meta or
    # token-count keys.
    assert parse_phase_1_results(tmp_path, []) == (["bi_prompt"], 2)
    assert parse_phase_1_results(tmp_path, ["bi_prompt"]) == ([], 2)
