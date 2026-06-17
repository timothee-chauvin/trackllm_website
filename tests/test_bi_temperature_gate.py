import asyncio

from trackllm_website.bi.phase_1 import (
    check_temperature,
    temperature_is_ignored,
)
from trackllm_website.config import Endpoint

ENDPOINT = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))


def test_ignored_when_t0_matches_t1():
    # per-prompt distinct-output counts at T=0 and T=1 are identical and both diverse
    t0 = {"p1": 3, "p2": 4, "p3": 2}
    t1 = {"p1": 3, "p2": 4, "p3": 2}
    assert temperature_is_ignored(t0, t1) is True


def test_honored_when_t1_broadens():
    t0 = {"p1": 1, "p2": 1, "p3": 2}
    t1 = {"p1": 3, "p2": 4, "p3": 5}
    assert temperature_is_ignored(t0, t1) is False


def _fake_sample_prompts(by_temp):
    """by_temp: {temperature: {prompt: [tokens]}} -> a sample_prompts stand-in."""

    async def sample(client, endpoint, strategy, prompts, n_per_prompt, temperature):
        dist = by_temp[temperature]
        return {p: [("ts", tok) for tok in dist[p]] for p in prompts}, 0

    return sample


def test_check_temperature_true_when_ignored(monkeypatch):
    from trackllm_website.bi import phase_1 as phase_1_mod

    # T=1 produces the exact same distinct tokens as T=0 -> ignored
    by_temp = {
        0.0: {"p1": ["a", "b"], "p2": ["c", "d"]},
        1.0: {"p1": ["a", "b"], "p2": ["c", "d"]},
    }
    monkeypatch.setattr(phase_1_mod, "sample_prompts", _fake_sample_prompts(by_temp))
    result = asyncio.run(check_temperature(None, ENDPOINT, None, ["p1", "p2"]))
    assert result is True


def test_check_temperature_false_when_honored(monkeypatch):
    from trackllm_website.bi import phase_1 as phase_1_mod

    # T=1 broadens p1 (3 distinct vs 1) -> honored
    by_temp = {
        0.0: {"p1": ["a"], "p2": ["c", "d"]},
        1.0: {"p1": ["a", "b", "c"], "p2": ["c", "d"]},
    }
    monkeypatch.setattr(phase_1_mod, "sample_prompts", _fake_sample_prompts(by_temp))
    result = asyncio.run(check_temperature(None, ENDPOINT, None, ["p1", "p2"]))
    assert result is False
