import asyncio

from trackllm_website.bi.phase_1 import (
    check_temperature,
    temperature_is_ignored,
)
from trackllm_website.config import Endpoint, config

ENDPOINT = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))


def _n_queries(prompts):
    return len(prompts) * config.bi.temperature_gate.check_samples


def test_ignored_when_t0_matches_t1():
    # per-prompt distinct-output counts at T=0 and T=1 are identical and both diverse
    t0 = {"p1": 3, "p2": 4, "p3": 2}
    t1 = {"p1": 3, "p2": 4, "p3": 2}
    assert temperature_is_ignored(t0, t1) is True


def test_honored_when_t1_broadens():
    t0 = {"p1": 1, "p2": 1, "p3": 2}
    t1 = {"p1": 3, "p2": 4, "p3": 5}
    assert temperature_is_ignored(t0, t1) is False


def _cycle(tokens, n):
    """n samples cycling through `tokens`; an empty token list means no output at all
    (errored or empty responses), which is what the real sampler returns for those."""
    return [tokens[i % len(tokens)] for i in range(n)] if tokens else []


def _fake_sample_prompts(by_temp, errors_by_temp):
    """by_temp: {temperature: {prompt: [tokens]}} -> a sample_prompts stand-in.

    The token lists are the samples actually returned (build them with _cycle);
    errors_by_temp: {temperature: n_errors}, as returned by the real sampler.
    """

    async def sample(client, endpoint, strategy, prompts, n_per_prompt, temperature):
        dist = by_temp[temperature]
        samples = {p: [("ts", tok) for tok in dist[p]] for p in prompts}
        return samples, errors_by_temp[temperature]

    return sample


def _patch_sampler(monkeypatch, by_temp, errors_by_temp):
    from trackllm_website.bi import phase_1 as phase_1_mod

    monkeypatch.setattr(
        phase_1_mod, "sample_prompts", _fake_sample_prompts(by_temp, errors_by_temp)
    )


SAMPLES = config.bi.temperature_gate.check_samples


def test_check_temperature_ignored(monkeypatch):
    # T=1 produces the exact same distinct tokens as T=0 -> ignored
    by_temp = {
        t: {"p1": _cycle(["a", "b"], SAMPLES), "p2": _cycle(["c", "d"], SAMPLES)}
        for t in (0.0, 1.0)
    }
    _patch_sampler(monkeypatch, by_temp, {0.0: 0, 1.0: 0})
    result = asyncio.run(check_temperature(None, ENDPOINT, None, ["p1", "p2"]))
    assert result == "ignored"


def test_check_temperature_honored(monkeypatch):
    # T=1 broadens p1 (3 distinct vs 1) -> honored
    by_temp = {
        0.0: {"p1": _cycle(["a"], SAMPLES), "p2": _cycle(["c", "d"], SAMPLES)},
        1.0: {
            "p1": _cycle(["a", "b", "c"], SAMPLES),
            "p2": _cycle(["c", "d"], SAMPLES),
        },
    }
    _patch_sampler(monkeypatch, by_temp, {0.0: 0, 1.0: 0})
    result = asyncio.run(check_temperature(None, ENDPOINT, None, ["p1", "p2"]))
    assert result == "honored"


def test_check_temperature_inconclusive_when_all_queries_error(monkeypatch):
    # A transient outage yields zero samples at both temperatures, which is
    # indistinguishable from "T=1 never broadens": the endpoint must not be
    # condemned to the bad_temperature cache on no data at all.
    prompts = ["p1", "p2"]
    by_temp = {t: {"p1": [], "p2": []} for t in (0.0, 1.0)}
    n = _n_queries(prompts)
    _patch_sampler(monkeypatch, by_temp, {0.0: n, 1.0: n})
    result = asyncio.run(check_temperature(None, ENDPOINT, None, prompts))
    assert result == "inconclusive"


def test_check_temperature_inconclusive_when_every_response_is_empty(monkeypatch):
    # sample_prompts counts an empty (non-error) response as a success, so error
    # counts alone would call this endpoint's all-zero distinct counts "ignored".
    prompts = ["p1", "p2"]
    by_temp = {t: {"p1": [], "p2": []} for t in (0.0, 1.0)}
    _patch_sampler(monkeypatch, by_temp, {0.0: 0, 1.0: 0})
    result = asyncio.run(check_temperature(None, ENDPOINT, None, prompts))
    assert result == "inconclusive"


def test_check_temperature_inconclusive_when_t1_mostly_errors(monkeypatch):
    # T=1 looks narrower than T=0, but only because most of its queries failed.
    prompts = ["p1", "p2"]
    n = _n_queries(prompts)
    by_temp = {
        0.0: {"p1": _cycle(["a", "b"], SAMPLES), "p2": _cycle(["c", "d"], SAMPLES)},
        1.0: {"p1": _cycle(["a"], 1), "p2": _cycle(["c"], 1)},
    }
    _patch_sampler(monkeypatch, by_temp, {0.0: 0, 1.0: n - 2})
    result = asyncio.run(check_temperature(None, ENDPOINT, None, prompts))
    assert result == "inconclusive"


def test_check_temperature_tolerates_a_minority_of_errors(monkeypatch):
    # Enough successful queries remain for the comparison to mean something.
    prompts = ["p1", "p2"]
    kept = SAMPLES - SAMPLES // 4
    by_temp = {
        0.0: {"p1": _cycle(["a", "b"], SAMPLES), "p2": _cycle(["c", "d"], SAMPLES)},
        1.0: {"p1": _cycle(["a", "b"], kept), "p2": _cycle(["c", "d"], kept)},
    }
    _patch_sampler(monkeypatch, by_temp, {0.0: 0, 1.0: 2 * (SAMPLES - kept)})
    result = asyncio.run(check_temperature(None, ENDPOINT, None, prompts))
    assert result == "ignored"


def test_check_temperature_inconclusive_without_prompts(monkeypatch):
    # No prompts means no evidence; the vacuous all() must not read as "ignored".
    _patch_sampler(monkeypatch, {0.0: {}, 1.0: {}}, {0.0: 0, 1.0: 0})
    result = asyncio.run(check_temperature(None, ENDPOINT, None, []))
    assert result == "inconclusive"
