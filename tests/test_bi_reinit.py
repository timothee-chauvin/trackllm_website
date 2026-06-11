import asyncio
from datetime import datetime, timezone

from trackllm_website.bi import reinit as reinit_mod
from trackllm_website.config import Endpoint

ENDPOINT = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))
NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)


def fake_sampler(distributions):
    """distributions: prompt -> list of tokens to cycle through."""

    async def sample(client, endpoint, strategy, prompts, n):
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


def test_reinit_keeps_survivors_and_ranks(monkeypatch):
    # old BI "dead" collapsed to one token; "alive" still has two
    monkeypatch.setattr(
        reinit_mod,
        "sample_prompts",
        fake_sampler({"alive": ["a", "b"], "dead": ["a"], "new1": ["x", "y"]}),
    )

    async def fake_discover(endpoint, exclude):
        return ["new1"]

    monkeypatch.setattr(reinit_mod, "discover_candidates", fake_discover)
    monkeypatch.setattr(reinit_mod.config.bi.reinit, "top_k_bis", 2)
    monkeypatch.setattr(reinit_mod.config.bi.reinit, "min_bis", 2)

    epoch = asyncio.run(
        reinit_mod.reinit(None, None, ENDPOINT, ["alive", "dead"], NOW)
    )
    assert epoch is not None
    assert sorted(epoch.border_inputs) == ["alive", "new1"]
    assert set(epoch.reference) == {"alive", "new1"}


def test_reinit_returns_none_below_min_bis(monkeypatch):
    monkeypatch.setattr(reinit_mod, "sample_prompts", fake_sampler({"dead": ["a"]}))

    async def fake_discover(endpoint, exclude):
        return []

    monkeypatch.setattr(reinit_mod, "discover_candidates", fake_discover)
    assert asyncio.run(reinit_mod.reinit(None, None, ENDPOINT, ["dead"], NOW)) is None
