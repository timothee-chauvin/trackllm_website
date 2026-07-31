import asyncio
from pathlib import Path

from trackllm_website.bi.costs import (
    build_cost_summary,
    costs_path,
    ensure_costs,
    format_preview,
)
from trackllm_website.bi.selection import Rule, SelectionPolicy
from trackllm_website.config import Endpoint, config


def ep(m, p, cpr=None):
    return Endpoint(
        api="openrouter", model=m, provider=p, cost=(1, 1), cost_per_request=cpr
    )


def test_summary_run_rate_and_top():
    policy = SelectionPolicy(
        budget_per_month=10,
        max_endpoint_cost=10,
        exclude=[],
        rules=[
            Rule(
                name="long-tail",
                kind="models",
                patterns=["*"],
                providers_per_model=1,
                max_monthly_cost=10,
            )
        ],
    )
    cands = [ep("m/a", "p", 0.00001), ep("m/b", "p", 0.00005)]
    summary = build_cost_summary(cands, policy, [])
    assert summary["budget_per_month"] == 10
    assert abs(summary["run_rate_per_month"] - (0.00001 + 0.00005) * 6000) < 1e-6
    # top endpoints sorted by monthly cost descending
    assert summary["endpoints"][0]["model"] == "m/b"
    assert summary["by_rule"]["long-tail"]["count"] == 2


def a_policy():
    return SelectionPolicy(
        budget_per_month=10,
        max_endpoint_cost=0.5,
        exclude=[],
        rules=[
            Rule(name="long-tail", kind="models", patterns=["*"], providers_per_model=1)
        ],
    )


def test_ensure_costs_only_probes_missing(monkeypatch):
    probed = []

    async def fake_vet(client, endpoint, strategy):
        probed.append(str(endpoint))
        from trackllm_website.bi.vetting import VetResult

        return VetResult(bucket="candidate", cost_per_request=0.00002)

    async def fake_resolve(client, eps, policy):
        from trackllm_website.bi.common import PlainStrategy

        return {str(e): PlainStrategy() for e in eps}, {}

    monkeypatch.setattr("trackllm_website.bi.costs.vet_endpoint", fake_vet)
    monkeypatch.setattr("trackllm_website.bi.costs.resolve_strategies", fake_resolve)
    cands = [ep("m/a", "p", cpr=0.00001), ep("m/b", "p")]  # b is missing
    filled = asyncio.run(ensure_costs(cands, a_policy(), save=False))
    assert probed == ["openrouter#m/b#p"]  # only the missing one probed
    assert all(e.cost_per_request is not None for e in filled)


def test_ensure_costs_probes_under_the_cost_guard(monkeypatch):
    """Without the policy, discover_strategy's too-expensive short-circuit is off:
    budget escalation would run unguarded on arbitrary un-priced catalog endpoints."""
    seen = {}

    async def fake_resolve(client, eps, policy):
        seen["policy"] = policy
        return {}, {}

    monkeypatch.setattr("trackllm_website.bi.costs.resolve_strategies", fake_resolve)
    policy = a_policy()
    asyncio.run(ensure_costs([ep("m/b", "p")], policy, save=False))
    assert seen["policy"] is policy


def test_format_preview_groups_and_totals():
    policy = SelectionPolicy(
        budget_per_month=10,
        max_endpoint_cost=10,
        exclude=[],
        rules=[
            Rule(
                name="long-tail",
                kind="models",
                patterns=["*"],
                providers_per_model=1,
                max_monthly_cost=10,
            )
        ],
    )
    cands = [ep("m/a", "p", 0.00001), ep("m/b", "p", 0.00005)]
    text = format_preview(build_cost_summary(cands, policy, []))
    assert "m/b" in text and "m/a" in text
    assert "/mo" in text
    assert "10.00" in text  # budget shown


def test_costs_path_under_b3it():
    assert costs_path() == config.bi.data_dir / "bi_costs.json"
    assert costs_path() == Path("website/data/b3it/bi_costs.json")
