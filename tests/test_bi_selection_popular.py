from trackllm_website.bi.selection import (
    Rule,
    SelectionPolicy,
    select_monitoring_targets,
)
from trackllm_website.config import Endpoint


def ep(model, provider, cpr):
    return Endpoint(
        api="openrouter",
        model=model,
        provider=provider,
        cost=(1, 1),
        cost_per_request=cpr,
    )


def test_popular_selects_top_n_present_in_candidates():
    pol = SelectionPolicy(
        budget_per_month=100,
        max_endpoint_cost=100,
        exclude=[],
        rules=[
            Rule(name="popular", kind="popular", patterns=[], providers_per_model=1)
        ],
    )
    cands = [
        ep("m/a", "cheap", 0.00001),
        ep("m/a", "pricey", 0.0001),
        ep("m/b", "p", 0.00001),
    ]
    # popularity order: m/b most popular, m/a next, m/c absent from candidates
    sel, breakdown = select_monitoring_targets(cands, pol, ["m/b", "m/a", "m/c"])
    assert set(e.model for e in sel) == {"m/a", "m/b"}
    assert breakdown[next(e for e in sel if e.model == "m/a")] == "popular"
    assert [e.provider for e in sel if e.model == "m/a"] == [
        "cheap"
    ]  # cheapest provider


def test_popular_respects_max_monthly_cost():
    pol = SelectionPolicy(
        budget_per_month=100,
        max_endpoint_cost=100,
        exclude=[],
        rules=[
            Rule(
                name="popular",
                kind="popular",
                patterns=[],
                providers_per_model=1,
                max_monthly_cost=0.10,
            )
        ],
    )  # 0.10/mo => cpr <= ~1.67e-5
    cands = [ep("m/cheap", "p", 0.00001), ep("m/pricey", "p", 0.00005)]
    sel, _ = select_monitoring_targets(cands, pol, ["m/pricey", "m/cheap"])
    assert [e.model for e in sel] == ["m/cheap"]


def test_popular_stops_at_budget_without_raising():
    # popular is a fill rule: 3 popular models at 0.6/mo each, budget 0.6 => take 1,
    # in popularity order, and do NOT raise the non-flagship-over-budget ValueError.
    pol = SelectionPolicy(
        budget_per_month=0.6,
        max_endpoint_cost=100,
        exclude=[],
        rules=[
            Rule(name="popular", kind="popular", patterns=[], providers_per_model=1)
        ],
    )
    cands = [ep("m/a", "p", 0.0001), ep("m/b", "p", 0.0001), ep("m/c", "p", 0.0001)]
    sel, breakdown = select_monitoring_targets(cands, pol, ["m/b", "m/a", "m/c"])
    assert [e.model for e in sel] == ["m/b"]  # most popular that fits
    assert all(lbl == "popular" for lbl in breakdown.values())
