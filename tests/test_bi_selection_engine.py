import logging

import pytest

from trackllm_website.bi.selection import (
    Rule,
    SelectionPolicy,
    monthly_cost,
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


# samples_per_month default 6000 in config; monthly = cpr * 6000
def test_monthly_cost():
    assert abs(monthly_cost(ep("m", "p", 0.0001)) - 0.6) < 1e-9


def test_flagship_selected_over_budget_and_exempt_from_ceiling():
    policy = SelectionPolicy(
        budget_per_month=1.0,
        max_endpoint_cost=0.5,
        exclude=[],
        rules=[
            Rule(
                name="flagships",
                kind="models",
                patterns=["openai/gpt-5"],
                providers_per_model=1,
                flagship=True,
            )
        ],
    )
    # gpt-5 monthly cost 3.0 > budget AND > ceiling, but flagship => selected
    cands = [ep("openai/gpt-5", "openai", 0.0005)]  # 0.0005*6000 = 3.0
    selected, breakdown = select_monitoring_targets(cands, policy)
    assert cands[0] in selected
    assert breakdown[cands[0]] == "flagships"


def test_cheapest_provider_per_flagship_model():
    policy = SelectionPolicy(
        budget_per_month=100.0,
        max_endpoint_cost=10.0,
        exclude=[],
        rules=[
            Rule(
                name="flagships",
                kind="models",
                patterns=["m/a"],
                providers_per_model=1,
                flagship=True,
            )
        ],
    )
    cands = [ep("m/a", "cheap", 0.00001), ep("m/a", "pricey", 0.0001)]
    selected, _ = select_monitoring_targets(cands, policy)
    assert [e.provider for e in selected] == ["cheap"]


def test_exclude_globs_win():
    policy = SelectionPolicy(
        budget_per_month=100.0,
        max_endpoint_cost=10.0,
        exclude=["*image*"],
        rules=[
            Rule(
                name="long-tail",
                kind="models",
                patterns=["*"],
                providers_per_model=1,
                max_monthly_cost=10.0,
            )
        ],
    )
    cands = [ep("openai/gpt-image", "openai", 0.00001), ep("m/b", "p", 0.00001)]
    selected, _ = select_monitoring_targets(cands, policy)
    assert [e.model for e in selected] == ["m/b"]


def test_max_monthly_cost_skips_pricey_in_wildcard_rule():
    policy = SelectionPolicy(
        budget_per_month=100.0,
        max_endpoint_cost=10.0,
        exclude=[],
        rules=[
            Rule(
                name="long-tail",
                kind="models",
                patterns=["*"],
                providers_per_model=1,
                max_monthly_cost=0.10,
            )
        ],  # 0.10/mo => cpr<=~1.67e-5
    )
    cands = [ep("m/cheap", "p", 0.00001), ep("m/pricey", "p", 0.00005)]
    selected, _ = select_monitoring_targets(cands, policy)
    assert [e.model for e in selected] == ["m/cheap"]


def test_budget_stops_wildcard_fill():
    policy = SelectionPolicy(
        budget_per_month=0.6,
        max_endpoint_cost=10.0,
        exclude=[],
        rules=[
            Rule(
                name="long-tail",
                kind="models",
                patterns=["*"],
                providers_per_model=1,
                max_monthly_cost=10.0,
            )
        ],
    )
    # each endpoint is 0.6/mo; budget 0.6 fits exactly one
    cands = [ep("m/a", "p", 0.0001), ep("m/b", "p", 0.0001)]
    selected, _ = select_monitoring_targets(cands, policy)
    assert len(selected) == 1


def test_flagship_over_budget_is_allowed_with_warning(caplog):
    # two flagships, each 6.0/mo, budget 10 => flagships are budget-exempt:
    # both selected, a warning is logged, NOT an error (per the design).
    policy = SelectionPolicy(
        budget_per_month=10.0,
        max_endpoint_cost=100.0,
        exclude=[],
        rules=[
            Rule(
                name="flagships",
                kind="models",
                patterns=["m/a", "m/b"],
                providers_per_model=1,
                flagship=True,
            )
        ],
    )
    cands = [ep("m/a", "p", 0.001), ep("m/b", "p", 0.001)]  # 6.0/mo each
    with caplog.at_level(logging.WARNING):
        selected, _ = select_monitoring_targets(cands, policy)
    assert set(cands) <= set(selected)
    assert caplog.records


def test_nonflagship_named_rule_over_budget_raises():
    # a NON-flagship named (non-wildcard) rule whose endpoints total > budget => loud error.
    policy = SelectionPolicy(
        budget_per_month=10.0,
        max_endpoint_cost=100.0,
        exclude=[],
        rules=[
            Rule(
                name="named",
                kind="models",
                patterns=["m/a", "m/b"],
                providers_per_model=1,
            )
        ],
    )
    cands = [ep("m/a", "p", 0.001), ep("m/b", "p", 0.001)]  # 6.0/mo each => 12.0 total
    with pytest.raises(ValueError, match="exceeds budget"):
        select_monitoring_targets(cands, policy)


def test_nonflagship_overshoot_then_wildcard_still_raises():
    # a non-flagship named rule overshoots budget, then a trailing wildcard fill rule.
    # the wildcard must NOT swallow the over-budget error via an early return.
    policy = SelectionPolicy(
        budget_per_month=10.0,
        max_endpoint_cost=100.0,
        exclude=[],
        rules=[
            Rule(
                name="named",
                kind="models",
                patterns=["m/a", "m/b"],
                providers_per_model=1,
            ),
            Rule(
                name="long-tail",
                kind="models",
                patterns=["*"],
                providers_per_model=1,
                max_monthly_cost=10.0,
            ),
        ],
    )
    # m/a, m/b: 6.0/mo each => named rule selects both (12.0 > budget). m/c is an
    # unselected wildcard candidate: with the buggy in-loop return, the wildcard
    # early-returns on m/c and the post-loop ValueError never fires.
    cands = [
        ep("m/a", "p", 0.001),
        ep("m/b", "p", 0.001),
        ep("m/c", "p", 0.001),
    ]
    with pytest.raises(ValueError, match="exceeds budget"):
        select_monitoring_targets(cands, policy)


def test_providers_branch_covers_and_skips_pricey():
    policy = SelectionPolicy(
        budget_per_month=100.0,
        max_endpoint_cost=100.0,
        exclude=[],
        rules=[
            Rule(
                name="providers",
                kind="providers",
                patterns=["*"],
                endpoints_per_provider=1,
                max_monthly_cost=0.25,
            )
        ],
    )
    # provA cheapest 0.12/mo (under cap), provB cheapest 0.6/mo (over cap)
    cands = [
        ep("m/a", "provA", 0.00002),  # 0.12/mo
        ep("m/b", "provA", 0.0001),  # 0.6/mo
        ep("m/c", "provB", 0.0001),  # 0.6/mo
    ]
    selected, _ = select_monitoring_targets(cands, policy)
    assert [e.provider for e in selected] == ["provA"]


def test_selection_is_deterministic():
    policy = SelectionPolicy(
        budget_per_month=100.0,
        max_endpoint_cost=100.0,
        exclude=[],
        rules=[
            Rule(
                name="long-tail",
                kind="models",
                patterns=["*"],
                providers_per_model="all",
                max_monthly_cost=100.0,
            )
        ],
    )
    # equal cost_per_request across several providers/models => order must be stable
    cands = [
        ep("m/a", "p2", 0.0001),
        ep("m/a", "p1", 0.0001),
        ep("m/b", "p1", 0.0001),
        ep("m/c", "p3", 0.0001),
    ]
    selected1, labels1 = select_monitoring_targets(list(cands), policy)
    selected2, labels2 = select_monitoring_targets(list(cands), policy)
    assert selected1 == selected2
    assert labels1 == labels2
