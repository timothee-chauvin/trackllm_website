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
        api="openrouter", model=model, provider=provider, cost=(1, 1), cost_per_request=cpr
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
