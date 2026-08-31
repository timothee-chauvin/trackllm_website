"""Budget applies to every rule, flagships included; named-rule skips are loud."""

import logging

from trackllm_website.bi.digest import OUTCOME
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


def flagship_policy(budget, patterns):
    return SelectionPolicy(
        budget_per_month=budget,
        max_endpoint_cost=100.0,
        exclude=[],
        rules=[
            Rule(
                name="flagships",
                kind="models",
                patterns=patterns,
                providers_per_model=1,
                flagship=True,
            )
        ],
    )


def test_flagship_over_budget_is_skipped_and_reported(caplog):
    # each 6.0/mo, budget fits one: the first pattern wins, the second is skipped.
    policy = flagship_policy(10.0, ["m/a", "m/b"])
    cands = [ep("m/a", "p", 0.001), ep("m/b", "p", 0.001)]
    with caplog.at_level(logging.ERROR):
        selected, labels, skipped = select_monitoring_targets(cands, policy, [])
    assert selected == [cands[0]]
    assert skipped == [(cands[1], "flagships")]
    assert any("m/b" in r.message for r in caplog.records)


def test_pattern_list_order_is_priority_not_cost():
    # m/pricey is first in the list and fits; m/cheap would be picked under
    # cheapest-first but list order must win.
    policy = flagship_policy(6.0, ["m/pricey", "m/cheap"])
    cands = [ep("m/pricey", "p", 0.001), ep("m/cheap", "p", 0.0001)]
    selected, _, skipped = select_monitoring_targets(cands, policy, [])
    assert selected == [cands[0]]
    assert skipped == [(cands[1], "flagships")]


def test_named_rule_skip_continues_to_cheaper_pattern():
    # the pricey first pattern doesn't fit, the cheaper second one still does.
    policy = flagship_policy(1.0, ["m/pricey", "m/cheap"])
    cands = [ep("m/pricey", "p", 0.001), ep("m/cheap", "p", 0.0001)]
    selected, _, skipped = select_monitoring_targets(cands, policy, [])
    assert selected == [cands[1]]
    assert skipped == [(cands[0], "flagships")]


def test_nonflagship_named_rule_over_budget_skips_not_raises():
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
    cands = [ep("m/a", "p", 0.001), ep("m/b", "p", 0.001)]
    selected, _, skipped = select_monitoring_targets(cands, policy, [])
    assert selected == [cands[0]]
    assert skipped == [(cands[1], "named")]


def test_wildcard_fill_stops_silently():
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
    cands = [ep("m/a", "p", 0.0001), ep("m/b", "p", 0.0001)]
    selected, _, skipped = select_monitoring_targets(cands, policy, [])
    assert len(selected) == 1
    assert skipped == []


def test_popular_flagship_rule_no_longer_budget_exempt():
    policy = SelectionPolicy(
        budget_per_month=0.6,
        max_endpoint_cost=100.0,
        exclude=[],
        rules=[
            Rule(
                name="popular",
                kind="popular",
                patterns=[],
                providers_per_model="all",
                flagship=True,
            )
        ],
    )
    cands = [ep("m/a", "p", 0.0001), ep("m/b", "p", 0.0001)]
    selected, _, skipped = select_monitoring_targets(cands, policy, ["m/a", "m/b"])
    assert len(selected) == 1
    assert skipped == []


def test_nonwildcard_providers_rule_respects_budget():
    policy = SelectionPolicy(
        budget_per_month=0.6,
        max_endpoint_cost=100.0,
        exclude=[],
        rules=[
            Rule(
                name="one-provider",
                kind="providers",
                patterns=["provA"],
                endpoints_per_provider=2,
            )
        ],
    )
    cands = [ep("m/a", "provA", 0.0001), ep("m/b", "provA", 0.0001)]
    selected, _, skipped = select_monitoring_targets(cands, policy, [])
    assert len(selected) == 1


def test_not_selected_budget_outcome_is_red():
    assert OUTCOME["not_selected_budget"][1] == "#cf222e"


def test_cost_summary_and_preview_show_skipped():
    from trackllm_website.bi.costs import build_cost_summary, format_preview

    policy = flagship_policy(6.0, ["m/a", "m/b"])
    cands = [ep("m/a", "p", 0.001), ep("m/b", "p", 0.001)]
    summary = build_cost_summary(cands, policy, [])
    assert summary["skipped"] == [
        {"model": "m/b", "provider": "p", "rule": "flagships", "monthly_cost": 6.0}
    ]
    text = format_preview(summary)
    assert "Skipped (over budget):" in text
    assert "m/b" in text
