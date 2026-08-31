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


def test_flagship_within_budget_exempt_from_ceiling_only():
    policy = SelectionPolicy(
        budget_per_month=5.0,
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
    # gpt-5 monthly cost 3.0 > ceiling but within budget: the engine doesn't
    # enforce the ceiling (vetting does), and budget now binds flagships too.
    cands = [ep("openai/gpt-5", "openai", 0.0005)]  # 0.0005*6000 = 3.0
    selected, breakdown, skipped = select_monitoring_targets(cands, policy, [])
    assert cands[0] in selected
    assert breakdown[cands[0]] == "flagships"
    assert skipped == []


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
    selected, _, _ = select_monitoring_targets(cands, policy, [])
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
    selected, _, _ = select_monitoring_targets(cands, policy, [])
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
    selected, _, _ = select_monitoring_targets(cands, policy, [])
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
    selected, _, _ = select_monitoring_targets(cands, policy, [])
    assert len(selected) == 1


def test_named_overshoot_then_wildcard_cannot_refill():
    # a named rule's skip must not free budget for a trailing wildcard to consume
    # past the named rule's own endpoints.
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
    # m/a fits (6.0), m/b is skipped (12.0 > 10), m/c (6.0) doesn't fit either.
    cands = [
        ep("m/a", "p", 0.001),
        ep("m/b", "p", 0.001),
        ep("m/c", "p", 0.001),
    ]
    selected, _, skipped = select_monitoring_targets(cands, policy, [])
    assert selected == [cands[0]]
    assert skipped == [(cands[1], "named")]


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
    selected, _, _ = select_monitoring_targets(cands, policy, [])
    assert [e.provider for e in selected] == ["provA"]


def _providers_policy(patterns):
    return SelectionPolicy(
        budget_per_month=100.0,
        max_endpoint_cost=100.0,
        exclude=[],
        rules=[
            Rule(
                name="providers",
                kind="providers",
                patterns=patterns,
                endpoints_per_provider=1,
            )
        ],
    )


def test_providers_rule_honors_patterns():
    # a narrowed providers rule must not silently select from every provider
    cands = [
        ep("m/a", "deepinfra", 0.00002),
        ep("m/b", "deepinfra-turbo", 0.00002),
        ep("m/c", "together", 0.00002),
        ep("m/d", "novita", 0.00002),
    ]
    selected, _, _ = select_monitoring_targets(
        cands, _providers_policy(["deepinfra*"]), []
    )
    assert sorted(e.provider for e in selected) == ["deepinfra", "deepinfra-turbo"]


def test_providers_rule_wildcard_covers_everyone():
    cands = [ep("m/a", "deepinfra", 0.00002), ep("m/c", "together", 0.00002)]
    selected, _, _ = select_monitoring_targets(cands, _providers_policy(["*"]), [])
    assert sorted(e.provider for e in selected) == ["deepinfra", "together"]


def test_providers_rule_matches_the_suffix_free_provider_name():
    # by_provider is keyed by provider_without_suffix, so an exact name matches
    # every variant of that provider
    cands = [ep("m/a", "deepinfra/fp8", 0.00002), ep("m/c", "together", 0.00002)]
    selected, _, _ = select_monitoring_targets(
        cands, _providers_policy(["deepinfra"]), []
    )
    assert [e.provider for e in selected] == ["deepinfra/fp8"]


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
    selected1, labels1, _ = select_monitoring_targets(list(cands), policy, [])
    selected2, labels2, _ = select_monitoring_targets(list(cands), policy, [])
    assert selected1 == selected2
    assert labels1 == labels2
