"""Tests: month-end spend projection and the budget kill-planner."""

from datetime import datetime, timezone

from trackllm_website.bi.budget import (
    KillPlan,
    daily_rate,
    daily_rates_by_slug,
    expected_reinit_cost,
    expected_reinit_queries,
    month_to_date,
    plan_kills,
    projected_month_end,
)
from trackllm_website.config import Endpoint, config
from trackllm_website.spend import Spend, append_entry

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
WINDOW = 7


def _seed(tmp_path, slug, kind, cost, when):
    append_entry(tmp_path, slug, kind, Spend(cost=cost, n_queries=1), when)


def test_month_to_date_sums_all_kinds_of_current_month(tmp_path):
    _seed(tmp_path, "a", "lt", 1.0, datetime(2026, 8, 3, tzinfo=timezone.utc))
    _seed(tmp_path, "a", "monitor", 2.0, datetime(2026, 8, 19, tzinfo=timezone.utc))
    _seed(tmp_path, "b", "onboard", 4.0, datetime(2026, 8, 20, 9, tzinfo=timezone.utc))
    _seed(tmp_path, "b", "lt", 100.0, datetime(2026, 7, 31, tzinfo=timezone.utc))
    assert abs(month_to_date(tmp_path, NOW) - 7.0) < 1e-9


def test_month_to_date_empty_ledger(tmp_path):
    assert month_to_date(tmp_path, NOW) == 0.0


def test_daily_rate_averages_over_calendar_days_not_active_days(tmp_path):
    # 14.0 spent on a single day inside a 7-day window -> 2.0/day, not 14.0/day.
    _seed(tmp_path, "a", "reinit", 14.0, datetime(2026, 8, 18, tzinfo=timezone.utc))
    assert abs(daily_rate(tmp_path, NOW, WINDOW) - 2.0) < 1e-9


def test_daily_rate_excludes_entries_outside_window(tmp_path):
    # Window is the 7 calendar days ending today: Aug 14-20 inclusive.
    _seed(tmp_path, "a", "lt", 7.0, datetime(2026, 8, 14, tzinfo=timezone.utc))
    _seed(tmp_path, "a", "lt", 70.0, datetime(2026, 8, 13, tzinfo=timezone.utc))
    assert abs(daily_rate(tmp_path, NOW, WINDOW) - 1.0) < 1e-9


def test_projected_month_end_adds_rate_times_remaining_days(tmp_path):
    _seed(tmp_path, "a", "lt", 10.0, datetime(2026, 8, 2, tzinfo=timezone.utc))
    _seed(tmp_path, "a", "lt", 7.0, datetime(2026, 8, 16, tzinfo=timezone.utc))
    # mtd = 17, rate = 1/day, 11 days left after Aug 20 (Aug has 31 days).
    assert abs(projected_month_end(tmp_path, NOW, WINDOW) - 28.0) < 1e-9


def test_expected_reinit_queries_derives_from_config():
    r = config.bi.reinit
    p1 = config.bi.phase_1
    expected = (
        r.top_k_bis * r.reprobe_samples
        + p1.tokens_per_endpoint * p1.queries_per_token
        + p1.target_border_inputs * r.reference_samples
    )
    assert expected_reinit_queries() == expected


def test_expected_reinit_cost_uses_guard_threshold_when_unpriced():
    priced = Endpoint(
        api="openrouter", model="m/a", provider="p", cost=(1, 1), cost_per_request=1e-5
    )
    unpriced = Endpoint(api="openrouter", model="m/b", provider="p", cost=(1, 1))
    n = expected_reinit_queries()
    assert abs(expected_reinit_cost(priced) - 1e-5 * n) < 1e-12
    assert (
        abs(expected_reinit_cost(unpriced) - config.api.max_cost_per_query * n) < 1e-12
    )


def test_plan_kills_nothing_when_under_cap():
    plan = plan_kills(
        overshoot=-1.0,
        pending=[("onb1", 5.0)],
        monitored=[("mon1", 0.5, False)],
        remaining_days=10,
    )
    assert plan == KillPlan(dropped_pending=[], retired=[])


def test_plan_kills_drops_most_expensive_pending_first():
    plan = plan_kills(
        overshoot=6.0,
        pending=[("cheap", 2.0), ("dear", 5.0), ("mid", 3.0)],
        monitored=[],
        remaining_days=10,
    )
    # 5.0 + 3.0 covers the 6.0 overshoot; "cheap" survives.
    assert plan.dropped_pending == ["dear", "mid"]
    assert plan.retired == []


def test_plan_kills_retires_monitored_only_after_all_pending():
    plan = plan_kills(
        overshoot=10.0,
        pending=[("onb", 2.0)],
        monitored=[("mon", 1.0, False)],
        remaining_days=10,
    )
    # Pending (2.0) is not enough: mon must go too (1.0/day x 10 days).
    assert plan.dropped_pending == ["onb"]
    assert plan.retired == ["mon"]


def test_plan_kills_orders_non_flagship_before_flagship_then_by_rate():
    plan = plan_kills(
        overshoot=100.0,
        pending=[],
        monitored=[
            ("flag_hot", 5.0, True),
            ("plain_cool", 0.1, False),
            ("plain_hot", 2.0, False),
        ],
        remaining_days=10,
    )
    assert plan.retired == ["plain_hot", "plain_cool", "flag_hot"]


def test_plan_kills_never_retires_zero_rate_endpoints():
    # An overshoot BI retirements cannot fix (e.g. driven by LT spend) must not
    # wipe out endpoints whose retirement frees nothing.
    plan = plan_kills(
        overshoot=50.0,
        pending=[],
        monitored=[("idle", 0.0, False), ("hot", 1.0, False)],
        remaining_days=10,
    )
    assert plan.retired == ["hot"]


def test_plan_kills_retires_nothing_on_last_day_of_month():
    plan = plan_kills(
        overshoot=50.0,
        pending=[("onb", 2.0)],
        monitored=[("hot", 1.0, False)],
        remaining_days=0,
    )
    assert plan.dropped_pending == ["onb"]
    assert plan.retired == []


def test_plan_kills_stops_once_overshoot_is_covered():
    plan = plan_kills(
        overshoot=5.0,
        pending=[],
        monitored=[("hot", 1.0, False), ("cool", 0.5, False)],
        remaining_days=10,
    )
    # hot alone frees 10.0 >= 5.0; cool survives.
    assert plan.retired == ["hot"]


def test_daily_rates_by_slug_single_pass(tmp_path):
    _seed(tmp_path, "a", "monitor", 7.0, datetime(2026, 8, 18, tzinfo=timezone.utc))
    _seed(tmp_path, "a", "monitor", 7.0, datetime(2026, 8, 19, tzinfo=timezone.utc))
    _seed(tmp_path, "b", "lt", 14.0, datetime(2026, 8, 20, tzinfo=timezone.utc))
    _seed(tmp_path, "b", "lt", 99.0, datetime(2026, 8, 1, tzinfo=timezone.utc))
    rates = daily_rates_by_slug(tmp_path, NOW, WINDOW)
    assert abs(rates["a"] - 2.0) < 1e-9
    assert abs(rates["b"] - 2.0) < 1e-9


def test_is_flagship_matches_flagship_rule_patterns():
    from trackllm_website.bi.selection import Rule, SelectionPolicy, is_flagship

    policy = SelectionPolicy(
        budget_per_month=10.0,
        max_endpoint_cost=0.5,
        exclude=[],
        rules=[
            Rule(
                name="flagships",
                kind="models",
                patterns=["big/model"],
                providers_per_model=1,
                flagship=True,
            )
        ],
    )
    flag = Endpoint(api="openrouter", model="big/model", provider="p", cost=(1, 1))
    plain = Endpoint(api="openrouter", model="small/model", provider="p", cost=(1, 1))
    assert is_flagship(flag, policy)
    assert not is_flagship(plain, policy)
