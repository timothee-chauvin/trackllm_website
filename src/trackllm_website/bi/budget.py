"""Month-end spend projection and the budget kill-planner (the "OOM killer").

Fires on *projected* month-end overshoot, not after the money is gone: callers
compare projected_month_end() (+ the cost of what they are about to do) against
config.budget.hard_cap_per_month and apply plan_kills() — pending onboards and
reinits are dropped first (most expensive first), then monitored endpoints are
retired by observed $/day, non-flagships before flagships.

Recovery from a `budget` retirement is deliberate, not automatic: the endpoint
comes back only through the existing recheck path, once selection and vetting
admit it again — the operator fixes the policy, not the killer.
"""

from calendar import monthrange
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from trackllm_website.config import Endpoint, config
from trackllm_website.spend import iter_ledger


def month_to_date(spend_dir: Path, now: datetime) -> float:
    month = f"{now:%Y-%m}"
    return sum(
        (
            rec["cost"]
            for _, rec in iter_ledger(spend_dir)
            if str(rec["timestamp"])[:7] == month
        ),
        0.0,
    )


def _window_start(now: datetime, window_days: int) -> str:
    return (now.date() - timedelta(days=window_days - 1)).isoformat()


def daily_rate(spend_dir: Path, now: datetime, window_days: int) -> float:
    """Mean daily spend (all kinds) over the trailing window_days calendar days,
    today included."""
    start = _window_start(now, window_days)
    total = sum(
        rec["cost"]
        for _, rec in iter_ledger(spend_dir)
        if start <= str(rec["timestamp"])[:10] <= now.date().isoformat()
    )
    return total / window_days


def remaining_days_in_month(now: datetime) -> int:
    return monthrange(now.year, now.month)[1] - now.day


def projected_month_end(spend_dir: Path, now: datetime, window_days: int) -> float:
    return month_to_date(spend_dir, now) + daily_rate(
        spend_dir, now, window_days
    ) * remaining_days_in_month(now)


def daily_rates_by_slug(
    spend_dir: Path, now: datetime, window_days: int
) -> dict[str, float]:
    """Mean daily spend per endpoint slug over the trailing window, in one
    ledger pass (per-slug reads would re-scan every file per endpoint)."""
    start = _window_start(now, window_days)
    end = now.date().isoformat()
    totals: dict[str, float] = defaultdict(float)
    for slug, rec in iter_ledger(spend_dir):
        if start <= str(rec["timestamp"])[:10] <= end:
            totals[slug] += rec["cost"]
    return {slug: total / window_days for slug, total in totals.items()}


def expected_reinit_queries() -> int:
    r = config.bi.reinit
    p1 = config.bi.phase_1
    return (
        r.top_k_bis * r.reprobe_samples
        + p1.tokens_per_endpoint * p1.queries_per_token
        + p1.target_border_inputs * r.reference_samples
    )


def expected_reinit_cost(endpoint: Endpoint) -> float:
    """Worst-case-ish cost of a full re-init/onboarding. Endpoints without a
    vetted cost_per_request are priced at the per-query guard threshold — the
    most any query can bill before QueryTooExpensive fires."""
    per_query = (
        endpoint.cost_per_request
        if endpoint.cost_per_request is not None
        else config.api.max_cost_per_query
    )
    return per_query * expected_reinit_queries()


@dataclass
class KillPlan:
    dropped_pending: list = field(default_factory=list)
    retired: list = field(default_factory=list)


def plan_kills(
    overshoot: float,
    pending: list[tuple],
    monitored: list[tuple],
    remaining_days: int,
) -> KillPlan:
    """Pure planner: pick what to drop to bring the projection back under cap.

    pending: (key, expected_cost) one-off actions; monitored: (key, daily_rate,
    is_flagship) recurring spenders. Dropping a pending action saves its full
    cost; retiring a monitored endpoint saves rate x remaining_days.
    """
    plan = KillPlan()
    if overshoot <= 0:
        return plan
    for key, cost in sorted(pending, key=lambda p: -p[1]):
        if overshoot <= 0:
            return plan
        plan.dropped_pending.append(key)
        overshoot -= cost
    for key, rate, _ in sorted(monitored, key=lambda m: (m[2], -m[1])):
        if overshoot <= 0:
            return plan
        # Retiring frees rate x remaining_days; when that is zero (idle endpoint,
        # last day of month, or an overshoot BI cannot fix, e.g. LT-driven),
        # retiring destroys monitoring continuity for no budget gain.
        if rate * remaining_days <= 0:
            continue
        plan.retired.append(key)
        overshoot -= rate * remaining_days
    return plan
