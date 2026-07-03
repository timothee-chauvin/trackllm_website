from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from trackllm_website.spend import iter_ledger

GROUPS = {
    "onboard": "onboarding",
    "recheck": "onboarding",
    "reinit": "onboarding",
    "monitor": "monitoring",
    "lt": "lt",
    "vetting": "vetting",
}
# Display order, emitted in spend.json as the single source of truth for the
# spend page's columns and the chart's traces.
GROUP_ORDER = ["onboarding", "monitoring", "lt", "vetting", "other"]


def group_for_kind(kind: str) -> str:
    return GROUPS.get(kind, "other")


def _ordered(groups: dict[str, float]) -> dict[str, float]:
    return {g: groups[g] for g in GROUP_ORDER if g in groups}


def aggregate_spend(spend_dir: Path, today: str) -> dict:
    cumulative: dict[str, float] = defaultdict(float)
    last_30d: dict[str, float] = defaultdict(float)
    daily: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_endpoint: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cutoff = date.fromisoformat(today) - timedelta(days=30)

    for slug, rec in iter_ledger(spend_dir):
        g = group_for_kind(rec["kind"])
        cost = rec["cost"]
        day = str(rec["timestamp"])[:10]
        cumulative[g] += cost
        by_endpoint[slug][g] += cost
        daily[day][g] += cost
        if date.fromisoformat(day) > cutoff:
            last_30d[g] += cost

    by_ep = [
        {"slug": s, "groups": _ordered(g), "total": sum(g.values())}
        for s, g in by_endpoint.items()
    ]
    by_ep.sort(key=lambda r: r["total"], reverse=True)
    return {
        "group_order": GROUP_ORDER,
        "cumulative": _ordered(cumulative),
        "last_30d": _ordered(last_30d),
        "daily": [{"date": d, "groups": _ordered(g)} for d, g in sorted(daily.items())],
        "by_endpoint": by_ep,
    }
