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
# endpoint page's spend breakdown.
GROUP_ORDER = ["onboarding", "monitoring", "lt", "vetting", "other"]
# Reader-facing labels for the internal group keys above.
GROUP_LABEL = {
    "onboarding": "B3IT (onboarding)",
    "monitoring": "B3IT (monitoring)",
    "lt": "LT",
    "vetting": "Vetting",
    "other": "Other",
}


def group_for_kind(kind: str) -> str:
    return GROUPS.get(kind, "other")


def _ordered(groups: dict[str, float]) -> dict[str, float]:
    return {g: groups[g] for g in GROUP_ORDER if g in groups}


def _new_endpoint() -> dict:
    return {
        "groups": defaultdict(float),
        "last_30d": 0.0,
        "n_queries": 0,
        "since": None,
    }


def aggregate_spend(spend_dir: Path, today: str) -> dict:
    """Site-wide totals by group, plus one record per billed slug for its page.

    `by_endpoint` covers every slug in the ledger, including discovery probes
    that never got a page; render.py looks up only the slugs it renders.
    """
    cumulative: dict[str, float] = defaultdict(float)
    last_30d: dict[str, float] = defaultdict(float)
    by_endpoint: dict[str, dict] = defaultdict(_new_endpoint)
    cutoff = date.fromisoformat(today) - timedelta(days=30)

    for slug, rec in iter_ledger(spend_dir):
        g = group_for_kind(rec["kind"])
        cost = rec["cost"]
        day = str(rec["timestamp"])[:10]
        ep = by_endpoint[slug]
        cumulative[g] += cost
        ep["groups"][g] += cost
        ep["n_queries"] += rec["n_queries"]
        ep["since"] = day if ep["since"] is None else min(ep["since"], day)
        if date.fromisoformat(day) > cutoff:
            last_30d[g] += cost
            ep["last_30d"] += cost

    return {
        "group_order": GROUP_ORDER,
        "group_label": GROUP_LABEL,
        "cumulative": _ordered(cumulative),
        "last_30d": _ordered(last_30d),
        "by_endpoint": {
            s: {
                **ep,
                "groups": _ordered(ep["groups"]),
                "total": sum(ep["groups"].values()),
            }
            for s, ep in by_endpoint.items()
        },
    }
