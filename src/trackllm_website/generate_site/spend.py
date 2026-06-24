import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

GROUPS = {
    "onboard": "onboarding",
    "recheck": "onboarding",
    "reinit": "onboarding",
    "monitor": "monitoring",
    "lt": "lt",
    "vetting": "vetting",
}


def group_for_kind(kind: str) -> str:
    return GROUPS.get(kind, "other")


def aggregate_spend(spend_dir: Path, today: str) -> dict:
    cumulative: dict[str, float] = defaultdict(float)
    last_30d: dict[str, float] = defaultdict(float)
    daily: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_endpoint: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cutoff = date.fromisoformat(today) - timedelta(days=30)

    if spend_dir.exists():
        for f in sorted(spend_dir.glob("*/*.jsonl")):
            slug = f.parent.name
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                g = group_for_kind(rec["kind"])
                cost = rec["cost"]
                day = str(rec["timestamp"])[:10]
                cumulative[g] += cost
                by_endpoint[slug][g] += cost
                daily[day][g] += cost
                if date.fromisoformat(day) > cutoff:
                    last_30d[g] += cost

    by_ep = [
        {"slug": s, "groups": dict(g), "total": sum(g.values())}
        for s, g in by_endpoint.items()
    ]
    by_ep.sort(key=lambda r: r["total"], reverse=True)
    return {
        "cumulative": dict(cumulative),
        "last_30d": dict(last_30d),
        "daily": [{"date": d, "groups": dict(g)} for d, g in sorted(daily.items())],
        "by_endpoint": by_ep,
    }
