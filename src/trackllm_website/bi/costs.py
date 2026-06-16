"""Compute the BI cost/spend summary consumed by the website costs page."""

from collections import defaultdict
from pathlib import Path

import orjson

from trackllm_website.bi.selection import (
    SelectionPolicy,
    monthly_cost,
    select_monitoring_targets,
)
from trackllm_website.config import Endpoint, config

COSTS_FILENAME = "bi_costs.json"


def build_cost_summary(candidates: list[Endpoint], policy: SelectionPolicy) -> dict:
    selected, breakdown = select_monitoring_targets(candidates, policy)
    rows = sorted(
        (
            {
                "model": e.model,
                "provider": e.provider,
                "rule": breakdown[e],
                "cost_per_request": e.cost_per_request,
                "monthly_cost": monthly_cost(e),
            }
            for e in selected
        ),
        key=lambda r: r["monthly_cost"],
        reverse=True,
    )
    by_rule: dict[str, dict] = defaultdict(lambda: {"count": 0, "monthly_cost": 0.0})
    for r in rows:
        by_rule[r["rule"]]["count"] += 1
        by_rule[r["rule"]]["monthly_cost"] += r["monthly_cost"]
    return {
        "budget_per_month": policy.budget_per_month,
        "run_rate_per_month": sum(r["monthly_cost"] for r in rows),
        "n_selected": len(selected),
        "by_rule": dict(by_rule),
        "endpoints": rows,  # full list, descending; frontend shows top-20 + expand
    }


def write_cost_summary() -> None:
    from trackllm_website.bi.selection import load_policy
    from trackllm_website.config import root

    policy = load_policy(root / config.bi.selection_path)
    summary = build_cost_summary(config.endpoints_bi, policy)
    path = Path(config.data_dir) / COSTS_FILENAME
    path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
