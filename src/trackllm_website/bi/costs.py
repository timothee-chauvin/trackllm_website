"""Compute the BI cost/spend summary consumed by the website costs page."""

from collections import defaultdict
from pathlib import Path

import orjson

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.common import resolve_strategies
from trackllm_website.bi.popularity import fetch_popular_models
from trackllm_website.bi.selection import (
    SelectionPolicy,
    monthly_cost,
    select_monitoring_targets,
)
from trackllm_website.bi.vetting import vet_endpoint
from trackllm_website.config import Endpoint, config, logger, root
from trackllm_website.util import gather_with_concurrency

COSTS_FILENAME = "bi_costs.json"


def build_cost_summary(
    candidates: list[Endpoint], policy: SelectionPolicy, popular_models: list[str]
) -> dict:
    selected, breakdown = select_monitoring_targets(candidates, policy, popular_models)
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


def _popular_models_best_effort() -> list[str]:
    """Popularity is advisory for previews/summaries: degrade to [] on any failure."""
    try:
        return fetch_popular_models(config.bi.popularity.top_n)
    except Exception as e:
        logger.warning(f"popularity fetch failed, proceeding without it: {e}")
        return []


def write_cost_summary() -> None:
    from trackllm_website.bi.selection import load_policy

    policy = load_policy(root / config.bi.selection_path)
    summary = build_cost_summary(
        config.endpoints_bi, policy, _popular_models_best_effort()
    )
    path = Path(config.data_dir) / COSTS_FILENAME
    path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))


async def ensure_costs(candidates: list[Endpoint], *, save: bool) -> list[Endpoint]:
    """Return candidates with cost_per_request filled, probing only those missing it.

    Probed costs are written back to endpoints_bi.yaml when save=True so repeated
    previews don't re-probe. Endpoints that fail vetting (liar/transient) are dropped.
    """
    missing = [e for e in candidates if e.cost_per_request is None]
    if missing:
        logger.info(f"cost-preview: measuring {len(missing)} endpoints missing a cost")
        async with OpenRouterClient(timeout=60.0) as probe:
            strategies, _ = await resolve_strategies(probe, missing)

        async def measure(client: OpenRouterClient, e: Endpoint) -> None:
            strat = strategies.get(str(e))
            if strat is None:
                return
            res = await vet_endpoint(client, e, strat)
            if res.bucket == "candidate":
                e.cost_per_request = res.cost_per_request

        async with OpenRouterClient() as client:
            await gather_with_concurrency(
                config.api.max_workers, *(measure(client, e) for e in missing)
            )

    priced = [e for e in candidates if e.cost_per_request is not None]
    if save:
        from trackllm_website.update_endpoints import save_endpoints_bi

        save_endpoints_bi(priced)
    return priced


def format_preview(summary: dict) -> str:
    lines = [
        f"Budget:   ${summary['budget_per_month']:.2f}/mo",
        f"Run-rate: ${summary['run_rate_per_month']:.2f}/mo  ({summary['n_selected']} endpoints)",
        "",
        "By rule:",
    ]
    for rule, info in sorted(
        summary["by_rule"].items(), key=lambda kv: -kv[1]["monthly_cost"]
    ):
        lines.append(
            f"  {rule:18s} {info['count']:4d} endpoints  ${info['monthly_cost']:.2f}/mo"
        )
    lines += ["", "Most expensive selected endpoints:"]
    for r in summary["endpoints"][:25]:
        lines.append(
            f"  ${r['monthly_cost']:6.2f}/mo  [{r['rule']:16s}] {r['model']} ({r['provider']})"
        )
    return "\n".join(lines)


async def preview(policy_path: str | None = None) -> None:
    """Price a bi_selection.toml (default: the configured one) without monitoring."""
    from trackllm_website.bi.selection import load_policy

    path = root / (policy_path or config.bi.selection_path)
    policy = load_policy(path)
    candidates = await ensure_costs(list(config.endpoints_bi), save=True)
    summary = build_cost_summary(candidates, policy, _popular_models_best_effort())
    print(format_preview(summary))


if __name__ == "__main__":
    import asyncio

    import fire

    fire.Fire(
        {
            "preview": lambda policy_path=None: asyncio.run(preview(policy_path)),
            "write": write_cost_summary,
        }
    )
