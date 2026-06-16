"""BI selection policy: data models + loader + the pure rule engine."""

import fnmatch
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from trackllm_website.config import Endpoint, config, logger


class Rule(BaseModel):
    name: str
    kind: Literal["models", "providers"]
    patterns: list[str]
    providers_per_model: int | Literal["all"] | None = None
    endpoints_per_provider: int | None = None
    max_monthly_cost: float | None = None
    flagship: bool = False


class SelectionPolicy(BaseModel):
    budget_per_month: float
    max_endpoint_cost: float
    exclude: list[str]
    rules: list[Rule]

    def flagship_patterns(self) -> list[str]:
        return [p for r in self.rules if r.flagship for p in r.patterns]


def load_policy(path: Path) -> SelectionPolicy:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    rules = [Rule(**r) for r in raw.pop("rule", [])]
    return SelectionPolicy(rules=rules, **raw)


def monthly_cost(endpoint: Endpoint) -> float:
    if endpoint.cost_per_request is None:
        raise ValueError(f"{endpoint} has no measured cost_per_request")
    return endpoint.cost_per_request * config.bi.samples_per_month


def _matches_any(endpoint: Endpoint, patterns: list[str]) -> bool:
    targets = (endpoint.model, f"{endpoint.model}#{endpoint.provider}")
    return any(fnmatch.fnmatch(t, p) for t in targets for p in patterns)


def select_monitoring_targets(
    candidates: list[Endpoint], policy: SelectionPolicy
) -> tuple[list[Endpoint], dict[Endpoint, str]]:
    """Pure: apply rules in order within budget. Returns (selected, rule-label-by-endpoint).

    Flagships are budget-exempt and ceiling-exempt: a flagship total over budget is
    logged as a warning, not an error. Wildcard fill rules stop at budget. A
    non-flagship named (non-wildcard) rule whose total exceeds budget raises ValueError
    (a config error, not silent truncation).
    """
    pool = [e for e in candidates if not _matches_any(e, policy.exclude)]
    by_model: dict[str, list[Endpoint]] = defaultdict(list)
    by_provider: dict[str, list[Endpoint]] = defaultdict(list)
    for e in pool:
        by_model[e.model].append(e)
        by_provider[e.provider_without_suffix].append(e)
    for d in (by_model, by_provider):
        for k in d:
            d[k].sort(key=monthly_cost)

    selected: dict[Endpoint, str] = {}
    spent = 0.0

    def add(e: Endpoint, label: str) -> None:
        nonlocal spent
        if e in selected:
            return
        selected[e] = label
        spent += monthly_cost(e)

    for rule in policy.rules:
        is_wildcard = rule.patterns == ["*"]
        if rule.kind == "models":
            model_keys = [
                m for m in by_model if _matches_any(by_model[m][0], rule.patterns)
            ]
            model_keys.sort(key=lambda m: monthly_cost(by_model[m][0]))
            for m in model_keys:
                eps = by_model[m]
                n = (
                    len(eps)
                    if rule.providers_per_model == "all"
                    else rule.providers_per_model
                )
                for e in eps[:n]:
                    if e in selected:
                        continue
                    if (
                        rule.max_monthly_cost is not None
                        and monthly_cost(e) > rule.max_monthly_cost
                    ):
                        continue
                    if (
                        not rule.flagship
                        and is_wildcard
                        and spent + monthly_cost(e) > policy.budget_per_month
                    ):
                        return list(selected), selected  # budget reached, stop fill
                    add(e, rule.name)
        else:  # providers
            for prov, eps in sorted(by_provider.items()):
                for e in eps[: rule.endpoints_per_provider]:
                    if e in selected:
                        continue
                    if (
                        rule.max_monthly_cost is not None
                        and monthly_cost(e) > rule.max_monthly_cost
                    ):
                        continue
                    if (
                        is_wildcard
                        and spent + monthly_cost(e) > policy.budget_per_month
                    ):
                        continue  # this provider too pricey; try cheaper providers
                    add(e, rule.name)

    flagship_names = {r.name for r in policy.rules if r.flagship}
    flagship_spend = sum(
        monthly_cost(e) for e, lbl in selected.items() if lbl in flagship_names
    )
    nonflag = sum(
        monthly_cost(e) for e, lbl in selected.items() if lbl not in flagship_names
    )
    if flagship_spend > policy.budget_per_month:
        logger.warning(
            f"flagship selection ${flagship_spend:.2f}/mo exceeds budget "
            f"${policy.budget_per_month:.2f} (flagships are budget-exempt)"
        )
    if nonflag > policy.budget_per_month:
        raise ValueError(
            f"non-flagship selection ${nonflag:.2f}/mo exceeds budget "
            f"${policy.budget_per_month:.2f}"
        )
    return list(selected), selected
