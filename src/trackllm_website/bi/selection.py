"""BI selection policy: data models + loader + the pure rule engine."""

import fnmatch
import tomllib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

from trackllm_website.config import Endpoint, config, logger


class Rule(BaseModel):
    name: str
    kind: Literal["models", "providers", "popular"]
    patterns: list[str]
    providers_per_model: int | Literal["all"] | None = None
    endpoints_per_provider: int | None = None
    max_monthly_cost: float | None = None
    latest_n: int | None = None
    flagship: bool = False

    @model_validator(mode="after")
    def _require_selection_width(self) -> "Rule":
        # A None width would silently slice eps[:None] == all providers.
        if self.kind == "providers":
            if self.endpoints_per_provider is None:
                raise ValueError(f"rule {self.name!r}: endpoints_per_provider required")
        elif self.providers_per_model is None:
            raise ValueError(f"rule {self.name!r}: providers_per_model required")
        return self


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


def exceeds_ceiling(
    cost_per_request: float, model: str, provider: str, policy: SelectionPolicy
) -> bool:
    """A non-flagship endpoint above the monthly ceiling is too_expensive to keep probing.

    Flagships (selection's flagship-rule patterns) are exempt — they're monitored
    regardless of cost.
    """
    fake = Endpoint(api="openrouter", model=model, provider=provider, cost=(0, 0))
    if _matches_any(fake, policy.flagship_patterns()):
        return False
    return cost_per_request * config.bi.samples_per_month > policy.max_endpoint_cost


def select_monitoring_targets(
    candidates: list[Endpoint], policy: SelectionPolicy, popular_models: list[str]
) -> tuple[list[Endpoint], dict[Endpoint, str], list[tuple[Endpoint, str]]]:
    """Pure: apply rules in order within budget.

    Returns (selected, rule-label-by-endpoint, skipped). budget_per_month binds
    every rule, flagships included; a named (non-wildcard) models rule lists an
    endpoint priority, so patterns fill in list order and an endpoint that no
    longer fits is skipped, logged as an error and returned in `skipped` so the
    daily digest can show it in red — never an exception, which would fail the
    daily job until the config changed. Wildcard and popular fill rules stop at
    budget silently, as before.
    """
    pool = [e for e in candidates if not _matches_any(e, policy.exclude)]
    by_model: dict[str, list[Endpoint]] = defaultdict(list)
    by_provider: dict[str, list[Endpoint]] = defaultdict(list)
    for e in pool:
        by_model[e.model].append(e)
        by_provider[e.provider_without_suffix].append(e)
    for d in (by_model, by_provider):
        for k in d:
            d[k].sort(key=lambda e: (monthly_cost(e), str(e)))

    selected: dict[Endpoint, str] = {}
    skipped: list[tuple[Endpoint, str]] = []
    spent = 0.0

    def add(e: Endpoint, label: str) -> None:
        nonlocal spent
        if e in selected:
            return
        selected[e] = label
        spent += monthly_cost(e)

    def skip(e: Endpoint, rule_name: str) -> None:
        skipped.append((e, rule_name))
        logger.error(
            f"rule {rule_name!r}: {e.model}#{e.provider} "
            f"(${monthly_cost(e):.2f}/mo) does not fit in the "
            f"${policy.budget_per_month:.2f}/mo budget (${spent:.2f} spent); skipped"
        )

    for rule in policy.rules:
        is_wildcard = rule.patterns == ["*"]
        if rule.kind == "models":
            stop = False
            # Each pattern is its own family: latest_n picks the newest versions
            # within that pattern before the per-model provider selection runs.
            for pattern in rule.patterns:
                pat_models = [
                    m for m in by_model if _matches_any(by_model[m][0], [pattern])
                ]
                if rule.latest_n is not None:
                    # Sort newest-first; model name is a deterministic tiebreak so a
                    # created-tie for the last slot doesn't flap with candidate order.
                    pat_models.sort(
                        key=lambda m: (
                            by_model[m][0].created
                            or datetime.min.replace(tzinfo=timezone.utc),
                            m,
                        ),
                        reverse=True,
                    )
                    pat_models = pat_models[: rule.latest_n]
                pat_models.sort(key=lambda m: (monthly_cost(by_model[m][0]), m))
                for m in pat_models:
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
                        if spent + monthly_cost(e) > policy.budget_per_month:
                            if is_wildcard:
                                stop = (
                                    True  # budget reached; remaining eps are costlier
                                )
                                break
                            skip(e, rule.name)  # a cheaper later pattern may still fit
                            continue
                        add(e, rule.name)
                    if stop:
                        break
                if stop:
                    break
        elif rule.kind == "popular":
            stop = False
            # popular_models is already popularity-ranked (top first); select those
            # present among candidates, cheapest provider first, like the models branch.
            for m in popular_models:
                if m not in by_model:
                    continue
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
                    # popular is a fill rule (popularity feed, not a curated family):
                    # stop gracefully at budget.
                    if spent + monthly_cost(e) > policy.budget_per_month:
                        stop = True
                        break
                    add(e, rule.name)
                if stop:
                    break
        else:  # providers
            for prov, eps in sorted(by_provider.items()):
                # Patterns match provider names here (the by_provider key, i.e. the
                # provider without its /fp8-style suffix), not models; ["*"] keeps
                # the every-provider fill behavior.
                if not any(fnmatch.fnmatch(prov, p) for p in rule.patterns):
                    continue
                for e in eps[: rule.endpoints_per_provider]:
                    if e in selected:
                        continue
                    if (
                        rule.max_monthly_cost is not None
                        and monthly_cost(e) > rule.max_monthly_cost
                    ):
                        continue
                    if spent + monthly_cost(e) > policy.budget_per_month:
                        continue  # this provider too pricey; try cheaper providers
                    add(e, rule.name)

    return list(selected), selected, skipped
