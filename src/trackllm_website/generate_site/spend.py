from collections import defaultdict
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

from trackllm_website.bi.state import EndpointBIState
from trackllm_website.bi.vetting import EndpointCache
from trackllm_website.config import config, root
from trackllm_website.generate_site.lt import EndpointInfo
from trackllm_website.generate_site.naming import display_name
from trackllm_website.spend import iter_ledger
from trackllm_website.util import slugify

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
# Reader-facing labels for the internal group keys above -- also emitted in
# spend.json, so the Jinja template and spend.ts share the one mapping.
GROUP_LABEL = {
    "onboarding": "B3IT (onboarding)",
    "monitoring": "B3IT (monitoring)",
    "lt": "LT",
    "vetting": "Vetting",
    "other": "Other",
}
VETTING_TIP = (
    "Vetting probes each candidate endpoint once, comparing what OpenRouter "
    "actually billed to what its advertised price implies, to catch endpoints "
    "that misreport cost and to measure the real per-request cost of the ones "
    "that check out. Only endpoints that pass go on to be onboarded for "
    "monitoring."
)
ENDPOINTS_CACHE_BI_PATH = root / "endpoints_cache_bi.yaml"


def group_for_kind(kind: str) -> str:
    return GROUPS.get(kind, "other")


def build_endpoint_names(
    discovered: Iterable[EndpointInfo], bi_states: dict[str, EndpointBIState]
) -> dict[str, str]:
    """slug -> "model @ provider" for every endpoint spend.py can name.

    The spend ledger covers every slug ever billed, including discovery probes
    that never made it onto a page -- so this checks every registry that can
    name a slug, in order of size/authority: the BI candidate catalogs (config,
    by far the largest -- everything ever considered for vetting), BI state
    (onboarded or retired), the vetting reject cache (probed and rejected, so
    never onboarded), and discovered LT endpoints. A slug that errored out
    before landing in any of these has no recoverable name; callers fall back
    to the slug itself.
    """
    names: dict[str, str] = {}
    for ep in config.endpoints_bi + config.endpoints_bi_prevalence:
        names[slugify(f"{ep.model}#{ep.provider}")] = display_name(
            ep.model, ep.provider
        )
    for slug, state in bi_states.items():
        names.setdefault(
            slug, display_name(state.endpoint.model, state.endpoint.provider)
        )
    cache = EndpointCache.load(ENDPOINTS_CACHE_BI_PATH)
    for ep in cache.liars + cache.too_expensive + cache.bad_temperature:
        names.setdefault(
            slugify(f"{ep.model}#{ep.provider}"), display_name(ep.model, ep.provider)
        )
    for entry in cache.unprobeable:
        ep = entry.endpoint
        names.setdefault(
            slugify(f"{ep.model}#{ep.provider}"), display_name(ep.model, ep.provider)
        )
    for ep in discovered:
        names.setdefault(ep.slug, display_name(ep.model, ep.provider))
    return names


def _ordered(groups: dict[str, float]) -> dict[str, float]:
    return {g: groups[g] for g in GROUP_ORDER if g in groups}


def aggregate_spend(spend_dir: Path, today: str, names: dict[str, str]) -> dict:
    """`names` is slug -> "model @ provider" (build_endpoint_names); a slug with
    no entry falls back to the slug itself, e.g. for a stray probe that never
    landed in any registry.
    """
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
        {
            "slug": s,
            "name": names.get(s, s),
            "groups": _ordered(g),
            "total": sum(g.values()),
        }
        for s, g in by_endpoint.items()
    ]
    by_ep.sort(key=lambda r: r["total"], reverse=True)
    return {
        "group_order": GROUP_ORDER,
        "group_label": GROUP_LABEL,
        "cumulative": _ordered(cumulative),
        "last_30d": _ordered(last_30d),
        "daily": [{"date": d, "groups": _ordered(g)} for d, g in sorted(daily.items())],
        "by_endpoint": by_ep,
    }
