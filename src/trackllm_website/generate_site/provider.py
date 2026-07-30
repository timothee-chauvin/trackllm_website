"""Per-provider aggregation: providers/<slug>.json plus the Overview's provider rows.

A provider is the company -- the part of the OpenRouter provider string before
the "/". Its serving variants (fp8, fp4, ...) are separate serving stacks and
drift separately, so they stay visible as rows inside the provider.

Reads only already-generated data (the parsed lt_scores.json series, B3IT
build-time views, changes.json) -- never raw logprobs.
"""

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.feed import build_feed_items
from trackllm_website.generate_site.lt import EndpointInfo, LTData, latest_date
from trackllm_website.generate_site.months import month_range
from trackllm_website.generate_site.naming import base_provider, variant_name
from trackllm_website.generate_site.rates import drift_rate, poisson_interval
from trackllm_website.util import slugify

DAYS_PER_YEAR = 365.25

METHODS = ("lt", "b3it")


def endpoint_years(first: str, last: str) -> float:
    """Monitoring exposure in endpoint-years; a single observation counts as a day."""
    span = (datetime.fromisoformat(last) - datetime.fromisoformat(first)).days
    return max(1, span) / DAYS_PER_YEAR


def _day(dt: datetime) -> str:
    return dt.date().isoformat()


def _method_block(endpoints: int, years: float, changes: int) -> dict:
    rate = drift_rate(changes, years)
    ci = poisson_interval(changes, years) if rate is not None else None
    return {
        "endpoints": endpoints,
        "years": round(years, 2),
        "changes": changes,
        "rate": round(rate, 2) if rate is not None else None,
        "ci": [round(ci[0], 2), round(ci[1], 2)] if ci else None,
    }


class _Span:
    """Per-method exposure accumulator for one provider variant."""

    def __init__(self):
        self.slugs: list[str] = []
        self.spans: list[tuple[str, str]] = []
        self.endpoints: dict[str, int] = dict.fromkeys(METHODS, 0)
        self.changes: dict[str, int] = dict.fromkeys(METHODS, 0)
        self.years: dict[str, float] = {m: 0.0 for m in METHODS}

    def add(self, method: str, span: tuple[str, str]) -> None:
        self.endpoints[method] += 1
        self.years[method] += endpoint_years(*span)

    def add_endpoint(self, slug: str, spans: list[tuple[str, str]]) -> None:
        """One span per endpoint, not per method: the monthly monitoring counts
        are endpoints under observation, and a dual-method endpoint is still one."""
        self.slugs.append(slug)
        if spans:
            self.spans.append((min(s[0] for s in spans), max(s[1] for s in spans)))

    def block(self, method: str) -> dict:
        return _method_block(
            self.endpoints[method], self.years[method], self.changes[method]
        )


def _total_block(accs: list[_Span], method: str) -> dict:
    # fsum, not sum: over an empty list sum() returns int 0, which is not the
    # float exposure the rate helpers are typed for.
    return _method_block(
        sum(a.endpoints[method] for a in accs),
        math.fsum(a.years[method] for a in accs),
        sum(a.changes[method] for a in accs),
    )


def build_provider_views(
    website_dir: Path,
    lt_data: dict[str, LTData],
    lt_endpoints: list[EndpointInfo],
    b3it_views: dict[str, B3ITView],
    endpoint_rows: list[dict],
) -> dict[str, dict]:
    data_dir = website_dir / "data"
    lt_by_slug = {e.slug: e for e in lt_endpoints}
    rows_by_slug = {r["slug"]: r for r in endpoint_rows}

    changes_path = data_dir / "changes.json"
    changes = json.loads(changes_path.read_text()) if changes_path.exists() else []

    lt_span = {
        slug: (_day(d.dates[0]), _day(d.dates[-1])) for slug, d in lt_data.items()
    }

    b3it_span: dict[str, tuple[str, str]] = {}
    for slug, view in b3it_views.items():
        dates = view.tv_series["dates"]
        if dates:
            b3it_span[slug] = (dates[0][:10], dates[-1][:10])

    now = latest_date(lt_data)
    drift_by_slug = {slug: d.drift for slug, d in lt_data.items()}
    items = build_feed_items(changes, drift_by_slug, b3it_views, now) if now else []

    by_provider: dict[str, dict[str, _Span]] = defaultdict(lambda: defaultdict(_Span))
    models: dict[str, set[str]] = defaultdict(set)
    for slug in sorted(set(lt_by_slug) | set(b3it_views)):
        ep = lt_by_slug.get(slug)
        view = b3it_views.get(slug)
        provider = ep.provider if ep else view.provider
        model = ep.model if ep else view.model
        base = base_provider(provider)
        acc = by_provider[base][variant_name(provider)]
        models[base].add(model)
        spans = []
        if slug in lt_span:
            acc.add("lt", lt_span[slug])
            spans.append(lt_span[slug])
        if slug in b3it_span:
            acc.add("b3it", b3it_span[slug])
            spans.append(b3it_span[slug])
        acc.add_endpoint(slug, spans)

    # Counted from the same enriched items the page lists, keyed by the item's own
    # provider string, so a provider's change count can never disagree with the
    # changes shown beneath it. Recomputing from lt_scores.json would: the
    # build-time recompute double-detects some changes on adjacent days, which the
    # canonical merged list does not carry.
    for item in items:
        base = base_provider(item["provider"])
        if base in by_provider:
            by_provider[base][variant_name(item["provider"])].changes[
                item["method"]
            ] += 1

    views: dict[str, dict] = {}
    for base, variants in sorted(by_provider.items()):
        accs = list(variants.values())
        spans = [s for acc in accs for s in acc.spans]
        first = min((s[0] for s in spans), default=None)
        last = max((s[1] for s in spans), default=None)
        months = month_range(first, last) if first and last else []
        slugs = {s for acc in accs for s in acc.slugs}

        variant_out = [
            {
                "name": name,
                "n_endpoints": len(acc.slugs),
                "lt": acc.block("lt"),
                "b3it": acc.block("b3it"),
                "monitoring": [
                    sum(1 for lo, hi in acc.spans if lo[:7] <= m <= hi[:7])
                    for m in months
                ],
            }
            for name, acc in sorted(
                variants.items(), key=lambda kv: (-len(kv[1].slugs), kv[0])
            )
        ]

        views[slugify(base)] = {
            "name": base,
            "slug": slugify(base),
            "n_endpoints": len(slugs),
            "n_models": len(models[base]),
            "n_variants": len(variants),
            "first": first,
            "last": last,
            "months": months,
            "lt": _total_block(accs, "lt"),
            "b3it": _total_block(accs, "b3it"),
            "variants": variant_out,
            # Matched on the raw provider string: the item's providerSlug is the
            # slugified base, and `base` is not.
            "changes": [i for i in items if base_provider(i["provider"]) == base],
            "endpoints": sorted(
                (rows_by_slug[s] for s in slugs if s in rows_by_slug),
                key=lambda r: (-r["nChanges"], r["model"]),
            ),
        }

    # Untracked catalog endpoints join their provider's table after the tracked
    # ones (headline badge rows); they carry no exposure, so every rate/count
    # block above ignores them. Providers with no tracked endpoint get no page.
    for row in endpoint_rows:
        if not row["methods"] and row["providerSlug"] in views:
            views[row["providerSlug"]]["endpoints"].append(row)
    return views


def overview_rows(views: dict[str, dict]) -> list[dict]:
    """Compact provider rows for the Overview's providers section."""
    rows = [
        {
            "name": view["name"],
            "slug": view["slug"],
            "n_endpoints": view["n_endpoints"],
            "n_models": view["n_models"],
            "n_variants": view["n_variants"],
            "lt_years": view["lt"]["years"],
            "lt_changes": view["lt"]["changes"],
            "lt_rate": view["lt"]["rate"],
            "lt_ci": view["lt"]["ci"],
            "b3it_endpoints": view["b3it"]["endpoints"],
            "b3it_years": view["b3it"]["years"],
            "last_change": view["changes"][0]["date"] if view["changes"] else None,
        }
        for view in views.values()
    ]
    # Providers too thin to have a rate sort last, not as a rate of zero.
    rows.sort(
        key=lambda r: (
            -(r["lt_rate"] if r["lt_rate"] is not None else -1),
            -r["lt_years"],
        )
    )
    return rows
