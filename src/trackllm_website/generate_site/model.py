"""Per-model models/<slug>.json: every provider serving a model on one shared timeline.

Which changes exist comes from changes.json, the canonical merged list; how far
each one moved comes from the series (lt_scores.json's drift/drift_dates, the B3IT
build-time views) -- LT drift is already smoothed upstream (lt_drift.py); B3IT tv
comes straight from the view's tv_series.
"""

import json
from collections import defaultdict
from pathlib import Path

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.lt import EndpointInfo, load_lt_data
from trackllm_website.generate_site.naming import base_provider
from trackllm_website.generate_site.peaks import (
    B3IT_PEAK_WINDOW,
    LT_PEAK_WINDOW,
    peak_from,
)
from trackllm_website.generate_site.status import (
    EndpointStatus,
    dominant_headline,
    status_json,
)
from trackllm_website.generate_site.status_io import SiteStatuses
from trackllm_website.util import slugify

TRACE_LEN = 90


def _downsample_pairs(pairs: list[tuple], n: int) -> list[tuple]:
    """Nearest-point downsample to at most n (date, value) pairs, keeping real dates."""
    if len(pairs) <= n:
        return pairs
    return [pairs[i * len(pairs) // n] for i in range(n)]


def _by_method(changes: list[dict], method: str) -> list[dict]:
    return sorted(
        (c for c in changes if c["method"] == method), key=lambda c: c["date"]
    )


def _lt_changes(
    canonical: list[dict], drift_pairs: list[tuple[str, float]]
) -> list[dict]:
    out = []
    for c in canonical:
        day = c["date"][:10]
        level = peak_from(day, drift_pairs, LT_PEAK_WINDOW)
        out.append(
            {
                "date": day,
                "sigma": c["magnitude_display"],
                "drift": round(level or 0.0, 2),
            }
        )
    return out


def _b3it_changes(
    canonical: list[dict], tv_pairs: list[tuple[str, float]]
) -> list[dict]:
    out = []
    for c in canonical:
        day = c["date"][:10]
        peak = peak_from(day, tv_pairs, B3IT_PEAK_WINDOW)
        out.append({"date": day, "peakTV": round(peak or 0.0, 3)})
    return out


def _build_endpoint(
    slug: str,
    ep: EndpointInfo | None,
    provider: str,
    view: B3ITView | None,
    lt_dir: Path,
    canonical: list[dict],
) -> tuple[dict, list[str]]:
    """`canonical` is this endpoint's slice of changes.json -- which changes happened.
    The series only says how far each one moved."""
    methods = []
    if ep is not None:
        methods.append("lt")
    if view is not None:
        methods.append("b3it")

    lt_out = None
    lt = load_lt_data(lt_dir, slug) if ep is not None else None
    if lt is not None:
        drift_pairs = [(d.date().isoformat(), v) for d, v in lt.drift]
        if drift_pairs:
            lt_out = {
                "drift": [list(p) for p in _downsample_pairs(drift_pairs, TRACE_LEN)],
                "changes": _lt_changes(_by_method(canonical, "LT"), drift_pairs),
            }

    b3it_out = None
    if view is not None and view.tv_series["dates"]:
        tv_pairs = [
            (d[:10], v)
            for d, v in zip(view.tv_series["dates"], view.tv_series["values"])
        ]
        b3it_out = {
            "tv": [list(p) for p in _downsample_pairs(tv_pairs, TRACE_LEN)],
            "changes": _b3it_changes(_by_method(canonical, "B3IT"), tv_pairs),
        }

    date_range: list[str] = []
    if lt_out and lt_out["drift"]:
        date_range += [lt_out["drift"][0][0], lt_out["drift"][-1][0]]
    if b3it_out and b3it_out["tv"]:
        date_range += [b3it_out["tv"][0][0], b3it_out["tv"][-1][0]]

    n_changes = (len(lt_out["changes"]) if lt_out else 0) + (
        len(b3it_out["changes"]) if b3it_out else 0
    )
    return (
        {
            "slug": slug,
            "provider": provider,
            "base": base_provider(provider),
            "providerSlug": slugify(base_provider(provider)),
            "methods": methods,
            "first": min(date_range) if date_range else None,
            "last": max(date_range) if date_range else None,
            "n_changes": n_changes,
            "lt": lt_out,
            "b3it": b3it_out,
        },
        date_range,
    )


def _untracked_endpoint(slug: str, provider: str, st: EndpointStatus) -> dict:
    """A model-page row for an endpoint with no series: badges, no chart data."""
    return {
        "slug": slug,
        "provider": provider,
        "base": base_provider(provider),
        "providerSlug": slugify(base_provider(provider)),
        "methods": [],
        "first": None,
        "last": None,
        "n_changes": 0,
        "lt": None,
        "b3it": None,
        "status": status_json(st),
    }


def build_model_views(
    website_dir: Path,
    lt_endpoints: list[EndpointInfo],
    b3it_views: dict[str, B3ITView],
    site: SiteStatuses,
) -> dict[str, dict]:
    data_dir = website_dir / "data"
    lt_dir = data_dir / "lt"

    changes_path = data_dir / "changes.json"
    canonical = json.loads(changes_path.read_text()) if changes_path.exists() else []
    canonical_by_slug: dict[str, list[dict]] = defaultdict(list)
    for c in canonical:
        canonical_by_slug[c["slug"]].append(c)

    lt_by_slug = {e.slug: e for e in lt_endpoints}

    slugs_by_model: dict[str, list[str]] = defaultdict(list)
    for slug in sorted(set(site.statuses) | set(lt_by_slug) | set(b3it_views)):
        ep = lt_by_slug.get(slug)
        view = b3it_views.get(slug)
        model = ep.model if ep else (view.model if view else site.names[slug][0])
        slugs_by_model[model].append(slug)

    out: dict[str, dict] = {}
    for model, slugs in slugs_by_model.items():
        endpoints = []
        alldates: list[str] = []
        for slug in slugs:
            ep = lt_by_slug.get(slug)
            view = b3it_views.get(slug)
            st = site.statuses[slug]
            if ep is None and view is None:
                endpoints.append(_untracked_endpoint(slug, site.names[slug][1], st))
                continue
            provider = ep.provider if ep else view.provider
            rec, date_range = _build_endpoint(
                slug, ep, provider, view, lt_dir, canonical_by_slug[slug]
            )
            rec["status"] = status_json(st)
            endpoints.append(rec)
            alldates += date_range
        endpoints.sort(key=lambda e: (not e["methods"], -e["n_changes"], e["provider"]))
        tracked = [e for e in endpoints if e["methods"]]

        # every change for the model, so the page can draw one all-providers strip
        changes = sorted(
            [
                {"date": c["date"], "method": "lt", "provider": e["provider"]}
                for e in endpoints
                if e["lt"]
                for c in e["lt"]["changes"]
            ]
            + [
                {"date": c["date"], "method": "b3it", "provider": e["provider"]}
                for e in endpoints
                if e["b3it"]
                for c in e["b3it"]["changes"]
            ],
            key=lambda c: c["date"],
        )

        drift_values = [
            v for e in endpoints if e["lt"] for _, v in e["lt"]["drift"]
        ] + [c["drift"] for e in endpoints if e["lt"] for c in e["lt"]["changes"]]
        max_drift = round(max(drift_values, default=0.0), 2)

        headlines = [site.statuses[s].headline for s in slugs]
        # "trackable" = at least one method could ever work, not "being tracked"
        n_trackable = sum(1 for h in headlines if h != "untrackable")
        n_total = len(slugs)

        out[slugify(model)] = {
            "model": model,
            "org": model.split("/")[0],
            "date_min": min(alldates) if alldates else None,
            "date_max": max(alldates) if alldates else None,
            # Endpoints are serving variants: two of them can be the same company
            # (chutes and chutes/fp8), so the two counts are not interchangeable.
            # The n_endpoints/n_providers counts describe the tracked fleet;
            # n_endpoints_total spans the whole catalog for this model.
            "n_endpoints": len(tracked),
            "n_providers": len({e["base"] for e in tracked}),
            "n_endpoints_total": n_total,
            "n_changed": sum(1 for e in tracked if e["n_changes"]),
            "max_drift": max_drift,
            "headline": dominant_headline(headlines),
            "status_summary": (
                f"{n_trackable} of {n_total} "
                f"endpoint{'s' if n_total != 1 else ''} trackable"
            ),
            "changes": changes,
            "endpoints": endpoints,
        }
    return out
