"""Per-model models/<slug>.json: every provider serving a model on one shared timeline.

Reads only already-generated data (lt_scores.json's drift/drift_dates, B3IT build-time
views) -- LT drift is already smoothed upstream (lt_drift.py); B3IT tv comes straight
from the view's tv_series.
"""

from collections import defaultdict
from pathlib import Path

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.lt import EndpointInfo, LTData, load_lt_data
from trackllm_website.generate_site.naming import base_provider
from trackllm_website.lt_scores import normalize_sigma
from trackllm_website.util import slugify

TRACE_LEN = 90
LT_PEAK_WINDOW = 20
B3IT_PEAK_WINDOW = 8


def _downsample_pairs(pairs: list[tuple], n: int) -> list[tuple]:
    """Nearest-point downsample to at most n (date, value) pairs, keeping real dates."""
    if len(pairs) <= n:
        return pairs
    return [pairs[i * len(pairs) // n] for i in range(n)]


def _sigma_display(sigma: float | None) -> str:
    if sigma is None or normalize_sigma(sigma) is None:
        return "∞σ"
    return f"{sigma:.0f}σ"


def _peak_from(day: str, pairs: list[tuple[str, float]], window: int) -> float | None:
    """Peak value from the first point on/after `day`, over the next `window` points."""
    on_or_after = [v for d, v in pairs if d >= day][:window]
    if on_or_after:
        return max(on_or_after)
    same_day = [v for d, v in pairs if d == day]
    return same_day[-1] if same_day else None


def _lt_changes(lt: LTData, drift_pairs: list[tuple[str, float]]) -> list[dict]:
    out = []
    for c in lt.changes:
        day = lt.dates[c["index"]].date().isoformat()
        level = _peak_from(day, drift_pairs, LT_PEAK_WINDOW)
        out.append(
            {
                "date": day,
                "sigma": _sigma_display(c["sigma"]),
                "drift": round(level or 0.0, 2),
            }
        )
    return out


def _b3it_changes(view: B3ITView, tv_pairs: list[tuple[str, float]]) -> list[dict]:
    out = []
    for ch in view.changes:
        day = ch["date"][:10]
        peak = _peak_from(day, tv_pairs, B3IT_PEAK_WINDOW)
        out.append({"date": day, "peakTV": round(peak or 0.0, 3)})
    return out


def _build_endpoint(
    slug: str,
    ep: EndpointInfo | None,
    provider: str,
    view: B3ITView | None,
    lt_dir: Path,
) -> tuple[dict, list[str]]:
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
                "changes": _lt_changes(lt, drift_pairs),
            }

    b3it_out = None
    if view is not None and view.tv_series["dates"]:
        tv_pairs = [
            (d[:10], v)
            for d, v in zip(view.tv_series["dates"], view.tv_series["values"])
        ]
        b3it_out = {
            "tv": [list(p) for p in _downsample_pairs(tv_pairs, TRACE_LEN)],
            "changes": _b3it_changes(view, tv_pairs),
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


def build_model_views(
    website_dir: Path,
    lt_endpoints: list[EndpointInfo],
    b3it_views: dict[str, B3ITView],
) -> dict[str, dict]:
    data_dir = website_dir / "data"
    lt_dir = data_dir / "lt"

    lt_by_slug = {e.slug: e for e in lt_endpoints}

    slugs_by_model: dict[str, list[str]] = defaultdict(list)
    for slug in sorted(set(lt_by_slug) | set(b3it_views)):
        ep = lt_by_slug.get(slug)
        view = b3it_views.get(slug)
        model = ep.model if ep else view.model
        slugs_by_model[model].append(slug)

    out: dict[str, dict] = {}
    for model, slugs in slugs_by_model.items():
        endpoints = []
        alldates: list[str] = []
        for slug in slugs:
            ep = lt_by_slug.get(slug)
            view = b3it_views.get(slug)
            provider = ep.provider if ep else view.provider
            rec, date_range = _build_endpoint(slug, ep, provider, view, lt_dir)
            endpoints.append(rec)
            alldates += date_range
        endpoints.sort(key=lambda e: (-e["n_changes"], e["provider"]))

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

        out[slugify(model)] = {
            "model": model,
            "org": model.split("/")[0],
            "date_min": min(alldates) if alldates else None,
            "date_max": max(alldates) if alldates else None,
            "n_providers": len(endpoints),
            "n_changed": sum(1 for e in endpoints if e["n_changes"]),
            "max_drift": max_drift,
            "changes": changes,
            "endpoints": endpoints,
        }
    return out
