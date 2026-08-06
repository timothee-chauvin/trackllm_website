"""Per-model models/<slug>.json: every provider serving a model on one shared
timeline (timeline.py), plus the model-level aggregates its page shows."""

from collections import defaultdict
from pathlib import Path

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.lt import EndpointInfo
from trackllm_website.generate_site.status import dominant_headline
from trackllm_website.generate_site.status_io import SiteStatuses
from trackllm_website.generate_site.timeline import (
    build_timeline,
    changes_by_slug,
    load_changes,
)
from trackllm_website.util import slugify


def build_model_views(
    website_dir: Path,
    lt_endpoints: list[EndpointInfo],
    b3it_views: dict[str, B3ITView],
    site: SiteStatuses,
) -> dict[str, dict]:
    data_dir = website_dir / "data"
    lt_dir = data_dir / "lt"
    canonical = changes_by_slug(load_changes(data_dir))
    lt_by_slug = {e.slug: e for e in lt_endpoints}

    slugs_by_model: dict[str, list[str]] = defaultdict(list)
    for slug in sorted(set(site.statuses) | set(lt_by_slug) | set(b3it_views)):
        ep = lt_by_slug.get(slug)
        view = b3it_views.get(slug)
        model = ep.model if ep else (view.model if view else site.names[slug][0])
        slugs_by_model[model].append(slug)

    out: dict[str, dict] = {}
    for model, slugs in slugs_by_model.items():
        timeline = build_timeline(
            slugs, lt_by_slug, b3it_views, site, lt_dir, canonical
        )
        endpoints = timeline["endpoints"]
        tracked = [e for e in endpoints if e["methods"]]

        drift_values = [
            v for e in endpoints if e["lt"] for _, v in e["lt"]["drift"]
        ] + [
            c["drift"]
            for e in endpoints
            if e["lt"]
            for c in e["lt"]["changes"]
            if c["drift"] is not None
        ]
        max_drift = round(max(drift_values, default=0.0), 2)

        headlines = [site.statuses[s].headline for s in slugs]
        # "trackable" = at least one method could ever work, not "being tracked"
        n_trackable = sum(1 for h in headlines if h != "untrackable")
        n_total = len(slugs)

        out[slugify(model)] = {
            "model": model,
            "org": model.split("/")[0],
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
            **timeline,
        }
    return out
