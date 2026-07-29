"""Per-organization orgs/<slug>.html: every model one org publishes, in one table.

An organization is the part of the model string before the "/" -- who trained the
model, as opposed to the provider that serves it (provider.py).

Derived from the already-built model views rather than from the endpoints again, so
an org page can never disagree with the model pages it links to.
"""

from collections import defaultdict

from trackllm_website.util import slugify


def _model_row(model_slug: str, view: dict) -> dict:
    changes = view["changes"]
    return {
        "name": view["model"].split("/")[-1],
        "slug": model_slug,
        "n_endpoints": view["n_endpoints"],
        "n_providers": view["n_providers"],
        "n_changed": view["n_changed"],
        "n_changes": len(changes),
        "last_change": changes[-1]["date"] if changes else None,
        "first": view["date_min"],
        "last": view["date_max"],
    }


def build_org_views(model_views: dict[str, dict]) -> dict[str, dict]:
    by_org: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for model_slug, view in model_views.items():
        by_org[view["org"]].append((model_slug, view))

    views: dict[str, dict] = {}
    for org, entries in by_org.items():
        models = sorted(
            (_model_row(s, v) for s, v in entries),
            key=lambda m: (-m["n_changes"], -m["n_endpoints"], m["name"]),
        )
        # Unioned, not summed: one provider serving three of an org's models is
        # still one provider, whereas each of those is its own endpoint.
        providers = {e["base"] for _, v in entries for e in v["endpoints"]}
        dates = [d for m in models for d in (m["first"], m["last"]) if d]

        views[slugify(org)] = {
            "name": org,
            "slug": slugify(org),
            "n_models": len(models),
            "n_endpoints": sum(m["n_endpoints"] for m in models),
            "n_providers": len(providers),
            "n_changed": sum(1 for m in models if m["n_changes"]),
            "n_changes": sum(m["n_changes"] for m in models),
            "first": min(dates, default=None),
            "last": max(dates, default=None),
            "models": models,
        }
    return views
