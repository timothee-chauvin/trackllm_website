"""Site-wide overview.json: site stats, change feed, endpoint directory.

Provider aggregation lives in provider.py; render.py injects its rows into overview.json.

Reads only already-generated data (the parsed lt_scores.json series, B3IT build-time
views, changes.json, spend.json) -- never raw logprobs.
"""

import json
from datetime import datetime
from pathlib import Path

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.feed import (
    TRACE_LEN,
    build_feed_items,
    downsample_trace,
)
from trackllm_website.generate_site.lt import EndpointInfo, LTData, latest_date
from trackllm_website.util import slugify

RECENT_CHANGE_DAYS = 60
RETIRED_GAP_DAYS = 14

FEED_LT_SIZE = 6
FEED_B3IT_SIZE = 4


def _b3it_status_trace(
    view: B3ITView, now: datetime
) -> tuple[str, list[float], int | None]:
    """Directory row (status/trace/stableDays) for a B3IT-only endpoint.

    Mirrors endpoint.ts::computeStatus (date-gap based), plus the view's own
    retired status -- a B3IT endpoint the pipeline has explicitly retired
    (e.g. delisted, no border inputs) must show as retired even if its last
    tv_series point happens to fall inside RETIRED_GAP_DAYS.
    """
    tv_dates = [datetime.fromisoformat(s) for s in view.tv_series["dates"]]
    tv_values = view.tv_series["values"]
    change_dates = sorted(datetime.fromisoformat(c["date"]) for c in view.changes)

    last_tv_date = tv_dates[-1] if tv_dates else None
    gap_retired = (
        last_tv_date is not None and (now - last_tv_date).days > RETIRED_GAP_DAYS
    )
    if view.status == "retired" or gap_retired:
        status = "retired"
    else:
        last_change_date = change_dates[-1] if change_dates else None
        recent = (
            last_change_date is not None
            and (now - last_change_date).days <= RECENT_CHANGE_DAYS
        )
        status = "changed" if recent else "stable"

    stable_since = (
        change_dates[-1] if change_dates else (tv_dates[0] if tv_dates else None)
    )
    stable_days = (now - stable_since).days if stable_since is not None else None
    trace = downsample_trace(tv_values, TRACE_LEN)
    return status, trace, stable_days


def build_overview(
    website_dir: Path,
    lt_data: dict[str, LTData],
    lt_endpoints: list[EndpointInfo],
    b3it_views: dict[str, B3ITView],
) -> dict:
    data_dir = website_dir / "data"

    lt_by_slug = {e.slug: e for e in lt_endpoints}

    now = latest_date(lt_data)

    changes_path = data_dir / "changes.json"
    changes = json.loads(changes_path.read_text()) if changes_path.exists() else []
    lt_changes = [c for c in changes if c["method"] == "LT"]

    all_slugs = sorted(set(lt_by_slug) | set(b3it_views))
    endpoint_recs = []
    models_set: set[str] = set()

    for slug in all_slugs:
        ep = lt_by_slug.get(slug)
        view = b3it_views.get(slug)
        full_model = ep.model if ep else view.model
        provider = ep.provider if ep else view.provider
        models_set.add(full_model)
        org = full_model.split("/")[0]

        methods = []
        if ep:
            methods.append("lt")
        if view:
            methods.append("b3it")

        trace: list[float] = []
        status = "stable"
        stable_days: int | None = None
        n_lt_changes = 0

        info = lt_data.get(slug)
        if info is not None and now is not None:
            active = (now - info.dates[-1]).days <= RETIRED_GAP_DAYS
            n_lt_changes = len(info.changes)
            last_change_date = (
                info.dates[info.changes[-1]["index"]] if info.changes else None
            )
            recent = (
                last_change_date is not None
                and (now - last_change_date).days <= RECENT_CHANGE_DAYS
            )
            status = "retired" if not active else ("changed" if recent else "stable")
            stable_days = (
                (now - last_change_date).days
                if last_change_date
                else (now - info.dates[0]).days
            )
            trace = downsample_trace([v for _, v in info.drift], TRACE_LEN)
        elif view is not None and now is not None:
            status, trace, stable_days = _b3it_status_trace(view, now)

        n_changes = n_lt_changes + (len(view.changes) if view else 0)
        endpoint_recs.append(
            {
                "slug": slug,
                "model": full_model.split("/")[-1],
                "modelSlug": slugify(full_model),
                "org": org,
                "provider": provider,
                "methods": methods,
                "status": status,
                "stableDays": stable_days,
                "nChanges": n_changes,
                "trace": trace,
            }
        )

    drift_by_slug = {slug: d.drift for slug, d in lt_data.items()}
    all_items = build_feed_items(changes, drift_by_slug, b3it_views, now) if now else []
    lt_items = [i for i in all_items if i["method"] == "lt"][:FEED_LT_SIZE]
    b3it_items = [i for i in all_items if i["method"] == "b3it"][:FEED_B3IT_SIZE]
    feed = sorted(lt_items + b3it_items, key=lambda i: i["iso"], reverse=True)

    spend_path = data_dir / "spend.json"
    spend = json.loads(spend_path.read_text()) if spend_path.exists() else {}

    b3it_starts = [
        e["start"]
        for view in b3it_views.values()
        for e in view.epochs
        if e.get("start")
    ]
    b3it_endpoints = sum(1 for view in b3it_views.values() if view.epochs)
    b3it_monitoring = sum(
        1 for view in b3it_views.values() if view.status == "monitoring"
    )

    def _changes_in_window(lo_days: int, hi_days: int) -> int:
        if now is None:
            return 0
        return sum(
            1
            for c in changes
            if lo_days <= (now - datetime.fromisoformat(c["date"])).days < hi_days
        )

    stats = {
        "endpoints": len(endpoint_recs),
        "providers": len({r["provider"] for r in endpoint_recs}),
        "provider_companies": len({r["provider"].split("/")[0] for r in endpoint_recs}),
        "models": len(models_set),
        "orgs": len({r["org"] for r in endpoint_recs}),
        "changes_total": len(changes),
        "changes_lt": len(lt_changes),
        "changes_b3it": sum(1 for c in changes if c["method"] == "B3IT"),
        "active": sum(1 for r in endpoint_recs if r["status"] != "retired"),
        "changed_endpoints": sum(1 for r in endpoint_recs if r["nChanges"] > 0),
        "changes_30d": _changes_in_window(0, 30),
        "lt_endpoints": len(lt_data),
        "b3it_endpoints": b3it_endpoints,
        "b3it_monitoring": b3it_monitoring,
        "b3it_since": (
            datetime.fromisoformat(min(b3it_starts)).strftime("%b %Y")
            if b3it_starts
            else None
        ),
        "queries": sum(len(d.scores) * d.n_per_test for d in lt_data.values()),
        "since": (
            min(d.dates[0] for d in lt_data.values()).strftime("%b %Y")
            if lt_data
            else None
        ),
        "spend_cumulative": round(sum(spend.get("cumulative", {}).values()), 2),
        "now": now.strftime("%Y-%m-%d") if now else None,
    }

    return {
        "stats": stats,
        "feed": feed,
        "endpoints": endpoint_recs,
    }
