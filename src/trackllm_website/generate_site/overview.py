"""Site-wide overview.json: site stats, change feed, endpoint directory.

Provider aggregation lives in provider.py; render.py injects its rows into overview.json.

Reads only already-generated data (the parsed lt_scores.json series, B3IT build-time
views, changes.json, spend.json) -- never raw logprobs.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from trackllm_website.config import HeroConfig
from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.clock import site_now
from trackllm_website.generate_site.feed import (
    TRACE_LEN,
    build_feed_items,
    downsample_trace,
)
from trackllm_website.generate_site.freshness import latest
from trackllm_website.generate_site.hero import build_hero
from trackllm_website.generate_site.lt import EndpointInfo, LTData
from trackllm_website.generate_site.naming import base_provider
from trackllm_website.generate_site.status import EndpointStatus, one_line_reason
from trackllm_website.generate_site.status_io import SiteStatuses
from trackllm_website.util import slugify

RECENT_CHANGE_DAYS = 60
RETIRED_GAP_DAYS = 14

FEED_LT_SIZE = 6
FEED_B3IT_SIZE = 4


def _row_state(
    now: datetime,
    obs_dates: list[datetime],
    change_dates: list[datetime],
    values: list[float],
    retired: bool,
) -> tuple[str, list[float], int | None]:
    """Directory row (status/trace/stableDays) for one endpoint.

    Mirrors endpoint.ts::computeStatus (date-gap based). `change_dates` is the
    endpoint's slice of the canonical merged list -- the same one the row's
    nChanges counts, so the row can never read "stable for N days" beside a
    nonzero count. `retired` is the pipeline's own verdict, independent of the
    observation gap.
    """
    last_obs = obs_dates[-1] if obs_dates else None
    gap_retired = last_obs is not None and (now - last_obs).days > RETIRED_GAP_DAYS
    if retired or gap_retired:
        status = "retired"
    else:
        last_change = change_dates[-1] if change_dates else None
        recent = (
            last_change is not None and (now - last_change).days <= RECENT_CHANGE_DAYS
        )
        status = "changed" if recent else "stable"

    stable_since = (
        change_dates[-1] if change_dates else (obs_dates[0] if obs_dates else None)
    )
    stable_days = (now - stable_since).days if stable_since is not None else None
    return status, downsample_trace(values, TRACE_LEN), stable_days


def _b3it_row_state(
    view: B3ITView, now: datetime, change_dates: list[datetime]
) -> tuple[str, list[float], int | None]:
    """Directory row for a B3IT-only endpoint: its tv_series is the trace.

    The view's own retired status is load-bearing -- a B3IT endpoint the pipeline
    has explicitly retired (e.g. delisted, no border inputs) must show as retired
    even if its last tv_series point happens to fall inside RETIRED_GAP_DAYS.
    """
    tv_dates = [datetime.fromisoformat(s) for s in view.tv_series["dates"]]
    return _row_state(
        now,
        tv_dates,
        change_dates,
        view.tv_series["values"],
        view.status == "retired",
    )


def _status_fields(st: EndpointStatus) -> dict:
    return {
        "headline": st.headline,
        "ltStatus": st.lt,
        "biStatus": st.bi,
        "reason": one_line_reason(st),
    }


def _untracked_row(slug: str, site: SiteStatuses) -> dict:
    """A directory row for an endpoint with no series: a status in place of a trace."""
    model, provider = site.names[slug]
    return {
        "slug": slug,
        "model": model.split("/")[-1],
        "modelSlug": slugify(model),
        "org": model.split("/")[0],
        "provider": provider,
        "providerSlug": slugify(base_provider(provider)),
        "methods": [],
        "status": None,
        "stableDays": None,
        "nChanges": 0,
        "trace": [],
        **_status_fields(site.statuses[slug]),
    }


def build_overview(
    website_dir: Path,
    lt_data: dict[str, LTData],
    lt_endpoints: list[EndpointInfo],
    b3it_views: dict[str, B3ITView],
    hero_pin: HeroConfig | None,
    site: SiteStatuses,
) -> dict:
    data_dir = website_dir / "data"

    lt_by_slug = {e.slug: e for e in lt_endpoints}

    now = site_now(lt_data, b3it_views)

    changes_path = data_dir / "changes.json"
    changes = json.loads(changes_path.read_text()) if changes_path.exists() else []
    lt_changes = [c for c in changes if c["method"] == "LT"]
    # Canonical per-endpoint changes -- both the count and the status/stableDays
    # each row publishes. Never the changes recomputed into lt_scores.json: that
    # recompute double-detects some changes on adjacent days, and the directory
    # row sits on the same pages as the merged change list (the Overview feed,
    # the provider tables, the Changes board).
    change_dates: dict[str, list[datetime]] = defaultdict(list)
    for c in changes:
        change_dates[c["slug"]].append(datetime.fromisoformat(c["date"]))
    for dates in change_dates.values():
        dates.sort()

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

        info = lt_data.get(slug)
        if info is not None and now is not None:
            status, trace, stable_days = _row_state(
                now,
                info.dates,
                change_dates[slug],
                [v for _, v in info.drift],
                retired=False,
            )
        elif view is not None and now is not None:
            status, trace, stable_days = _b3it_row_state(view, now, change_dates[slug])

        endpoint_recs.append(
            {
                "slug": slug,
                "model": full_model.split("/")[-1],
                "modelSlug": slugify(full_model),
                "org": org,
                "provider": provider,
                # The slug provider pages are written under, so the link can never 404.
                "providerSlug": slugify(base_provider(provider)),
                "methods": methods,
                "status": status,
                "stableDays": stable_days,
                "nChanges": len(change_dates[slug]),
                "trace": trace,
                **_status_fields(site.statuses[slug]),
            }
        )

    drift_by_slug = {slug: d.drift for slug, d in lt_data.items()}
    all_items = build_feed_items(changes, drift_by_slug, b3it_views, now) if now else []
    lt_items = [i for i in all_items if i["method"] == "lt"][:FEED_LT_SIZE]
    b3it_items = [i for i in all_items if i["method"] == "b3it"][:FEED_B3IT_SIZE]
    feed = sorted(lt_items + b3it_items, key=lambda i: i["iso"], reverse=True)
    # None only where a caller has no hero to draw (fixtures); the site build always
    # passes config.hero, and a pin that cannot resolve raises rather than blanking.
    hero = (
        build_hero(changes, drift_by_slug, b3it_views, now, hero_pin)
        if now and hero_pin
        else None
    )

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

    # Catalog/previously-tracked endpoints with no series: rows with a status in
    # place of a trace.
    untracked_recs = [
        _untracked_row(slug, site)
        for slug in sorted(set(site.statuses) - set(all_slugs))
    ]
    # The headline (status.py) is what the directory's status chips filter on, so
    # the headline numbers up top must be counted the same way -- over every row,
    # including the ones with no series. A row's `status` is a display state read
    # off the trace's date gaps: an endpoint whose queries have started failing
    # still reads "stable" for two weeks, and one we monitor but have not plotted
    # yet has no `status` at all, so counting it here would disagree with the
    # "Tracked" chip right below.
    headlines = Counter(r["headline"] for r in endpoint_recs + untracked_recs)

    stats = {
        # the fleet we have ever tracked: everything the two chips below cover
        "endpoints": headlines["tracked"] + headlines["retired"],
        "providers": len({r["provider"] for r in endpoint_recs}),
        "provider_companies": len(
            {base_provider(r["provider"]) for r in endpoint_recs}
        ),
        "models": len(models_set),
        "orgs": len({r["org"] for r in endpoint_recs}),
        "changes_total": len(changes),
        "changes_lt": len(lt_changes),
        "changes_b3it": sum(1 for c in changes if c["method"] == "B3IT"),
        "active": headlines["tracked"],
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
        # Absolute instants, not ages: overview.ts turns them into "14m ago" at
        # page load, so a stale build cannot claim to be fresh.
        "last_query_lt": latest(e.last_query_date for e in lt_endpoints),
        "last_query_b3it": latest(v.last_query for v in b3it_views.values()),
    }

    return {
        "stats": stats,
        "hero": hero,
        "feed": feed,
        "endpoints": endpoint_recs + untracked_recs,
    }
