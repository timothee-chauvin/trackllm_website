"""The shared timeline: every endpoint in a group (a model's providers, a
provider's models) as one strip per endpoint on one date axis.

Which changes exist comes from changes.json, the canonical merged list; how far
each one moved comes from the series (lt_scores.json's drift/drift_dates, the B3IT
build-time views) -- LT drift is already smoothed upstream (lt_drift.py); B3IT tv
comes straight from the view's tv_series.
"""

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.freshness import latest
from trackllm_website.generate_site.lt import EndpointInfo, load_lt_data
from trackllm_website.generate_site.naming import base_provider
from trackllm_website.generate_site.peaks import (
    B3IT_PEAK_WINDOW,
    LT_PEAK_WINDOW,
    peak_from,
)
from trackllm_website.generate_site.status import EndpointStatus, status_json
from trackllm_website.generate_site.status_io import SiteStatuses
from trackllm_website.util import slugify

TRACE_LEN = 90


def load_changes(data_dir: Path) -> list[dict]:
    """The canonical merged change list render.py writes before the views build."""
    path = data_dir / "changes.json"
    return json.loads(path.read_text()) if path.exists() else []


def changes_by_slug(changes: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for c in changes:
        out[c["slug"]].append(c)
    return out


def _downsample_pairs(pairs: list[tuple], n: int) -> list[tuple]:
    """Nearest-point downsample to at most n (date, value) pairs, keeping real dates."""
    if len(pairs) <= n:
        return pairs
    return [pairs[i * len(pairs) // n] for i in range(n)]


def _by_method(changes: list[dict], method: str) -> list[dict]:
    return sorted(
        (c for c in changes if c["method"] == method), key=lambda c: c["date"]
    )


# Both helpers below publish None, never 0.0, for a change the series has no point
# on or after (one dated after the last observation): the level it reached is
# unknown, and a published 0.00 would read as a change that moved nothing. feed.py
# publishes the same null; timeline.ts and endpoint.ts render it as an em dash.
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
                "drift": round(level, 2) if level is not None else None,
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
        out.append(
            {"date": day, "peakTV": round(peak, 3) if peak is not None else None}
        )
    return out


def _last_query(ep: EndpointInfo | None, view: B3ITView | None) -> str | None:
    """The UTC day this endpoint last answered us, over both methods: LT records
    the last non-error query, B3IT the last phase-2 sample (from the raw results,
    so a freshly re-initialized endpoint counts as alive with an empty series).

    Truncated to the day on purpose: everything queried in the same round is
    equally alive, and ordering those by the second would hand the top of the page
    to whichever endpoint the round happened to reach first.
    """
    instant = latest(
        [ep.last_query_date if ep else None, view.last_query if view else None]
    )
    return instant[:10] if instant else None


def _staleness(day: str | None) -> int:
    """Row rank, most recently alive first. A real day ranks as its negated
    ordinal, so the 0 of an endpoint that never answered sorts after all of them."""
    return -date.fromisoformat(day).toordinal() if day else 0


def _naming(model: str, provider: str) -> dict:
    return {
        "provider": provider,
        "base": base_provider(provider),
        "providerSlug": slugify(base_provider(provider)),
        "model": model,
        "modelSlug": slugify(model),
    }


def _build_endpoint(
    slug: str,
    ep: EndpointInfo | None,
    model: str,
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
        # The changes are published whether or not the drift lane has points to
        # level them against: an empty lane (lt_drift.py yields none under three
        # distinct observation days) makes each level unknown, not each change
        # nonexistent. Dropping them would leave the endpoint page and the
        # directory row disagreeing about how many changes this endpoint has.
        drift_pairs = [(d.date().isoformat(), v) for d, v in lt.drift]
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
    # The changes are on the axis too, and a change can fall outside the observed
    # span (one recorded after the endpoint's last sampled point). timeline.ts maps
    # dates onto date_min..date_max, so leaving it out puts its mark -- the dashed
    # "level unknown" rule -- outside the viewBox, where it is simply invisible.
    for lane in (lt_out, b3it_out):
        date_range += [c["date"] for c in lane["changes"]] if lane else []

    n_changes = (len(lt_out["changes"]) if lt_out else 0) + (
        len(b3it_out["changes"]) if b3it_out else 0
    )
    return (
        {
            "slug": slug,
            **_naming(model, provider),
            "methods": methods,
            "first": min(date_range) if date_range else None,
            "last": max(date_range) if date_range else None,
            "last_query": _last_query(ep, view),
            "n_changes": n_changes,
            "lt": lt_out,
            "b3it": b3it_out,
        },
        date_range,
    )


def _untracked_endpoint(
    slug: str, model: str, provider: str, st: EndpointStatus
) -> dict:
    """A timeline row for an endpoint with no series: badges, no chart data."""
    return {
        "slug": slug,
        **_naming(model, provider),
        "methods": [],
        "first": None,
        "last": None,
        "last_query": None,
        "n_changes": 0,
        "lt": None,
        "b3it": None,
        "status": status_json(st),
    }


def build_timeline(
    slugs: list[str],
    lt_by_slug: dict[str, EndpointInfo],
    b3it_views: dict[str, B3ITView],
    site: SiteStatuses,
    lt_dir: Path,
    canonical_by_slug: dict[str, list[dict]],
) -> dict:
    """One shared-axis timeline over `slugs`: per-endpoint strips plus the
    flattened change list the all-endpoints strip draws."""
    endpoints: list[dict] = []
    alldates: list[str] = []
    for slug in slugs:
        ep = lt_by_slug.get(slug)
        view = b3it_views.get(slug)
        st = site.statuses[slug]
        if ep is None and view is None:
            endpoints.append(_untracked_endpoint(slug, *site.names[slug], st))
            continue
        model = ep.model if ep else view.model
        provider = ep.provider if ep else view.provider
        rec, date_range = _build_endpoint(
            slug, ep, model, provider, view, lt_dir, canonical_by_slug.get(slug, [])
        )
        rec["status"] = status_json(st)
        endpoints.append(rec)
        alldates += date_range
    # Freshest first: a page opening on a pile of retired endpoints says nothing
    # about the fleet it is meant to describe. Endpoints that never answered (and
    # the badge-only catalog rows) sort last, then the change count breaks ties.
    endpoints.sort(
        key=lambda e: (
            not e["methods"],
            _staleness(e["last_query"]),
            -e["n_changes"],
            e["provider"],
        )
    )

    changes = sorted(
        [
            {
                "date": c["date"],
                "method": method,
                "provider": e["provider"],
                "model": e["model"],
            }
            for e in endpoints
            for method in ("lt", "b3it")
            if e[method]
            for c in e[method]["changes"]
        ],
        key=lambda c: c["date"],
    )

    return {
        "date_min": min(alldates) if alldates else None,
        "date_max": max(alldates) if alldates else None,
        "changes": changes,
        "endpoints": endpoints,
    }
