"""Site-wide overview.json: change feed, endpoint directory, provider drift rates.

Reads only already-generated data (lt_scores.json's drift/drift_dates, B3IT build-time
views, changes.json, spend.json) -- never raw logprobs.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.lt import EndpointInfo, load_lt_scores

TRACE_LEN = 28
RECENT_CHANGE_DAYS = 60
RETIRED_GAP_DAYS = 14

FEED_LT_SIZE = 6
FEED_B3IT_SIZE = 4
FEED_TRACE_LEN = 40
FEED_WINDOW_BEFORE = 60
FEED_WINDOW_AFTER = 20
FEED_MIN_WINDOW = 6
FEED_PEAK_WINDOW = 20
FEED_DEFAULT_CHANGE_FRAC = 0.5
LT_ALERT_THRESHOLD = 0.8
B3IT_ALERT_THRESHOLD = 0.6
CHANGED_THRESHOLD = 0.3

PROVIDER_MIN_ENDPOINT_YEARS = 0.5
PROVIDER_CONF_FLOOR = 0.3
PROVIDER_CONF_FULL_YEARS = 4.0


def downsample_trace(vals: list[float | int], n: int) -> list[float]:
    """Bucket-mean downsample to at most n points, rounded to 3 decimals."""
    if not vals:
        return []
    if len(vals) <= n:
        return [round(float(v), 3) for v in vals]
    out = []
    for b in range(n):
        lo = b * len(vals) // n
        hi = (b + 1) * len(vals) // n
        chunk = vals[lo:hi] or [vals[min(lo, len(vals) - 1)]]
        out.append(round(sum(chunk) / len(chunk), 3))
    return out


@dataclass
class _LTData:
    dates: list[datetime]
    scores: list[float]
    n_per_test: int
    changes: list[dict]
    drift: list[tuple[datetime, float]]


def _load_lt_data(lt_dir: Path, slug: str) -> _LTData | None:
    d = load_lt_scores(lt_dir, slug)
    if d is None:
        return None
    drift_dates = [datetime.fromisoformat(s) for s in d.get("drift_dates", [])]
    drift = list(zip(drift_dates, d.get("drift", [])))
    return _LTData(
        dates=[datetime.fromisoformat(s) for s in d["dates"]],
        scores=d["scores"],
        n_per_test=d["n_per_test"],
        changes=d["changes"],
        drift=drift,
    )


def _nearest_index(pairs: list[tuple[datetime, float]], target: datetime) -> int:
    return min(
        range(len(pairs)), key=lambda i: abs((pairs[i][0] - target).total_seconds())
    )


def _feed_window(
    pairs: list[tuple[datetime, float]], k: int
) -> tuple[list[float], float]:
    lo = max(0, k - FEED_WINDOW_BEFORE)
    hi = min(len(pairs), k + FEED_WINDOW_AFTER)
    window = [v for _, v in pairs[lo:hi]]
    if len(window) < FEED_MIN_WINDOW:
        return [], FEED_DEFAULT_CHANGE_FRAC
    return downsample_trace(window, FEED_TRACE_LEN), round((k - lo) / (hi - lo), 3)


def _build_lt_feed_item(
    change: dict, drift: list[tuple[datetime, float]], now: datetime
) -> dict:
    cd = datetime.fromisoformat(change["date"])
    drift_at = None
    ftrace: list[float] = []
    cfrac = FEED_DEFAULT_CHANGE_FRAC
    if drift:
        k = _nearest_index(drift, cd)
        peak_hi = min(len(drift), k + FEED_PEAK_WINDOW)
        drift_at = round(max(v for _, v in drift[k:peak_hi]), 2)
        ftrace, cfrac = _feed_window(drift, k)
    sev = (
        "alert"
        if (drift_at or 0) >= LT_ALERT_THRESHOLD
        else "changed"
        if (drift_at or 0) >= CHANGED_THRESHOLD
        else "stable"
    )
    drift_display = drift_at if drift_at is not None else "—"
    return {
        "date": change["date"][:10],
        "iso": change["date"],
        "daysAgo": (now - cd).days,
        "model": change["model"].split("/")[-1],
        "provider": change["provider"],
        "method": "lt",
        "desc": f"Logprob averages moved {drift_display} nats from the reference period.",
        "primary": f"drift {drift_display}",
        "secondary": f"{change['magnitude_display']} conf",
        "sevKey": sev,
        "trace": ftrace,
        "changeFrac": cfrac,
    }


def _build_b3it_feed_item(view: B3ITView, change: dict, now: datetime) -> dict:
    pairs = list(
        zip(
            (datetime.fromisoformat(s) for s in view.tv_series["dates"]),
            view.tv_series["values"],
        )
    )
    cd = datetime.fromisoformat(change["date"])
    peak = 0.0
    ftrace: list[float] = []
    cfrac = FEED_DEFAULT_CHANGE_FRAC
    if pairs:
        k = _nearest_index(pairs, cd)
        peak_hi = min(len(pairs), k + FEED_PEAK_WINDOW)
        peak = round(max(v for _, v in pairs[k:peak_hi]), 3)
        ftrace, cfrac = _feed_window(pairs, k)
    sev = (
        "alert"
        if peak >= B3IT_ALERT_THRESHOLD
        else "changed"
        if peak >= CHANGED_THRESHOLD
        else "stable"
    )
    return {
        "date": change["date"][:10],
        "iso": change["date"],
        "daysAgo": (now - cd).days,
        "model": view.model.split("/")[-1],
        "provider": view.provider,
        "method": "b3it",
        "desc": f"Border-input output distribution moved (TV {peak:.2f}) from the reference.",
        "primary": f"TV {peak:.2f}",
        "secondary": "border-input shift",
        "sevKey": sev,
        "trace": ftrace,
        "changeFrac": cfrac,
    }


def _build_feed(
    lt_changes: list[dict],
    lt_data: dict[str, _LTData],
    b3it_views: dict[str, B3ITView],
    now: datetime | None,
) -> list[dict]:
    if now is None:
        return []
    feed = []
    top_lt = sorted(lt_changes, key=lambda c: c["date"], reverse=True)[:FEED_LT_SIZE]
    for c in top_lt:
        info = lt_data.get(c["slug"])
        feed.append(_build_lt_feed_item(c, info.drift if info else [], now))

    b3it_changes = sorted(
        ((view, ch) for view in b3it_views.values() for ch in view.changes),
        key=lambda t: t[1]["date"],
        reverse=True,
    )[:FEED_B3IT_SIZE]
    for view, ch in b3it_changes:
        feed.append(_build_b3it_feed_item(view, ch, now))

    feed.sort(key=lambda f: f["iso"], reverse=True)
    return feed


def build_overview(
    website_dir: Path,
    lt_endpoints: list[EndpointInfo],
    b3it_views: dict[str, B3ITView],
) -> dict:
    data_dir = website_dir / "data"
    lt_dir = data_dir / "lt"

    lt_by_slug = {e.slug: e for e in lt_endpoints}

    lt_data: dict[str, _LTData] = {}
    for slug in lt_by_slug:
        d = _load_lt_data(lt_dir, slug)
        if d is not None:
            lt_data[slug] = d

    now = max((d.dates[-1] for d in lt_data.values()), default=None)

    changes_path = data_dir / "changes.json"
    changes = json.loads(changes_path.read_text()) if changes_path.exists() else []
    lt_changes = [c for c in changes if c["method"] == "LT"]

    all_slugs = sorted(set(lt_by_slug) | set(b3it_views))
    endpoint_recs = []
    provider_stats: dict[str, dict] = defaultdict(
        lambda: {"endpoint_days": 0.0, "n_endpoints": 0, "n_changes": 0}
    )
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
            ps = provider_stats[provider]
            ps["endpoint_days"] += max(
                1.0, (info.dates[-1] - info.dates[0]).total_seconds() / 86400
            )
            ps["n_endpoints"] += 1

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

        n_changes = n_lt_changes + (len(view.changes) if view else 0)
        endpoint_recs.append(
            {
                "slug": slug,
                "model": full_model.split("/")[-1],
                "org": org,
                "provider": provider,
                "methods": methods,
                "status": status,
                "stableDays": stable_days,
                "nChanges": n_changes,
                "trace": trace,
            }
        )

    for c in lt_changes:
        if c["provider"]:
            provider_stats[c["provider"]]["n_changes"] += 1

    providers = []
    for prov, s in provider_stats.items():
        ey = s["endpoint_days"] / 365.25
        if ey < PROVIDER_MIN_ENDPOINT_YEARS:
            continue
        providers.append(
            {
                "name": prov,
                "n_endpoints": s["n_endpoints"],
                "endpoint_years": round(ey, 2),
                "months": round(ey * 12),
                "n_changes": s["n_changes"],
                "rate": round(s["n_changes"] / ey, 2) if ey else 0,
                "conf": round(
                    PROVIDER_CONF_FLOOR
                    + (1 - PROVIDER_CONF_FLOOR)
                    * min(1.0, ey / PROVIDER_CONF_FULL_YEARS),
                    3,
                ),
            }
        )
    providers.sort(key=lambda x: (-x["rate"], -x["endpoint_years"]))

    feed = _build_feed(lt_changes, lt_data, b3it_views, now)

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
        "providers": providers,
        "endpoints": endpoint_recs,
    }
