"""Change-feed enrichment: magnitude, sparkline window and link slugs per change.

Shared by the Overview (its latest slice) and the Changes page (all of them), so
the two never disagree about what a change looked like.
"""

from datetime import datetime

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.naming import base_provider
from trackllm_website.util import slugify

TRACE_LEN = 28

FEED_TRACE_LEN = 40
FEED_WINDOW_BEFORE = 60
FEED_WINDOW_AFTER = 20
FEED_MIN_WINDOW = 6
FEED_LT_PEAK_WINDOW = 20
FEED_B3IT_PEAK_WINDOW = 8
FEED_DEFAULT_CHANGE_FRAC = 0.5
LT_ALERT_THRESHOLD = 0.8
B3IT_ALERT_THRESHOLD = 0.6
CHANGED_THRESHOLD = 0.3


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


def _nearest_index(pairs: list[tuple[datetime, float]], target: datetime) -> int:
    return min(
        range(len(pairs)), key=lambda i: abs((pairs[i][0] - target).total_seconds())
    )


def _window(pairs: list[tuple[datetime, float]], k: int) -> tuple[list[float], float]:
    lo = max(0, k - FEED_WINDOW_BEFORE)
    hi = min(len(pairs), k + FEED_WINDOW_AFTER)
    window = [v for _, v in pairs[lo:hi]]
    if len(window) < FEED_MIN_WINDOW:
        return [], FEED_DEFAULT_CHANGE_FRAC
    return downsample_trace(window, FEED_TRACE_LEN), round((k - lo) / (hi - lo), 3)


def _severity(value: float, alert: float) -> str:
    if value >= alert:
        return "alert"
    return "changed" if value >= CHANGED_THRESHOLD else "stable"


def _links(change: dict) -> dict:
    model = change["model"]
    provider = change["provider"] or ""
    return {
        "slug": change["slug"],
        "model": model.split("/")[-1],
        "org": model.split("/")[0],
        "modelSlug": slugify(model),
        "provider": provider,
        # The slug provider pages are written under, so the link can never 404.
        "providerSlug": slugify(base_provider(provider)),
    }


def _lt_item(change: dict, drift: list[tuple[datetime, float]], now: datetime) -> dict:
    cd = datetime.fromisoformat(change["date"])
    magnitude = None
    trace: list[float] = []
    frac = FEED_DEFAULT_CHANGE_FRAC
    if drift:
        k = _nearest_index(drift, cd)
        peak_hi = min(len(drift), k + FEED_LT_PEAK_WINDOW)
        magnitude = round(max(v for _, v in drift[k:peak_hi]), 2)
        trace, frac = _window(drift, k)
    display = magnitude if magnitude is not None else "—"
    return {
        "date": change["date"][:10],
        "iso": change["date"],
        "daysAgo": (now - cd).days,
        "method": "lt",
        "magnitude": magnitude,
        "desc": f"Logprob averages moved {display} nats from the reference period.",
        "primary": f"drift {display}",
        "secondary": f"{change['magnitude_display']} conf",
        "sevKey": _severity(magnitude or 0.0, LT_ALERT_THRESHOLD),
        "trace": trace,
        "changeFrac": frac,
        **_links(change),
    }


def _b3it_item(change: dict, view: B3ITView | None, now: datetime) -> dict:
    cd = datetime.fromisoformat(change["date"])
    peak = 0.0
    trace: list[float] = []
    frac = FEED_DEFAULT_CHANGE_FRAC
    pairs = (
        list(
            zip(
                (datetime.fromisoformat(s) for s in view.tv_series["dates"]),
                view.tv_series["values"],
            )
        )
        if view
        else []
    )
    if pairs:
        k = _nearest_index(pairs, cd)
        peak_hi = min(len(pairs), k + FEED_B3IT_PEAK_WINDOW)
        peak = round(max(v for _, v in pairs[k:peak_hi]), 3)
        trace, frac = _window(pairs, k)
    return {
        "date": change["date"][:10],
        "iso": change["date"],
        "daysAgo": (now - cd).days,
        "method": "b3it",
        "magnitude": peak,
        "desc": f"Border-input output distribution moved (TV {peak:.2f}) from the reference.",
        "primary": f"TV {peak:.2f}",
        "secondary": "border-input shift",
        "sevKey": _severity(peak, B3IT_ALERT_THRESHOLD),
        "trace": trace,
        "changeFrac": frac,
        **_links(change),
    }


def build_feed_items(
    changes: list[dict],
    drift_by_slug: dict[str, list[tuple[datetime, float]]],
    b3it_by_slug: dict[str, B3ITView],
    now: datetime,
) -> list[dict]:
    """Enrich merged change events (changes.json shape) for display, newest first."""
    items = []
    for change in changes:
        if change["method"] == "LT":
            items.append(_lt_item(change, drift_by_slug.get(change["slug"], []), now))
        else:
            items.append(_b3it_item(change, b3it_by_slug.get(change["slug"]), now))
    items.sort(key=lambda i: i["iso"], reverse=True)
    return items
