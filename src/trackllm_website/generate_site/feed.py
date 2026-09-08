"""Change-feed enrichment: magnitude, sparkline window and link slugs per change.

Shared by the Overview (its latest slice) and the Changes page (all of them), so
the two never disagree about what a change looked like.
"""

from datetime import datetime

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.brands import Brand, brand_json
from trackllm_website.generate_site.naming import base_provider, variant_name
from trackllm_website.lt_drift import LT_SHIFT_WINDOW_DAYS
from trackllm_website.util import slugify

TRACE_LEN = 28

FEED_TRACE_LEN = 40
FEED_WINDOW_BEFORE = 60
FEED_WINDOW_AFTER = 20
FEED_MIN_WINDOW = 6
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
    # The sparkline draws point i of n at x = i/(n-1): map the change to its
    # (possibly downsampled) point index and normalize by n-1, or the mark
    # lands one point-width left of the sample it dates.
    n = min(len(window), FEED_TRACE_LEN)
    bucket = (k - lo) * n // len(window)
    return downsample_trace(window, FEED_TRACE_LEN), round(bucket / (n - 1), 3)


def _severity(value: float, alert: float) -> str:
    if value >= alert:
        return "alert"
    return "changed" if value >= CHANGED_THRESHOLD else "stable"


def change_links(change: dict) -> dict:
    """Display names and page slugs for one change -- shared with hero.py.

    An endpoint that has left the fleet reaches changes.json through the
    merge_changes fallback: model = its own slug, no provider. No endpoint, model
    or provider page is generated for it, so it gets no page slugs -- the UI
    renders those names as plain text rather than linking a 404.
    """
    model = change["model"]
    provider = change["provider"] or ""
    in_fleet = bool(provider)
    return {
        "slug": change["slug"],
        # The slug an endpoint page was written under; empty when there is none.
        "endpointSlug": change["slug"] if in_fleet else "",
        "model": model.split("/")[-1],
        "org": model.split("/")[0],
        "modelSlug": slugify(model) if in_fleet else "",
        "provider": provider,
        # The slug provider pages are written under, so the link can never 404.
        "providerSlug": slugify(base_provider(provider)) if in_fleet else "",
    }


def brand_fields(change: dict, brands: dict[str, Brand]) -> dict:
    """How a feed row names the serving company: "served by <brand> (<variant>)",
    as the endpoint head shows it. A departed endpoint has no provider and gets
    a nameless brand, which the row leaves out."""
    provider = change["provider"] or ""
    return {
        "brand": brand_json(brands, slugify(base_provider(provider)))
        if provider
        else Brand(name="").model_dump(),
        "variant": variant_name(provider),
    }


def _lt_item(change: dict, drift: list[tuple[datetime, float]], now: datetime) -> dict:
    cd = datetime.fromisoformat(change["date"])
    # The magnitude is the event's own level shift (lt_events), the number its
    # publication gate passed on -- not re-levelled here, so the feed and the
    # gate can never disagree.
    magnitude = change["magnitude"]
    trace: list[float] = []
    frac = FEED_DEFAULT_CHANGE_FRAC
    if drift:
        trace, frac = _window(drift, _nearest_index(drift, cd))
    display = "—" if magnitude is None else f"{magnitude:.2f}"
    return {
        "date": change["date"][:10],
        "iso": change["date"],
        "daysAgo": (now - cd).days,
        "method": "lt",
        "magnitude": magnitude,
        "desc": f"Logprob averages moved {display} nats from baseline",
        "primary": f"shift {display}",
        "secondary": f"±{LT_SHIFT_WINDOW_DAYS}-day level shift",
        "sevKey": _severity(magnitude or 0.0, LT_ALERT_THRESHOLD),
        "trace": trace,
        "changeFrac": frac,
        **change_links(change),
    }


def _b3it_item(change: dict, view: B3ITView | None, now: datetime) -> dict:
    cd = datetime.fromisoformat(change["date"])
    # None, never 0.0: without the view the TV this change reached is unknown, and a
    # published "TV 0.00" would read as a change that moved nothing.
    peak: float | None = None
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
        level = view.change_mags.get(change["date"][:10])
        peak = round(level, 3) if level is not None else None
        trace, frac = _window(pairs, _nearest_index(pairs, cd))
    display = f"{peak:.2f}" if peak is not None else "—"
    return {
        "date": change["date"][:10],
        "iso": change["date"],
        "daysAgo": (now - cd).days,
        "method": "b3it",
        "magnitude": peak,
        "desc": f"Probe-prompt outputs shifted (total variation {display})",
        "primary": f"TV {display}",
        "secondary": "border-input shift",
        "sevKey": _severity(peak or 0.0, B3IT_ALERT_THRESHOLD),
        "trace": trace,
        "changeFrac": frac,
        **change_links(change),
    }


def build_feed_items(
    changes: list[dict],
    drift_by_slug: dict[str, list[tuple[datetime, float]]],
    b3it_by_slug: dict[str, B3ITView],
    brands: dict[str, Brand],
    now: datetime,
) -> list[dict]:
    """Enrich merged change events (changes.json shape) for display, newest first."""
    items = []
    for change in changes:
        fields = brand_fields(change, brands)
        # No fallback branch: a method this file does not know how to enrich would
        # otherwise be published as B3IT, wrong scale and wrong badge included.
        if change["method"] == "LT":
            item = _lt_item(change, drift_by_slug.get(change["slug"], []), now)
        elif change["method"] == "B3IT":
            item = _b3it_item(change, b3it_by_slug.get(change["slug"]), now)
        else:
            raise ValueError(
                f"unknown change method {change['method']!r} for {change['slug']}"
            )
        items.append({**item, **fields})
    items.sort(key=lambda i: i["iso"], reverse=True)
    return items
