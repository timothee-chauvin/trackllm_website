"""data/changes_page.json: the complete change log, its month histogram and rankings.

Reads only already-generated data; the enrichment itself lives in feed.py so the
Overview's latest-changes slice and this log can never disagree.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.clock import site_now
from trackllm_website.generate_site.feed import build_feed_items
from trackllm_website.generate_site.lt import LTData
from trackllm_website.generate_site.months import month_range

TOP_ENDPOINTS = 5
RECENT_DAYS = 30


def _day(dt: datetime) -> str:
    return dt.date().isoformat()


def build_changes_page(
    website_dir: Path,
    lt_data: dict[str, LTData],
    b3it_views: dict[str, B3ITView],
) -> dict:
    data_dir = website_dir / "data"

    all_dates: list[str] = []
    for d in lt_data.values():
        all_dates += [_day(d.dates[0]), _day(d.dates[-1])]
    for view in b3it_views.values():
        dates = view.tv_series["dates"]
        if dates:
            all_dates += [dates[0][:10], dates[-1][:10]]

    changes_path = data_dir / "changes.json"
    changes = json.loads(changes_path.read_text()) if changes_path.exists() else []

    now = site_now(lt_data, b3it_views)
    drift_by_slug = {slug: d.drift for slug, d in lt_data.items()}
    items = build_feed_items(changes, drift_by_slug, b3it_views, now) if now else []

    # A change outside the observed span (e.g. an epoch closure recorded after an
    # endpoint's last sampled point) must still land in a bucket, or the histogram
    # would silently drop it.
    span = all_dates + [i["date"] for i in items]
    months = month_range(min(span), max(span)) if span else []
    lt_counts = Counter(i["date"][:7] for i in items if i["method"] == "lt")
    b3it_counts = Counter(i["date"][:7] for i in items if i["method"] == "b3it")

    per_endpoint: dict[str, dict] = {}
    for item in items:  # items are newest first, so the first hit is the latest
        rec = per_endpoint.setdefault(
            item["slug"],
            {
                "slug": item["slug"],
                "model": item["model"],
                "provider": item["provider"],
                "providerSlug": item["providerSlug"],
                "modelSlug": item["modelSlug"],
                "n": 0,
                "last": item["date"],
            },
        )
        rec["n"] += 1

    lt_drifts = [
        i["magnitude"] for i in items if i["method"] == "lt" and i["magnitude"]
    ]
    return {
        "stats": {
            "total": len(items),
            "lt": sum(1 for i in items if i["method"] == "lt"),
            "b3it": sum(1 for i in items if i["method"] == "b3it"),
            "endpoints_affected": len(per_endpoint),
            # An endpoint that has left the fleet carries no provider (feed.py
            # leaves its slugs empty); the empty string is not a provider.
            "providers_involved": len(
                {i["providerSlug"] for i in items if i["providerSlug"]}
            ),
            "changes_30d": sum(1 for i in items if i["daysAgo"] < RECENT_DAYS),
            "largest_lt_drift": max(lt_drifts, default=None),
            "since": min(all_dates) if all_dates else None,
            "now": now.strftime("%Y-%m-%d") if now else None,
        },
        "items": items,
        "months": [
            {"month": m, "lt": lt_counts.get(m, 0), "b3it": b3it_counts.get(m, 0)}
            for m in months
        ],
        "top_endpoints": sorted(
            per_endpoint.values(), key=lambda r: (-r["n"], r["slug"])
        )[:TOP_ENDPOINTS],
    }
