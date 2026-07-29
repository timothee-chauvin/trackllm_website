"""The Overview hero curve: one pinned endpoint over one pinned window.

The hero is the site's first claim -- "LLM API outputs are unstable" -- so the
curve is a vetted endpoint and a vetted date range from `config.hero`, not
whatever this build happened to score highest. A picture that reshuffles itself
nightly cannot be checked by eye, and the auto-selector kept landing on a young
B3IT endpoint with twelve points either side of its changepoint.

Values are published at full daily resolution: no downsampling, and (since
lt_drift dropped its rolling median) no filtering, so the line on the page is
the measurement.
"""

import statistics
from datetime import datetime

from trackllm_website.config import HeroConfig
from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.feed import change_links

HERO_MIN_POINTS = 20  # below this the window cannot show a baseline and an after
HERO_HEADROOM = 1.12  # top of the drawn band, above the peak


def _b3it_series(view: B3ITView) -> list[tuple[datetime, float]]:
    return list(
        zip(
            (datetime.fromisoformat(s) for s in view.tv_series["dates"]),
            view.tv_series["values"],
        )
    )


def build_hero(
    changes: list[dict],
    drift_by_slug: dict[str, list[tuple[datetime, float]]],
    b3it_by_slug: dict[str, B3ITView],
    now: datetime,
    pin: HeroConfig,
) -> dict:
    """The pinned change event, as the Overview hero draws it.

    Raises rather than returning None on every failure: a stale pin must break the
    build loudly instead of silently blanking the homepage's headline graphic.
    """
    if pin.method == "lt":
        series = drift_by_slug.get(pin.slug, [])
    else:
        view = b3it_by_slug.get(pin.slug)
        series = _b3it_series(view) if view else []
    if not series:
        raise ValueError(
            f"hero pin {pin.slug!r}: no {pin.method} series in this build -- "
            "the pinned endpoint's data is missing or its slug changed"
        )

    window = [(dt, v) for dt, v in series if pin.start <= dt.date() <= pin.end]
    if len(window) < HERO_MIN_POINTS:
        raise ValueError(
            f"hero pin {pin.slug!r}: {pin.start}..{pin.end} covers only "
            f"{len(window)} points, need at least {HERO_MIN_POINTS}"
        )

    dates = [dt for dt, _ in window]
    values = [v for _, v in window]

    method_key = pin.method.upper()
    in_window = [
        c
        for c in changes
        if c["slug"] == pin.slug
        and c["method"] == method_key
        and pin.start <= datetime.fromisoformat(c["date"]).date() <= pin.end
    ]
    if not in_window:
        raise ValueError(
            f"hero pin {pin.slug!r}: no {method_key} change between "
            f"{pin.start} and {pin.end} -- the card has no detection to name"
        )
    change = max(in_window, key=lambda c: c["date"])
    when = datetime.fromisoformat(change["date"])
    k = min(range(len(dates)), key=lambda i: abs((dates[i] - when).total_seconds()))

    return {
        "method": pin.method,
        "date": when.date().isoformat(),
        "iso": change["date"],
        "daysAgo": (now - when).days,
        "start": dates[0].date().isoformat(),
        "end": dates[-1].date().isoformat(),
        "baseline": round(statistics.median(values[:k]), 3) if k else values[0],
        "magnitude": round(max(values[k:]), 3),
        "values": [round(v, 4) for v in values],
        # against the drawn x positions (i / (n-1)), so the dashed changepoint rule
        # lands on the step rather than one point to its left
        "changeFrac": round(k / (len(values) - 1), 4),
        "yMax": round(max(values) * HERO_HEADROOM, 3),
        **change_links(change),
    }
