"""data/home.json: the front page's own slice of the overview.

The front page shows the site stats, the hero, the latest changes, the ten most
drift-prone providers and the ten most-changed endpoints. Fetching overview.json
for that meant downloading every directory row -- 77 KB gzipped, 984 KB raw -- to
draw ten of them, and only after the script had arrived. This file is what the
page actually needs (about 5 KB gzipped); render.py inlines it into index.html,
so the sections fill without a second round trip, and writes it beside
overview.json for anyone fetching it.

The slices are chosen here, once: index.md renders the same rows, and overview.ts
draws them as given rather than re-selecting.
"""

PLOT_PREVIEW = 10
DIR_PREVIEW = 10


def rateable_rows(provs: list[dict]) -> list[dict]:
    """The drift-rate plot's rows (rate_plot.ts::rateablePlotRows): providers with
    a rate at all, most drift-prone first, a zero rate ordered by its upper bound."""
    return sorted(
        (p for p in provs if p["lt_rate"] is not None),
        key=lambda p: (-p["lt_rate"], -p["lt_ci"][1], p["name"]),
    )


def most_changed(rows: list[dict], n: int) -> list[dict]:
    """The endpoint preview: actively tracked rows, most changes first."""
    tracked = [r for r in rows if r["headline"] == "tracked"]
    return sorted(tracked, key=lambda r: (-r["nChanges"], r["model"].lower()))[:n]


def build_home(overview: dict) -> dict:
    return {
        "stats": overview["stats"],
        "hero": overview["hero"],
        "feed": overview["feed"],
        "providers": rateable_rows(overview["providers"])[:PLOT_PREVIEW],
        # the preview links a provider only when it has a page: every slug that does
        "providerPages": sorted(p["slug"] for p in overview["providers"]),
        "endpoints": most_changed(overview["endpoints"], DIR_PREVIEW),
    }
