from trackllm_website.generate_site.home import (
    DIR_PREVIEW,
    PLOT_PREVIEW,
    build_home,
    most_changed,
    rateable_rows,
)


def _prov(name: str, rate: float | None, hi: float = 1.0) -> dict:
    return {
        "name": name,
        "slug": name,
        "lt_rate": rate,
        "lt_ci": None if rate is None else [0.0, hi],
    }


def _row(model: str, n: int, headline: str = "tracked") -> dict:
    return {"model": model, "nChanges": n, "headline": headline}


def test_rateable_rows_drop_unrated_and_order_by_rate_then_upper_bound():
    rows = rateable_rows(
        [
            _prov("quiet", 0.0, 0.5),
            _prov("none", None),
            _prov("busy", 2.0),
            _prov("still", 0.0, 0.9),
        ]
    )
    assert [p["name"] for p in rows] == ["busy", "still", "quiet"]


def test_most_changed_keeps_tracked_rows_most_changes_first():
    rows = [
        _row("b", 2),
        _row("a", 2),
        _row("c", 5, "retired"),
        _row("d", 0),
        _row("e", 9),
    ]
    assert [r["model"] for r in most_changed(rows, 3)] == ["e", "a", "b"]


def test_build_home_is_the_front_page_slice():
    provs = [_prov(f"p{i}", float(i)) for i in range(PLOT_PREVIEW + 3)] + [
        _prov("unrated", None)
    ]
    rows = [_row(f"m{i}", i) for i in range(DIR_PREVIEW + 5)]
    overview = {
        "stats": {"active": 1},
        "hero": None,
        "feed": [{"slug": "x"}],
        "providers": provs,
        "endpoints": rows,
    }
    home = build_home(overview)
    assert set(home) == {
        "stats",
        "hero",
        "feed",
        "providers",
        "providerPages",
        "endpoints",
    }
    assert home["stats"] is overview["stats"] and home["feed"] is overview["feed"]
    assert len(home["providers"]) == PLOT_PREVIEW
    assert home["providers"][0]["name"] == f"p{PLOT_PREVIEW + 2}"
    # every provider with a page, rated or not, so the preview can link the right ones
    assert home["providerPages"] == sorted(p["slug"] for p in provs)
    assert [r["nChanges"] for r in home["endpoints"]] == list(
        range(DIR_PREVIEW + 4, 4, -1)
    )
