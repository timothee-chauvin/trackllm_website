"""The shared timeline's series thinning: what it keeps, and where it says the
endpoint was not observed at all."""

from datetime import date, timedelta

from conftest import (
    b3it_views_for,
    empty_status_inputs,
    site_statuses_for,
    write_lt_endpoint,
)
from trackllm_website.generate_site.lt import discover_lt_endpoints
from trackllm_website.generate_site.model import build_model_views
from trackllm_website.generate_site.timeline import downsample_runs
from trackllm_website.util import slugify


def _days(start: str, n: int) -> list[str]:
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


def _series(dates: list[str]) -> list[tuple[str, float]]:
    return [(d, float(i)) for i, d in enumerate(dates)]


def test_a_contiguous_series_is_one_run():
    pairs = _series(_days("2026-01-01", 10))
    assert downsample_runs(pairs, 90) == (pairs, [])


def test_an_empty_series_has_nothing_to_break():
    assert downsample_runs([], 90) == ([], [])


def test_a_single_day_is_one_run_of_one_point():
    pairs = _series(["2026-01-01"])
    assert downsample_runs(pairs, 90) == (pairs, [])


def test_one_missing_day_breaks_the_series_in_two():
    pairs = _series(_days("2026-01-01", 3) + _days("2026-01-05", 3))
    kept, breaks = downsample_runs(pairs, 90)
    assert kept == pairs
    assert breaks == [3]


def test_every_missing_stretch_gets_its_own_break():
    pairs = _series(
        _days("2026-01-01", 2) + _days("2026-01-10", 2) + _days("2026-02-01", 4)
    )
    kept, breaks = downsample_runs(pairs, 90)
    assert kept == pairs
    assert breaks == [2, 4]


def test_thinning_keeps_both_ends_of_the_series():
    pairs = _series(_days("2026-01-01", 400))
    kept, breaks = downsample_runs(pairs, 90)
    assert breaks == []
    assert len(kept) <= 90
    assert (kept[0], kept[-1]) == (pairs[0], pairs[-1])


def test_thinning_keeps_both_ends_of_every_run():
    left = _series(_days("2026-01-01", 200))
    right = _series(_days("2026-09-01", 200))
    kept, breaks = downsample_runs(left + right, 90)
    assert len(breaks) == 1
    cut = breaks[0]
    assert (kept[0], kept[cut - 1]) == (left[0], left[-1])
    assert (kept[cut], kept[-1]) == (right[0], right[-1])


def test_a_run_never_thins_away_entirely():
    """A one-day run inside a long series still has to be drawable."""
    pairs = _series(_days("2026-01-01", 300) + ["2026-12-25"])
    kept, breaks = downsample_runs(pairs, 90)
    assert breaks == [len(kept) - 1]
    assert kept[-1][0] == "2026-12-25"


def test_the_timeline_publishes_the_breaks_it_found(tmp_path):
    root = tmp_path / "website"
    dates = [f"{d}T00:00:00Z" for d in _days("2026-06-01", 5) + _days("2026-06-10", 5)]
    write_lt_endpoint(
        root, "m2fa23p1", "m/a", "p1", dates=dates, changes=[], drift=[0.1] * 10
    )
    (root / "data" / "changes.json").write_text("[]")
    site = site_statuses_for(root, empty_status_inputs())
    views = build_model_views(
        root,
        list(discover_lt_endpoints(root / "data" / "lt")),
        b3it_views_for(root),
        site,
    )
    lt = views[slugify("m/a")]["endpoints"][0]["lt"]
    assert lt["breaks"], "the four unobserved days left no break in the drift lane"
    assert lt["drift"][lt["breaks"][0]][0] == "2026-06-10"
