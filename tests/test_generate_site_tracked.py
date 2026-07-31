import json
from pathlib import Path

from conftest import (
    b3it_slug,
    b3it_views_for,
    write_b3it_series,
    write_b3it_state,
    write_lt_endpoint,
    write_month_dir,
)
from trackllm_website.generate_site.lt import discover_lt_endpoints, load_all_lt_data
from trackllm_website.generate_site.tracked import with_observations

DATES = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 6)]


def write_lt_endpoint_without_scores(root: Path, slug: str, model: str, provider: str):
    """An endpoint whose every query errored: the directory layout is there, but
    lt_scores.json was never written."""
    d = root / "data" / "lt" / slug / "default"
    d.mkdir(parents=True)
    (d / "info.json").write_text(
        json.dumps({"prompt": "hi", "endpoint": {"model": model, "provider": provider}})
    )
    write_month_dir(d, "2026-06", [["24 10:00:00", "e0"]])


def _fleets(root: Path):
    lt_dir = root / "data" / "lt"
    discovered = discover_lt_endpoints(lt_dir) if lt_dir.exists() else []
    lt_by_slug = {e.slug: e for e in discovered}
    return with_observations(
        lt_by_slug,
        load_all_lt_data(lt_dir, lt_by_slug),
        b3it_views_for(root),
    )


def test_lt_endpoint_with_no_scores_is_not_tracked(tmp_path):
    root = tmp_path / "website"
    write_lt_endpoint(
        root, "m2fa23good", "m/a", "good", dates=DATES, changes=[], drift=[0.1] * 5
    )
    write_lt_endpoint_without_scores(root, "m2fa23dead", "m/a", "dead")

    lt_by_slug, b3it_views = _fleets(root)
    assert set(lt_by_slug) == {"m2fa23good"}
    assert b3it_views == {}


def test_b3it_endpoint_with_no_series_is_not_tracked(tmp_path):
    root = tmp_path / "website"
    write_b3it_series(
        root,
        "m/a",
        "good",
        status="monitoring",
        retired=None,
        month="2026-06",
        tokens=["A"] * 10,
    )
    write_b3it_state(root, "m/a", "dead", status="retired")

    _, b3it_views = _fleets(root)
    assert set(b3it_views) == {b3it_slug("m/a", "good")}


def test_a_dead_lt_series_does_not_take_a_live_b3it_one_with_it(tmp_path):
    """Filtered per method: the endpoint stays, its empty lt badge does not."""
    root = tmp_path / "website"
    write_lt_endpoint_without_scores(root, b3it_slug("m/a", "p"), "m/a", "p")
    write_b3it_series(
        root,
        "m/a",
        "p",
        status="monitoring",
        retired=None,
        month="2026-06",
        tokens=["A"] * 10,
    )

    lt_by_slug, b3it_views = _fleets(root)
    assert lt_by_slug == {}
    assert set(b3it_views) == {b3it_slug("m/a", "p")}


def test_endpoints_with_observations_are_left_alone(tmp_path):
    root = tmp_path / "website"
    write_lt_endpoint(
        root, "m2fa23p", "m/a", "p", dates=DATES, changes=[], drift=[0.1] * 5
    )
    write_b3it_series(
        root,
        "m/b",
        "q",
        status="monitoring",
        retired=None,
        month="2026-06",
        tokens=["A"] * 10,
    )

    lt_by_slug, b3it_views = _fleets(root)
    assert set(lt_by_slug) == {"m2fa23p"}
    assert set(b3it_views) == {b3it_slug("m/b", "q")}
