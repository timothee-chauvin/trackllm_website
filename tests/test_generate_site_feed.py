import json
from datetime import datetime, timezone

import pytest

from conftest import write_b3it_state, write_lt_endpoint
from trackllm_website.generate_site.b3it import B3ITView, discover_b3it_views
from trackllm_website.generate_site.feed import build_feed_items, downsample_trace
from trackllm_website.generate_site.lt import discover_lt_endpoints, load_all_lt_data
from trackllm_website.generate_site.overview import build_overview

NOW = datetime(2026, 6, 30, tzinfo=timezone.utc)


def _drift(n: int, jump_at: int):
    return [
        (
            datetime(2026, 6, 1, tzinfo=timezone.utc).replace(day=1 + i),
            0.1 if i < jump_at else 1.2,
        )
        for i in range(n)
    ]


def _b3it_view(slug: str) -> B3ITView:
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 21)]
    values = [0.05] * 12 + [0.8] * 8
    return B3ITView(
        slug=slug,
        model="m/a",
        provider="p/fp8",
        status="monitoring",
        retired_reason=None,
        n_bis=3,
        unstable=False,
        epochs=[],
        tv_series={"dates": dates, "values": values},
        changes=[{"date": dates[12], "kind": "onset"}],
        last_query=dates[-1],
    )


def test_downsample_trace_caps_length():
    assert len(downsample_trace(list(range(200)), 28)) == 28


def test_downsample_trace_short_input_untouched():
    assert downsample_trace([1.0, 2.0], 28) == [1.0, 2.0]


def test_lt_item_carries_drift_magnitude_and_link_slugs():
    changes = [
        {
            "date": "2026-06-15T00:00:00Z",
            "slug": "m2fa23p",
            "model": "org/model-x",
            "provider": "chutes/fp8",
            "method": "LT",
            "magnitude": 40.0,
            "magnitude_display": "40σ",
        }
    ]
    items = build_feed_items(changes, {"m2fa23p": _drift(20, 14)}, {}, NOW)
    (item,) = items
    assert item["method"] == "lt"
    assert item["magnitude"] == 1.2
    assert item["model"] == "model-x"
    assert item["org"] == "org"
    assert item["providerSlug"] == "chutes"
    assert item["modelSlug"] == "org2fmodel-x"
    assert item["slug"] == "m2fa23p"
    assert item["secondary"] == "40σ conf"
    assert item["trace"]


def test_lt_item_without_drift_series():
    changes = [
        {
            "date": "2026-06-15T00:00:00Z",
            "slug": "m2fa23p",
            "model": "org/model-x",
            "provider": "chutes/fp8",
            "method": "LT",
            "magnitude": 40.0,
            "magnitude_display": "40σ",
        }
    ]
    (item,) = build_feed_items(changes, {}, {}, NOW)
    assert item["magnitude"] is None
    assert item["primary"] == "drift —"
    assert item["sevKey"] == "stable"
    assert item["trace"] == []


def test_b3it_item_uses_peak_tv_from_the_view():
    changes = [
        {
            "date": "2026-06-13T00:00:00Z",
            "slug": "s1",
            "model": "org/model-x",
            "provider": "p/fp8",
            "method": "B3IT",
            "magnitude": None,
            "magnitude_display": "",
        }
    ]
    items = build_feed_items(changes, {}, {"s1": _b3it_view("s1")}, NOW)
    (item,) = items
    assert item["method"] == "b3it"
    assert item["magnitude"] == 0.8
    assert item["sevKey"] == "alert"


def test_b3it_item_without_a_view_reports_no_magnitude():
    """A real detected change whose B3IT view is missing must not publish a
    fabricated TV of 0.00 -- the magnitude is unknown, and says so."""
    changes = [
        {
            "date": "2026-06-13T00:00:00Z",
            "slug": "s1",
            "model": "org/model-x",
            "provider": "p/fp8",
            "method": "B3IT",
            "magnitude": None,
            "magnitude_display": "",
        }
    ]
    (item,) = build_feed_items(changes, {}, {}, NOW)
    assert item["magnitude"] is None
    assert item["primary"] == "TV —"
    assert "TV —" in item["desc"]
    assert item["trace"] == []


def test_unrecognised_method_raises():
    changes = [
        {
            "date": "2026-06-13T00:00:00Z",
            "slug": "s1",
            "model": "org/model-x",
            "provider": "p",
            "method": "SOMETHING_NEW",
            "magnitude": None,
            "magnitude_display": "",
        }
    ]
    with pytest.raises(ValueError, match="SOMETHING_NEW"):
        build_feed_items(changes, {}, {}, NOW)


def test_items_sorted_newest_first():
    changes = [
        {
            "date": f"2026-06-{d:02d}T00:00:00Z",
            "slug": "m2fa23p",
            "model": "org/model-x",
            "provider": "p",
            "method": "LT",
            "magnitude": 10.0,
            "magnitude_display": "10σ",
        }
        for d in (3, 20, 11)
    ]
    items = build_feed_items(changes, {"m2fa23p": _drift(20, 14)}, {}, NOW)
    assert [i["date"] for i in items] == ["2026-06-20", "2026-06-11", "2026-06-03"]


@pytest.fixture
def fake_site_feed_agreement(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(
        root,
        "m2fa23p",
        "m/a",
        "p",
        dates=dates,
        changes=[{"index": 24, "sigma": 40.0}],
        drift=[0.1] * 24 + [1.5] * 6,
    )
    write_b3it_state(root, "m/a", "p", status="monitoring")
    changes = [
        {
            "date": dates[24],
            "slug": "m2fa23p",
            "model": "m/a",
            "provider": "p",
            "method": "LT",
            "magnitude": 40.0,
            "magnitude_display": "40σ",
        }
    ]
    (root / "data" / "changes.json").write_text(json.dumps(changes))
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {"lt": 1.0}}))
    lt_dir = root / "data" / "lt"
    lt_endpoints = list(discover_lt_endpoints(lt_dir))
    lt_data = load_all_lt_data(lt_dir, [e.slug for e in lt_endpoints])
    views = discover_b3it_views(
        root / "data" / "b3it" / "state", root / "data" / "b3it" / "phase_2"
    )
    return build_overview(root, lt_data, lt_endpoints, views), changes


def test_overview_feed_entries_come_from_changes_json(fake_site_feed_agreement):
    # the Overview's slice must be a subset of the canonical merged change list
    ov, changes = fake_site_feed_agreement
    canonical = {(c["date"][:10], c["slug"], c["method"].lower()) for c in changes}
    assert ov["feed"], "no feed to compare against"
    for item in ov["feed"]:
        assert (item["date"], item["slug"], item["method"]) in canonical
