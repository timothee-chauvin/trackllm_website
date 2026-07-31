import json

import pytest

from conftest import b3it_slug, write_b3it_series, write_lt_endpoint
from trackllm_website.generate_site.b3it import discover_b3it_views
from trackllm_website.generate_site.changes_page import build_changes_page
from trackllm_website.generate_site.lt import discover_lt_endpoints, load_all_lt_data

TOP_N = 5


@pytest.fixture
def fake_site(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-0{m}-{d:02d}T00:00:00Z" for m in (4, 5, 6) for d in range(1, 11)]
    write_lt_endpoint(
        root,
        "a23p",
        "org/a",
        "p",
        dates=dates,
        changes=[{"index": 5, "sigma": 40.0}, {"index": 22, "sigma": 12.0}],
        drift=[0.1] * 5 + [1.4] * 17 + [2.0] * 8,
    )
    write_lt_endpoint(
        root,
        "b23q",
        "org/b",
        "q",
        dates=dates,
        changes=[{"index": 12, "sigma": 20.0}],
        drift=[0.1] * 12 + [0.9] * 18,
    )
    changes = [
        {
            "date": dates[i],
            "slug": slug,
            "model": model,
            "provider": provider,
            "method": "LT",
            "magnitude": 20.0,
            "magnitude_display": "20σ",
        }
        for i, slug, model, provider in (
            (5, "a23p", "org/a", "p"),
            (22, "a23p", "org/a", "p"),
            (12, "b23q", "org/b", "q"),
        )
    ]
    (root / "data" / "changes.json").write_text(json.dumps(changes))
    return root


@pytest.fixture
def fake_site_with_b3it(fake_site):
    write_b3it_series(
        fake_site,
        "org/c",
        "r",
        status="monitoring",
        retired=None,
        month="2026-05",
        tokens=["A"] * 12 + ["B"] * 12,
    )
    changes_path = fake_site / "data" / "changes.json"
    changes = json.loads(changes_path.read_text())
    changes.append(
        {
            "date": "2026-05-13T00:00:00+00:00",
            "slug": b3it_slug("org/c", "r"),
            "model": "org/c",
            "provider": "r",
            "method": "B3IT",
            "magnitude": None,
            "magnitude_display": "",
        }
    )
    changes_path.write_text(json.dumps(changes))
    return fake_site


def _build(root):
    lt_dir = root / "data" / "lt"
    lt_endpoints = list(discover_lt_endpoints(lt_dir))
    lt_data = load_all_lt_data(lt_dir, [e.slug for e in lt_endpoints])
    return build_changes_page(
        root,
        lt_data,
        discover_b3it_views(
            root / "data" / "b3it" / "state", root / "data" / "b3it" / "phase_2"
        ),
    )


def test_every_change_appears_exactly_once(fake_site):
    page = _build(fake_site)
    assert len(page["items"]) == 3
    assert page["stats"]["total"] == 3
    assert page["stats"]["lt"] == 3
    assert page["stats"]["b3it"] == 0


def test_items_sorted_newest_first(fake_site):
    dates = [i["date"] for i in _build(fake_site)["items"]]
    assert dates == sorted(dates, reverse=True)


def test_month_histogram_totals_equal_the_change_count(fake_site):
    page = _build(fake_site)
    assert sum(m["lt"] + m["b3it"] for m in page["months"]) == page["stats"]["total"]


def test_months_are_contiguous(fake_site):
    months = [m["month"] for m in _build(fake_site)["months"]]
    assert months == ["2026-04", "2026-05", "2026-06"]


def test_top_endpoints_ranked_by_change_count(fake_site):
    top = _build(fake_site)["top_endpoints"]
    assert len(top) <= TOP_N
    assert top[0]["slug"] == "a23p"
    assert top[0]["n"] == 2
    assert top[0]["last"] == max(
        i["date"] for i in _build(fake_site)["items"] if i["slug"] == "a23p"
    )


def test_stats_report_affected_endpoints_and_providers(fake_site):
    stats = _build(fake_site)["stats"]
    assert stats["endpoints_affected"] == 2
    assert stats["providers_involved"] == 2
    assert stats["largest_lt_drift"] == pytest.approx(2.0)


def test_providers_involved_ignores_changes_with_no_provider(fake_site):
    """A change whose endpoint has left the fleet carries no provider (the
    merge_changes fallback); an empty providerSlug is not a provider."""
    changes_path = fake_site / "data" / "changes.json"
    changes = json.loads(changes_path.read_text())
    changes.append(
        {
            "date": "2026-06-05T00:00:00Z",
            "slug": "gone2fslug",
            "model": "gone2fslug",
            "provider": "",
            "method": "LT",
            "magnitude": 5.0,
            "magnitude_display": "5σ",
        }
    )
    changes_path.write_text(json.dumps(changes))

    stats = _build(fake_site)["stats"]
    assert stats["total"] == 4
    assert stats["providers_involved"] == 2


def test_changes_30d_spans_b3it_observations_newer_than_the_last_lt_one(fake_site):
    """The window is measured against the newest observation of either method, so
    a B3IT change after the last logprob observation is not dated in the future."""
    write_b3it_series(
        fake_site,
        "org/c",
        "r",
        status="monitoring",
        retired=None,
        month="2026-07",
        tokens=["A"] * 12 + ["B"] * 12,
    )
    changes_path = fake_site / "data" / "changes.json"
    changes = json.loads(changes_path.read_text())
    changes.append(
        {
            "date": "2026-07-13T00:00:00+00:00",
            "slug": b3it_slug("org/c", "r"),
            "model": "org/c",
            "provider": "r",
            "method": "B3IT",
            "magnitude": None,
            "magnitude_display": "",
        }
    )
    changes_path.write_text(json.dumps(changes))

    page = _build(fake_site)
    assert page["stats"]["now"] == "2026-07-24"
    assert min(i["daysAgo"] for i in page["items"]) >= 0
    assert page["stats"]["changes_30d"] == 1


def test_b3it_changes_are_counted_alongside_lt(fake_site_with_b3it):
    page = _build(fake_site_with_b3it)
    assert page["stats"]["total"] == 4
    assert page["stats"]["lt"] == 3
    assert page["stats"]["b3it"] == 1
    assert page["stats"]["endpoints_affected"] == 3
    assert page["stats"]["providers_involved"] == 3
    by_month = {m["month"]: m for m in page["months"]}
    assert by_month["2026-05"]["b3it"] == 1
    assert sum(m["lt"] + m["b3it"] for m in page["months"]) == 4
