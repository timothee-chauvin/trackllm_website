import json

import pytest

from conftest import write_b3it_series, write_lt_endpoint
from trackllm_website.generate_site.b3it import discover_b3it_views
from trackllm_website.generate_site.lt import discover_lt_endpoints, load_all_lt_data
from trackllm_website.generate_site.overview import build_overview
from trackllm_website.generate_site.provider import (
    base_provider,
    build_provider_views,
    overview_rows,
    variant_name,
)
from trackllm_website.util import slugify


def test_base_provider_and_variant_split():
    assert base_provider("chutes/fp8") == "chutes"
    assert variant_name("chutes/fp8") == "fp8"
    assert base_provider("chutes") == "chutes"
    assert variant_name("chutes") == ""


@pytest.fixture
def fake_site(tmp_path):
    """One provider `p` serving two models under two variants, one of which changed."""
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(
        root,
        "org2fa23p",
        "org/a",
        "p",
        dates=dates,
        changes=[{"index": 24, "sigma": 40.0}],
        drift=[0.1] * 24 + [1.5] * 6,
    )
    write_lt_endpoint(
        root,
        "org2fb23p2ffp8",
        "org/b",
        "p/fp8",
        dates=dates,
        changes=[],
        drift=[0.1] * 30,
    )
    write_b3it_series(
        root,
        "org/a",
        "p",
        status="monitoring",
        retired=None,
        month="2026-06",
        tokens=["A"] * 24,
    )
    changes = [
        {
            "date": dates[24],
            "slug": "org2fa23p",
            "model": "org/a",
            "provider": "p",
            "method": "LT",
            "magnitude": 40.0,
            "magnitude_display": "40σ",
        }
    ]
    (root / "data" / "changes.json").write_text(json.dumps(changes))
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))
    return root


def _views(root):
    lt_dir = root / "data" / "lt"
    lt_endpoints = list(discover_lt_endpoints(lt_dir))
    lt_data = load_all_lt_data(lt_dir, [e.slug for e in lt_endpoints])
    b3it = discover_b3it_views(
        root / "data" / "b3it" / "state", root / "data" / "b3it" / "phase_2"
    )
    rows = build_overview(root, lt_data, lt_endpoints, b3it)["endpoints"]
    return build_provider_views(root, lt_data, lt_endpoints, b3it, rows)


def test_variants_group_under_one_provider(fake_site):
    views = _views(fake_site)
    assert list(views) == ["p"]
    view = views["p"]
    assert view["name"] == "p"
    assert view["n_endpoints"] == 2
    assert view["n_models"] == 2
    assert view["n_variants"] == 2
    assert {v["name"] for v in view["variants"]} == {"", "fp8"}


def test_exposure_summed_across_endpoints_and_rate_withheld_when_thin(fake_site):
    view = _views(fake_site)["p"]
    # two endpoints x 29 days each: the sum, not either variant's 0.08 alone
    assert view["lt"]["endpoints"] == 2
    assert view["lt"]["changes"] == 1
    assert view["lt"]["years"] == pytest.approx(0.16, abs=0.01)
    assert [v["lt"]["years"] for v in view["variants"]] == [
        pytest.approx(0.08, abs=0.01)
    ] * 2
    # b3it is observed over its own 24-day series, not the LT span
    assert view["b3it"]["years"] == pytest.approx(0.06, abs=0.01)
    # still well under MIN_ENDPOINT_YEARS, so no rate is published
    assert view["lt"]["rate"] is None
    assert view["lt"]["ci"] is None
    assert view["b3it"]["rate"] is None


def test_methods_kept_separate(fake_site):
    view = _views(fake_site)["p"]
    assert view["b3it"]["endpoints"] == 1
    assert view["b3it"]["changes"] == 0
    assert view["lt"]["changes"] == 1


def test_rate_published_once_exposure_clears_threshold(tmp_path):
    root = tmp_path / "website"
    dates = [f"2025-{m:02d}-01T00:00:00Z" for m in range(1, 13)] + [
        "2026-01-01T00:00:00Z"
    ]
    write_lt_endpoint(
        root,
        "org2fa23q",
        "org/a",
        "q",
        dates=dates,
        changes=[{"index": 6, "sigma": 30.0}],
        drift=[0.1] * 6 + [1.0] * 7,
    )
    (root / "data" / "changes.json").write_text(
        json.dumps(
            [
                {
                    "date": dates[6],
                    "slug": "org2fa23q",
                    "model": "org/a",
                    "provider": "q",
                    "method": "LT",
                    "magnitude": 30.0,
                    "magnitude_display": "30σ",
                }
            ]
        )
    )
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))
    view = _views(root)["q"]
    assert view["lt"]["years"] == pytest.approx(1.0, abs=0.05)
    assert view["lt"]["rate"] == pytest.approx(1.0, abs=0.06)
    assert view["lt"]["ci"][0] < view["lt"]["rate"] < view["lt"]["ci"][1]
    # the variant row clears the threshold on its own and publishes the same rate
    (variant,) = view["variants"]
    assert variant["lt"]["rate"] == view["lt"]["rate"]
    assert variant["lt"]["ci"] == view["lt"]["ci"]


def test_monthly_monitoring_counts_match_endpoint_spans(fake_site):
    view = _views(fake_site)["p"]
    assert view["months"] == ["2026-06"]
    for variant in view["variants"]:
        assert variant["monitoring"] == [1]


def test_provider_carries_its_changes_and_endpoint_rows(fake_site):
    view = _views(fake_site)["p"]
    assert [c["date"] for c in view["changes"]] == ["2026-06-25"]
    assert {e["slug"] for e in view["endpoints"]} == {"org2fa23p", "org2fb23p2ffp8"}


def test_change_count_equals_the_changes_listed(fake_site):
    view = _views(fake_site)["p"]
    lt_items = [c for c in view["changes"] if c["method"] == "lt"]
    assert view["lt"]["changes"] == len(lt_items)
    for variant in view["variants"]:
        listed = [c for c in lt_items if variant_name(c["provider"]) == variant["name"]]
        assert variant["lt"]["changes"] == len(listed)


def test_change_count_follows_changes_json_not_the_recomputed_scores(fake_site):
    """changes.json is canonical; the build-time recompute stored in lt_scores.json
    double-detects some changes on adjacent days, and those must not be counted."""
    scores_path = fake_site / "data" / "lt" / "org2fa23p" / "lt_scores.json"
    scores = json.loads(scores_path.read_text())
    scores["changes"] = [{"index": 24, "sigma": 40.0}, {"index": 25, "sigma": 38.0}]
    scores_path.write_text(json.dumps(scores))

    view = _views(fake_site)["p"]
    assert view["lt"]["changes"] == 1
    assert view["lt"]["changes"] == len(view["changes"])


def test_change_links_use_the_slug_the_provider_view_is_keyed_by(tmp_path):
    """A provider whose name does not slugify to itself must still resolve: the
    feed item's providerSlug is what the page links to, and provider pages are
    written under slugify(base)."""
    root = tmp_path / "website"
    provider = "acme corp"
    slug = slugify(provider)
    assert slug != provider
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(
        root,
        "org2fa23acme-corp",
        "org/a",
        provider,
        dates=dates,
        changes=[{"index": 24, "sigma": 40.0}],
        drift=[0.1] * 24 + [1.5] * 6,
    )
    (root / "data" / "changes.json").write_text(
        json.dumps(
            [
                {
                    "date": dates[24],
                    "slug": "org2fa23acme-corp",
                    "model": "org/a",
                    "provider": provider,
                    "method": "LT",
                    "magnitude": 40.0,
                    "magnitude_display": "40σ",
                }
            ]
        )
    )
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))

    views = _views(root)
    assert list(views) == [slug]
    (item,) = views[slug]["changes"]
    assert item["providerSlug"] == slug


def test_overview_rows_are_one_per_provider_with_last_change(fake_site):
    rows = overview_rows(_views(fake_site))
    (row,) = rows
    assert row["name"] == "p"
    assert row["slug"] == "p"
    assert row["n_variants"] == 2
    assert row["lt_rate"] is None
    assert row["last_change"] == "2026-06-25"
