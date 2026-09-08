import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from conftest import (
    b3it_views_for,
    catalog_entry,
    empty_status_inputs,
    site_statuses_for,
    write_b3it_series,
    write_b3it_state,
    write_lt_endpoint,
)
from trackllm_website.bi.state import RetiredInfo
from trackllm_website.config import Endpoint, HeroConfig
from trackllm_website.generate_site.changes import merge_changes, to_json
from trackllm_website.generate_site.lt import discover_lt_endpoints, load_all_lt_data
from trackllm_website.generate_site.feed import build_feed_items, downsample_trace
from trackllm_website.generate_site.overview import FEED_SIZE, build_overview
from trackllm_website.generate_site.status import STATUS_COPY
from trackllm_website.util import slugify

# The build clock: the fixtures' last observed day, so a same-day observation
# reads as 0 days old and the age expectations below stay exact.
NOW = datetime(2026, 6, 30, tzinfo=timezone.utc)


def _build_overview_with(root: Path, inputs, now: datetime) -> dict:
    lt_dir = root / "data" / "lt"
    lt_endpoints = list(discover_lt_endpoints(lt_dir)) if lt_dir.exists() else []
    lt_data = load_all_lt_data(lt_dir, [e.slug for e in lt_endpoints])
    b3it_views = b3it_views_for(root)
    site = site_statuses_for(root, inputs)
    return build_overview(root, lt_data, lt_endpoints, b3it_views, None, site, now)


def _build_overview(root: Path) -> dict:
    return _build_overview_with(root, empty_status_inputs(), NOW)


def test_downsample_trace_caps_length():
    assert len(downsample_trace(list(range(200)), 28)) == 28


def test_downsample_trace_short_input_untouched():
    assert downsample_trace([1.0, 2.0], 28) == [1.0, 2.0]


def test_downsample_trace_empty():
    assert downsample_trace([], 28) == []


def _write_b3it_with_transition(
    root: Path, model: str, provider: str, *, status, retired=None
):
    """A b3it endpoint whose reference actually produces a TV transition."""
    write_b3it_series(
        root,
        model,
        provider,
        status=status,
        retired=retired,
        month="2026-01",
        tokens=["A"] * 12 + ["B"] * 12,
    )


@pytest.fixture
def fake_site(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    drift = [0.1] * 24 + [1.5] * 6
    changes = [{"index": 24, "sigma": 40.0}]
    write_lt_endpoint(
        root, "m2fa23p", "m/a", "p", dates=dates, changes=changes, drift=drift
    )
    write_b3it_state(root, "m/a", "p", status="monitoring")

    (root / "data" / "changes.json").write_text(
        json.dumps(
            [
                {
                    "date": dates[24],
                    "slug": "m2fa23p",
                    "model": "m/a",
                    "provider": "p",
                    "method": "LT",
                    "magnitude": 1.4,
                }
            ]
        )
    )
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {"lt": 1.23}}))
    return root


def test_build_overview_shape(fake_site):
    ov = _build_overview(fake_site)
    assert set(ov) == {"stats", "hero", "feed", "endpoints"}
    ep = ov["endpoints"][0]
    assert set(ep) >= {
        "slug",
        "model",
        "provider",
        "methods",
        "status",
        "nChanges",
        "trace",
        "changeFracs",
    }
    assert (
        ov["stats"]["changes_total"]
        == ov["stats"]["changes_lt"] + ov["stats"]["changes_b3it"]
    )


def test_endpoint_trace_is_downsampled_drift(fake_site):
    ov = _build_overview(fake_site)
    ep = next(e for e in ov["endpoints"] if e["slug"] == "m2fa23p")
    assert ep["methods"] == ["lt", "b3it"]
    assert len(ep["trace"]) == 28
    assert ep["trace"][-1] > ep["trace"][0]


def test_status_changed_when_recent_change(fake_site):
    ov = _build_overview(fake_site)
    ep = next(e for e in ov["endpoints"] if e["slug"] == "m2fa23p")
    assert ep["status"] == "changed"
    assert ep["nChanges"] == 1
    assert ep["lastChange"] == "2026-06-25"
    assert ep["recent"] is True


def test_status_retired_when_no_recent_observation(tmp_path):
    root = tmp_path / "website"
    old_dates = [f"2025-01-{d:02d}T00:00:00Z" for d in range(1, 29)]
    recent_dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 29)]
    write_lt_endpoint(
        root, "old2fa23p", "old/a", "p", dates=old_dates, changes=[], drift=[0.1] * 28
    )
    write_lt_endpoint(
        root,
        "new2fa23p",
        "new/a",
        "p",
        dates=recent_dates,
        changes=[],
        drift=[0.1] * 28,
    )
    (root / "data" / "changes.json").write_text(json.dumps([]))
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))

    ov = _build_overview(root)
    old_ep = next(e for e in ov["endpoints"] if e["slug"] == "old2fa23p")
    new_ep = next(e for e in ov["endpoints"] if e["slug"] == "new2fa23p")
    assert old_ep["status"] == "retired"
    assert new_ep["status"] == "stable"


def test_feed_lt_item_has_drift_level_and_conf(fake_site):
    ov = _build_overview(fake_site)
    lt_item = next(f for f in ov["feed"] if f["method"] == "lt")
    assert lt_item["primary"] == "shift 1.40"  # the event's own level shift
    assert lt_item["secondary"] == "±7-day level shift"
    assert lt_item["sevKey"] == "alert"
    assert lt_item["desc"] == "Logprob averages moved 1.40 nats from baseline"
    assert len(lt_item["trace"]) > 0
    assert lt_item["model"] == "a"
    assert lt_item["provider"] == "p"


def test_feed_includes_b3it_item_from_view_transition(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 6)]
    write_lt_endpoint(
        root, "m2fa23p", "m/a", "p", dates=dates, changes=[], drift=[0.1] * 5
    )
    _write_b3it_with_transition(root, "m/b", "q", status="monitoring")
    # the feed reads B3IT items from the merged change list, as render.py writes it
    views = b3it_views_for(root)
    (root / "data" / "changes.json").write_text(
        json.dumps(to_json(merge_changes({}, {}, views)))
    )
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))

    ov = _build_overview(root)
    b3it_items = [f for f in ov["feed"] if f["method"] == "b3it"]
    assert b3it_items, "expected a b3it feed item from the view's transition"
    item = b3it_items[0]
    assert item["primary"].startswith("TV ")
    assert item["secondary"] == "border-input shift"
    assert item["sevKey"] in {"alert", "changed", "stable"}
    assert item["model"] == "b"
    assert item["provider"] == "q"


def test_feed_is_the_head_of_the_merged_change_list(tmp_path):
    """The front page's latest changes are the newest FEED_SIZE items of the same
    merged list /changes shows, whatever their method -- no per-method quota, so
    a B3IT change older than FEED_SIZE LT changes is not pulled in."""
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(
        root, "m2fa23p", "m/a", "p", dates=dates, changes=[], drift=[0.1] * 30
    )
    _write_b3it_with_transition(root, "m/b", "q", status="monitoring")
    views = b3it_views_for(root)
    lt_changes = [
        {
            "date": dates[i],
            "slug": "m2fa23p",
            "model": "m/a",
            "provider": "p",
            "method": "LT",
            "magnitude": 1.0,
        }
        for i in range(10, 10 + FEED_SIZE + 2)
    ]
    changes = to_json(merge_changes({}, {}, views)) + lt_changes
    (root / "data" / "changes.json").write_text(json.dumps(changes))
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))

    ov = _build_overview(root)
    lt_dir = root / "data" / "lt"
    lt_data = load_all_lt_data(lt_dir, ["m2fa23p"])
    drift_by_slug = {slug: d.drift for slug, d in lt_data.items()}
    merged = build_feed_items(changes, drift_by_slug, views, NOW)
    assert len(merged) > FEED_SIZE
    assert ov["feed"] == merged[:FEED_SIZE]
    assert {i["method"] for i in ov["feed"]} == {"lt"}


def test_b3it_only_retired_endpoint_gets_retired_status(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 6)]
    write_lt_endpoint(
        root, "m2fa23p", "m/a", "p", dates=dates, changes=[], drift=[0.1] * 5
    )
    _write_b3it_with_transition(
        root,
        "m/b",
        "q",
        status="retired",
        retired=RetiredInfo(
            reason="delisted",
            since=datetime(2026, 1, 25, tzinfo=timezone.utc),
            last_recheck=datetime(2026, 1, 25, tzinfo=timezone.utc),
        ),
    )
    (root / "data" / "changes.json").write_text(json.dumps([]))
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))

    inputs = empty_status_inputs()
    inputs.endpoints_lt = [
        Endpoint(api="openrouter", model="m/a", provider="p", cost=(1, 2))
    ]
    ov = _build_overview_with(root, inputs, NOW)
    ep = next(e for e in ov["endpoints"] if e["slug"] == "m2fb23q")
    assert ep["methods"] == ["b3it"]
    assert ep["status"] == "retired"
    assert len(ep["trace"]) > 0
    assert ov["stats"]["active"] == 1  # only the still-monitoring LT endpoint


def test_a_stalled_endpoint_with_a_fresh_trace_is_not_active(tmp_path):
    """`active` is the headline count the "Tracked" chip filters on, not a trace
    verdict: an endpoint whose queries have all started failing keeps a "stable"
    trace status for RETIRED_GAP_DAYS after its last good day, and counting that
    as active made the headline claim more endpoints than the chip could show."""
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(
        root, "m2fa23p", "m/a", "p", dates=dates, changes=[], drift=[0.1] * 30
    )
    (root / "data" / "changes.json").write_text(json.dumps([]))
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))

    # observed but absent from endpoints_lt, so status.py reads it as stalled
    ov = _build_overview(root)
    row = next(e for e in ov["endpoints"] if e["slug"] == "m2fa23p")
    assert (row["headline"], row["status"]) == ("retired", "stable")
    assert ov["stats"]["active"] == 0
    assert ov["stats"]["endpoints"] == 1


def test_a_monitored_endpoint_without_a_series_is_active(tmp_path):
    """The other direction: a freshly onboarded B3IT endpoint is monitored before
    its second batch gives it a series, so it has no trace status to read."""
    root = tmp_path / "website"
    write_b3it_state(root, "m/b", "q", status="monitoring")
    (root / "data" / "changes.json").write_text(json.dumps([]))
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))

    # its view carries no series, so tracked.py withholds it from the fleet
    site = site_statuses_for(root, empty_status_inputs())
    ov = build_overview(root, {}, [], {}, None, site, NOW)
    row = next(e for e in ov["endpoints"] if e["slug"] == slugify("m/b#q"))
    assert (row["headline"], row["status"]) == ("tracked", None)
    assert ov["stats"]["active"] == 1
    assert ov["stats"]["endpoints"] == 1


def test_change_count_follows_changes_json_not_the_recomputed_scores(fake_site):
    """changes.json is canonical; the build-time recompute stored in lt_scores.json
    double-detects some changes on adjacent days, and those must not be counted --
    the directory row and the change feed sit on the same page."""
    scores_path = fake_site / "data" / "lt" / "m2fa23p" / "lt_scores.json"
    scores = json.loads(scores_path.read_text())
    scores["changes"] = [{"index": 24, "sigma": 40.0}, {"index": 25, "sigma": 38.0}]
    scores_path.write_text(json.dumps(scores))

    ov = _build_overview(fake_site)
    ep = next(e for e in ov["endpoints"] if e["slug"] == "m2fa23p")
    assert ep["nChanges"] == 1
    assert ov["stats"]["changed_endpoints"] == 1
    assert sum(e["nChanges"] for e in ov["endpoints"]) == ov["stats"]["changes_total"]


def test_directory_status_ignores_a_recompute_change_absent_from_changes_json(
    fake_site,
):
    """status/lastChange read the same canonical list as nChanges. Reading them
    from lt_scores.json instead put a last-change date next to a zero change
    count -- or "changed" next to a zero one -- on the very same row."""
    (fake_site / "data" / "changes.json").write_text(json.dumps([]))

    ov = _build_overview(fake_site)
    ep = next(e for e in ov["endpoints"] if e["slug"] == "m2fa23p")
    assert ep["nChanges"] == 0
    assert ep["status"] == "stable"
    assert ep["lastChange"] is None
    assert ep["recent"] is False


def test_directory_status_follows_a_canonical_change_absent_from_the_recompute(
    fake_site,
):
    scores_path = fake_site / "data" / "lt" / "m2fa23p" / "lt_scores.json"
    scores = json.loads(scores_path.read_text())
    scores["changes"] = []
    scores_path.write_text(json.dumps(scores))

    ov = _build_overview(fake_site)
    ep = next(e for e in ov["endpoints"] if e["slug"] == "m2fa23p")
    assert ep["nChanges"] == 1
    assert ep["status"] == "changed"
    assert ep["lastChange"] == "2026-06-25"


def test_b3it_row_status_counts_an_epoch_closure_change(tmp_path):
    """A closure-only change (`change_detected`, no derived onset in the view)
    reaches changes.json through merge_changes, so the row must not read stable."""
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(
        root, "m2fa23p", "m/a", "p", dates=dates, changes=[], drift=[0.1] * 30
    )
    write_b3it_series(
        root,
        "m/b",
        "q",
        status="monitoring",
        retired=None,
        month="2026-06",
        tokens=["A"] * 24,  # a flat series: the view derives no onset of its own
    )
    slug = slugify("m/b#q")
    (root / "data" / "changes.json").write_text(
        json.dumps(
            [
                {
                    "date": "2026-06-20T00:00:00+00:00",
                    "slug": slug,
                    "model": "m/b",
                    "provider": "q",
                    "method": "B3IT",
                    "magnitude": None,
                }
            ]
        )
    )

    ov = _build_overview(root)
    ep = next(e for e in ov["endpoints"] if e["slug"] == slug)
    assert ep["nChanges"] == 1
    assert ep["status"] == "changed"
    assert ep["lastChange"] == "2026-06-20"


def test_ages_are_measured_against_the_build_clock(tmp_path):
    """The site clock is the build's own, not the newest observation: an endpoint
    nobody has sampled for three weeks is retired even if it is the freshest one
    the build has, and the "as of" date is the build day."""
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 11)]
    write_lt_endpoint(
        root, "m2fa23p", "m/a", "p", dates=dates, changes=[], drift=[0.1] * 10
    )
    (root / "data" / "changes.json").write_text(json.dumps([]))

    now = datetime(2026, 7, 1, tzinfo=timezone.utc)
    ov = _build_overview_with(root, empty_status_inputs(), now)
    assert ov["stats"]["now"] == "2026-07-01"
    row = next(e for e in ov["endpoints"] if e["slug"] == "m2fa23p")
    assert row["status"] == "retired"


def test_stalled_lt_endpoint_with_fresh_b3it_sampling_is_not_retired(tmp_path):
    """The row's pill and the B3IT status line share a page: an endpoint whose LT
    queries stopped weeks ago but whose border inputs are sampled daily is still
    being watched, and must not read Retired above "B3IT: actively monitored"."""
    root = tmp_path / "website"
    old_dates = [f"2026-05-{d:02d}T00:00:00Z" for d in range(1, 11)]
    write_lt_endpoint(
        root, "m2fb23q", "m/b", "q", dates=old_dates, changes=[], drift=[0.1] * 10
    )
    write_b3it_series(
        root,
        "m/b",
        "q",
        status="monitoring",
        retired=None,
        month="2026-06",
        tokens=["A"] * 24,
    )
    (root / "data" / "changes.json").write_text(json.dumps([]))

    now = datetime(2026, 6, 25, tzinfo=timezone.utc)
    ov = _build_overview_with(root, empty_status_inputs(), now)
    row = next(e for e in ov["endpoints"] if e["slug"] == "m2fb23q")
    assert row["methods"] == ["lt", "b3it"]
    assert (row["status"], row["lastChange"], row["recent"]) == ("stable", None, False)


def test_endpoint_rows_carry_model_slug(fake_site):
    ov = _build_overview(fake_site)
    ep = next(e for e in ov["endpoints"] if e["slug"] == "m2fa23p")
    assert ep["modelSlug"] == slugify("m/a")


def test_endpoint_rows_carry_provider_slug_of_the_base_provider(tmp_path):
    """The link target is the provider *page*, which is keyed by company, not variant."""
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(
        root, "m2fa23p2ffp8", "m/a", "p/fp8", dates=dates, changes=[], drift=[0.1] * 30
    )
    (root / "data" / "changes.json").write_text(json.dumps([]))
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))

    ov = _build_overview(root)
    ep = next(e for e in ov["endpoints"] if e["provider"] == "p/fp8")
    assert ep["providerSlug"] == slugify("p") == "p"


def test_stats_count_provider_companies_and_variants(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(
        root, "a23p", "org/a", "p", dates=dates, changes=[], drift=[0.1] * 30
    )
    write_lt_endpoint(
        root, "b23p2ffp8", "org/b", "p/fp8", dates=dates, changes=[], drift=[0.1] * 30
    )
    (root / "data" / "changes.json").write_text(json.dumps([]))
    ov = _build_overview(root)
    assert ov["stats"]["providers"] == 2  # serving variants
    assert ov["stats"]["provider_companies"] == 1


def test_stats_carry_the_last_query_of_each_method(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(
        root, "m2fa23p", "m/a", "p", dates=dates, changes=[], drift=[0.1] * 30
    )
    _write_b3it_with_transition(root, "m/b", "q", status="monitoring")

    stats = _build_overview(root)["stats"]
    assert stats["last_query_lt"] == "2026-06-30T00:00:00Z"
    assert stats["last_query_b3it"] == "2026-01-24T00:00:00Z"


def test_stats_last_query_is_none_for_a_method_with_no_data(fake_site):
    """The fixture's b3it endpoint has a state file but no phase-2 results."""
    stats = _build_overview(fake_site)["stats"]
    assert stats["last_query_lt"] == "2026-06-30T00:00:00Z"
    assert stats["last_query_b3it"] is None


def test_stats_counts_match_endpoint_and_changes_lists(fake_site):
    ov = _build_overview(fake_site)
    assert ov["stats"]["endpoints"] == len(ov["endpoints"])
    assert ov["stats"]["catalog_endpoints"] == len(ov["endpoints"])
    assert ov["stats"]["lt_endpoints"] == 1
    assert ov["stats"]["b3it_endpoints"] == 1
    assert ov["stats"]["spend_cumulative"] == 1.23


def test_untracked_catalog_endpoint_gets_a_row_with_reason(fake_site):
    inputs = empty_status_inputs()
    inputs.catalog = [
        catalog_entry(
            "openai/gpt-5.4",
            "openai",
            supports_temperature=False,
            supports_logprobs=False,
        )
    ]
    inputs.bi_cache.add_bad_temperature(
        Endpoint(
            api="openrouter", model="openai/gpt-5.4", provider="openai", cost=(1, 2)
        )
    )
    ov = _build_overview_with(fake_site, inputs, NOW)
    row = next(
        r for r in ov["endpoints"] if r["slug"] == slugify("openai/gpt-5.4#openai")
    )
    assert row["methods"] == [] and row["trace"] == []
    assert row["model"] == "gpt-5.4" and row["org"] == "openai"
    assert row["providerSlug"] == "openai"
    assert row["headline"] == "untrackable"
    assert row["headlines"] == ["untrackable"]
    assert (row["ltStatus"], row["biStatus"]) == ("no_logprobs", "bad_temperature")
    assert row["reason"] == STATUS_COPY["untrackable"]
    # untracked rows do not join the tracked-fleet stats, only the catalog count
    assert ov["stats"]["endpoints"] == 1
    assert ov["stats"]["catalog_endpoints"] == 2


def test_tracked_rows_carry_status_fields(fake_site):
    inputs = empty_status_inputs()
    inputs.endpoints_lt = [
        Endpoint(api="openrouter", model="m/a", provider="p", cost=(1, 2))
    ]
    ov = _build_overview_with(fake_site, inputs, NOW)
    row = next(r for r in ov["endpoints"] if r["slug"] == "m2fa23p")
    assert row["headline"] == "tracked"
    assert row["headlines"] == ["tracked"]
    assert (row["ltStatus"], row["biStatus"]) == ("tracked", "monitoring")
    assert row["reason"] == STATUS_COPY["tracked"]


def test_a_dead_too_expensive_endpoint_is_also_retired(tmp_path):
    """An endpoint B3IT retired for cost, whose series then went quiet: its
    dominant headline stays too_expensive (badges, model pages), but it carries
    "retired" as well, so the Retired chip lists it, its reason says both, and
    the fleet count counts it once."""
    root = tmp_path / "website"
    _write_b3it_with_transition(
        root,
        "m/b",
        "q",
        status="retired",
        retired=RetiredInfo(
            reason="too_expensive",
            since=datetime(2026, 1, 25, tzinfo=timezone.utc),
            last_recheck=datetime(2026, 1, 25, tzinfo=timezone.utc),
        ),
    )
    (root / "data" / "changes.json").write_text(json.dumps([]))
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))

    ov = _build_overview(root)
    row = next(r for r in ov["endpoints"] if r["slug"] == slugify("m/b#q"))
    assert (row["status"], row["headline"]) == ("retired", "too_expensive")
    assert row["headlines"] == ["retired", "too_expensive"]
    assert row["reason"].startswith("Monitoring was retired: a single query costs")
    assert ov["stats"]["endpoints"] == 1
    assert ov["stats"]["active"] == 0


def test_pinned_hero_reaches_overview_json(tmp_path):
    """End-to-end: the configured pin, not a per-build scoring pass, decides the
    hero -- and it arrives at full daily resolution."""
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(
        root,
        "m2fa23p",
        "m/a",
        "p",
        dates=dates,
        changes=[{"index": 20, "sigma": 99.0}],
        drift=[0.05] * 20 + [1.4] * 10,
    )
    changes = [
        {
            "date": dates[20],
            "slug": "m2fa23p",
            "model": "m/a",
            "provider": "p",
            "method": "LT",
            "magnitude": 99.0,
        }
    ]
    (root / "data" / "changes.json").write_text(json.dumps(changes))

    lt_dir = root / "data" / "lt"
    lt_endpoints = list(discover_lt_endpoints(lt_dir))
    lt_data = load_all_lt_data(lt_dir, [e.slug for e in lt_endpoints])
    pin = HeroConfig(
        slug="m2fa23p", method="lt", start=date(2026, 6, 1), end=date(2026, 6, 30)
    )
    ov = build_overview(
        root,
        lt_data,
        lt_endpoints,
        {},
        pin,
        site_statuses_for(root, empty_status_inputs()),
        NOW,
    )

    hero = ov["hero"]
    assert hero["slug"] == "m2fa23p"
    assert hero["start"] == "2026-06-01" and hero["end"] == "2026-06-30"
    assert len(hero["values"]) == 30, "the series was downsampled or clipped"
    assert hero["magnitude"] == 1.4 and hero["baseline"] == 0.05


def test_stale_hero_pin_fails_the_build(tmp_path):
    root = tmp_path / "website"
    write_lt_endpoint(
        root,
        "m2fa23p",
        "m/a",
        "p",
        dates=[f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)],
        changes=[],
        drift=[0.05] * 30,
    )
    lt_dir = root / "data" / "lt"
    lt_endpoints = list(discover_lt_endpoints(lt_dir))
    lt_data = load_all_lt_data(lt_dir, [e.slug for e in lt_endpoints])
    pin = HeroConfig(
        slug="gone2fmissing", method="lt", start=date(2026, 6, 1), end=date(2026, 6, 30)
    )
    with pytest.raises(ValueError, match="hero pin"):
        build_overview(
            root,
            lt_data,
            lt_endpoints,
            {},
            pin,
            site_statuses_for(root, empty_status_inputs()),
            NOW,
        )
