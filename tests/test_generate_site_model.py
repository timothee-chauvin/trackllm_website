import json
from datetime import datetime, timezone
from pathlib import Path

from conftest import write_lt_endpoint
from trackllm_website.bi.phase_2 import save_results
from trackllm_website.bi.state import EndpointBIState, Epoch
from trackllm_website.config import Endpoint
from trackllm_website.generate_site.b3it import discover_b3it_views
from trackllm_website.generate_site.changes import merge_changes, to_json
from trackllm_website.generate_site.lt import discover_lt_endpoints, load_lt_data
from trackllm_website.generate_site.model import build_model_views
from trackllm_website.util import slugify


def _write_changes_json(root: Path) -> None:
    """The canonical merged list, as render.py writes it before model views are
    built. Here the detector's events agree with the recompute in lt_scores.json;
    where they don't, changes.json wins.
    """
    lt_dir = root / "data" / "lt"
    lt_by_slug = {e.slug: e for e in discover_lt_endpoints(lt_dir)}
    lt_changes = {}
    for slug in lt_by_slug:
        data = load_lt_data(lt_dir, slug)
        if data is None:
            continue
        lt_changes[slug] = [
            {"date": data.dates[c["index"]].isoformat(), "sigma": c["sigma"]}
            for c in data.changes
        ]
    b3it_views = discover_b3it_views(
        root / "data" / "b3it" / "state", root / "data" / "b3it" / "phase_2"
    )
    events = merge_changes(lt_changes, lt_by_slug, b3it_views)
    (root / "data" / "changes.json").write_text(json.dumps(to_json(events)))


def _build_model_views(root: Path) -> dict:
    lt_dir = root / "data" / "lt"
    lt_endpoints = list(discover_lt_endpoints(lt_dir)) if lt_dir.exists() else []
    b3it_views = discover_b3it_views(
        root / "data" / "b3it" / "state", root / "data" / "b3it" / "phase_2"
    )
    return build_model_views(root, lt_endpoints, b3it_views)


def _daily_batch(day: int, token: str):
    ts = f"2026-01-{day:02d}T00:00:00+00:00"
    return ts, [(ts, token)] * 10


def _write_b3it_with_transition(root: Path, slug: str, model: str, provider: str):
    ep = Endpoint(api="openrouter", model=model, provider=provider, cost=(0.1, 0.2))
    ref = {"p1": [("2026-01-01T00:00:00Z", "A")] * 10}
    results = {
        "p1": dict(
            [_daily_batch(d, "A") for d in range(1, 13)]
            + [_daily_batch(d, "B") for d in range(13, 25)]
        )
    }
    state = EndpointBIState(
        endpoint=ep,
        status="monitoring",
        retired=None,
        epochs=[
            Epoch(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                border_inputs=["p1"],
                reference=ref,
            )
        ],
    )
    state.save(root / "data" / "b3it" / "state")
    p2_dir = root / "data" / "b3it" / "phase_2" / slug
    p2_dir.mkdir(parents=True)
    save_results(p2_dir / "p1.json", results)


def test_build_model_views_groups_two_providers_of_one_model(tmp_path):
    root = tmp_path / "website"
    dates_a = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 21)]
    dates_b = [f"2026-06-{d:02d}T00:00:00Z" for d in range(5, 25)]
    write_lt_endpoint(
        root,
        "m2fa23p1",
        "m/a",
        "p1",
        dates=dates_a,
        changes=[{"index": 15, "sigma": 12.0}],
        drift=[0.1] * 15 + [1.2] * 5,
    )
    write_lt_endpoint(
        root,
        "m2fa23p2",
        "m/a",
        "p2",
        dates=dates_b,
        changes=[],
        drift=[0.2] * len(dates_b),
    )
    _write_changes_json(root)

    views = _build_model_views(root)
    modelslug = slugify("m/a")
    assert modelslug in views
    view = views[modelslug]

    assert view["model"] == "m/a"
    assert view["org"] == "m"
    assert view["n_endpoints"] == 2
    assert view["n_changed"] == 1
    assert {e["provider"] for e in view["endpoints"]} == {"p1", "p2"}
    assert view["date_min"] == min(dates_a[0][:10], dates_b[0][:10])
    assert view["date_max"] == max(dates_a[-1][:10], dates_b[-1][:10])

    ep1 = next(e for e in view["endpoints"] if e["provider"] == "p1")
    assert ep1["lt"] is not None
    assert ep1["lt"]["changes"][0]["sigma"] == "12σ"
    assert ep1["lt"]["changes"][0]["drift"] == 1.2
    assert ep1["n_changes"] == 1
    assert ep1["b3it"] is None

    ep2 = next(e for e in view["endpoints"] if e["provider"] == "p2")
    assert ep2["n_changes"] == 0


def test_build_model_views_includes_b3it_endpoint(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 6)]
    write_lt_endpoint(
        root, "m2fa23p1", "m/a", "p1", dates=dates, changes=[], drift=[0.1] * 5
    )
    _write_b3it_with_transition(root, "m2fa23p2", "m/a", "p2")
    _write_changes_json(root)

    views = _build_model_views(root)
    view = views[slugify("m/a")]
    assert view["n_endpoints"] == 2

    b3_ep = next(e for e in view["endpoints"] if e["provider"] == "p2")
    assert b3_ep["lt"] is None
    assert b3_ep["b3it"] is not None
    assert b3_ep["b3it"]["tv"], "expected a non-empty tv series"
    assert b3_ep["b3it"]["changes"], "expected a detected transition"
    assert b3_ep["b3it"]["changes"][0]["peakTV"] > 0
    assert b3_ep["n_changes"] == len(b3_ep["b3it"]["changes"])


def _two_variant_model(root: Path):
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 21)]
    write_lt_endpoint(
        root,
        "m2fa23chutes",
        "m/a",
        "chutes",
        dates=dates,
        changes=[{"index": 15, "sigma": 12.0}],
        drift=[0.1] * 15 + [1.2] * 5,
    )
    write_lt_endpoint(
        root,
        "m2fa23chutes2ffp8",
        "m/a",
        "chutes/fp8",
        dates=dates,
        changes=[{"index": 10, "sigma": 30.0}],
        drift=[0.1] * 10 + [0.9] * 10,
    )
    _write_changes_json(root)


def test_endpoint_and_provider_counts_are_separate_quantities(tmp_path):
    """Two serving variants of one company are two endpoints, one provider."""
    root = tmp_path / "website"
    _two_variant_model(root)
    view = _build_model_views(root)[slugify("m/a")]
    assert view["n_endpoints"] == 2
    assert view["n_providers"] == 1


def test_model_endpoints_carry_base_provider(tmp_path):
    root = tmp_path / "website"
    _two_variant_model(root)
    view = _build_model_views(root)[slugify("m/a")]
    assert {e["base"] for e in view["endpoints"]} == {"chutes"}
    assert {e["providerSlug"] for e in view["endpoints"]} == {"chutes"}
    assert {e["provider"] for e in view["endpoints"]} == {"chutes", "chutes/fp8"}


def test_model_view_has_flattened_change_list(tmp_path):
    root = tmp_path / "website"
    _two_variant_model(root)
    view = _build_model_views(root)[slugify("m/a")]
    dates = [c["date"] for c in view["changes"]]
    assert dates == sorted(dates)
    assert sum(e["n_changes"] for e in view["endpoints"]) == len(view["changes"])
    assert {c["method"] for c in view["changes"]} == {"lt"}
    assert {c["provider"] for c in view["changes"]} == {"chutes", "chutes/fp8"}


def test_change_count_follows_changes_json_not_the_recomputed_scores(tmp_path):
    """changes.json is canonical; the build-time recompute stored in lt_scores.json
    double-detects some changes on adjacent days, and those must neither be drawn
    on the strip nor counted in the tiles."""
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 21)]
    write_lt_endpoint(
        root,
        "m2fa23p1",
        "m/a",
        "p1",
        dates=dates,
        changes=[{"index": 15, "sigma": 12.0}, {"index": 16, "sigma": 9.0}],
        drift=[0.1] * 15 + [1.2] * 5,
    )
    (root / "data" / "changes.json").write_text(
        json.dumps(
            [
                {
                    "date": dates[15],
                    "slug": "m2fa23p1",
                    "model": "m/a",
                    "provider": "p1",
                    "method": "LT",
                    "magnitude": 12.0,
                    "magnitude_display": "12σ",
                }
            ]
        )
    )

    view = _build_model_views(root)[slugify("m/a")]
    ep = view["endpoints"][0]
    assert [c["date"] for c in ep["lt"]["changes"]] == [dates[15][:10]]
    assert ep["lt"]["changes"][0]["sigma"] == "12σ"
    assert ep["lt"]["changes"][0]["drift"] == 1.2
    assert ep["n_changes"] == 1
    assert view["n_changed"] == 1
    assert len(view["changes"]) == 1


def test_endpoint_with_no_lt_scores_file_yields_null_lt(tmp_path):
    root = tmp_path / "website"
    d = root / "data" / "lt" / "m2fa23p1" / "default"
    d.mkdir(parents=True)
    (d / "info.json").write_text(
        json.dumps({"prompt": "hi", "endpoint": {"model": "m/a", "provider": "p1"}})
    )
    md = d / "2026-06"
    md.mkdir()
    (md / "queries.json").write_text(json.dumps([["24 10:00:00", 0]]))
    # no lt_scores.json written
    _write_changes_json(root)

    views = _build_model_views(root)
    view = views[slugify("m/a")]
    ep = view["endpoints"][0]
    assert ep["methods"] == ["lt"]
    assert ep["lt"] is None
    assert ep["first"] is None and ep["last"] is None
    assert ep["n_changes"] == 0
