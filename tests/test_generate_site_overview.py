import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from trackllm_website.bi.phase_2 import save_results
from trackllm_website.bi.state import EndpointBIState, Epoch, RetiredInfo
from trackllm_website.config import Endpoint
from trackllm_website.generate_site.b3it import discover_b3it_views
from trackllm_website.generate_site.lt import discover_lt_endpoints
from trackllm_website.generate_site.overview import build_overview, downsample_trace


def _build_overview(root: Path) -> dict:
    lt_dir = root / "data" / "lt"
    lt_endpoints = list(discover_lt_endpoints(lt_dir)) if lt_dir.exists() else []
    b3it_views = discover_b3it_views(
        root / "data" / "b3it" / "state", root / "data" / "b3it" / "phase_2"
    )
    return build_overview(root, lt_endpoints, b3it_views)


def test_downsample_trace_caps_length():
    assert len(downsample_trace(list(range(200)), 28)) == 28


def test_downsample_trace_short_input_untouched():
    assert downsample_trace([1.0, 2.0], 28) == [1.0, 2.0]


def test_downsample_trace_empty():
    assert downsample_trace([], 28) == []


def _write_lt_endpoint(
    root: Path, slug: str, model: str, provider: str, *, dates, changes, drift
):
    d = root / "data" / "lt" / slug
    prompt_dir = d / "default"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "info.json").write_text(
        json.dumps({"prompt": "hi", "endpoint": {"model": model, "provider": provider}})
    )
    month = dates[-1][:7]
    day = dates[-1][8:10]
    month_dir = prompt_dir / month
    month_dir.mkdir()
    (month_dir / "queries.json").write_text(json.dumps([[f"{day} 00:00:00", 0]]))
    scores = [0.5] * len(dates)
    (d / "lt_scores.json").write_text(
        json.dumps(
            {
                "n_per_test": 24,
                "dates": dates,
                "scores": scores,
                "sigmas": [None] * len(dates),
                "changes": changes,
                "drift_dates": dates,
                "drift": drift,
            }
        )
    )


def _write_b3it_state(
    root: Path, slug: str, model: str, provider: str, *, status="monitoring"
):
    state = {
        "endpoint": {
            "api": "openrouter",
            "model": model,
            "provider": provider,
            "cost": [0.1, 0.2],
            "max_logprobs": None,
        },
        "status": status,
        "retired": None,
        "epochs": [
            {
                "start": "2026-01-01T00:00:00Z",
                "border_inputs": [],
                "reference": {},
                "end": None,
            }
        ],
    }
    sd = root / "data" / "b3it" / "state"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / f"{slug}.json").write_text(json.dumps(state))


def _daily_batch(day: int, token: str):
    ts = f"2026-01-{day:02d}T00:00:00+00:00"
    return ts, [(ts, token)] * 10


def _write_b3it_with_transition(
    root: Path, slug: str, model: str, provider: str, *, status, retired=None
):
    """A b3it endpoint whose reference actually produces a TV transition."""
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
        status=status,
        retired=retired,
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


@pytest.fixture
def fake_site(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    drift = [0.1] * 24 + [1.5] * 6
    changes = [{"index": 24, "sigma": 40.0}]
    _write_lt_endpoint(
        root, "m2fa23p", "m/a", "p", dates=dates, changes=changes, drift=drift
    )
    _write_b3it_state(root, "m2fa23p", "m/a", "p")

    (root / "data" / "changes.json").write_text(
        json.dumps(
            [
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
        )
    )
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {"lt": 1.23}}))
    return root


def test_build_overview_shape(fake_site):
    ov = _build_overview(fake_site)
    assert set(ov) == {"stats", "feed", "providers", "endpoints"}
    ep = ov["endpoints"][0]
    assert set(ep) >= {
        "slug",
        "model",
        "provider",
        "methods",
        "status",
        "nChanges",
        "trace",
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
    assert ep["stableDays"] is not None


def test_status_retired_when_no_recent_observation(tmp_path):
    root = tmp_path / "website"
    old_dates = [f"2025-01-{d:02d}T00:00:00Z" for d in range(1, 29)]
    recent_dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 29)]
    _write_lt_endpoint(
        root, "old2fa23p", "old/a", "p", dates=old_dates, changes=[], drift=[0.1] * 28
    )
    _write_lt_endpoint(
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
    assert lt_item["primary"] == "drift 1.5"
    assert lt_item["secondary"] == "40σ conf"
    assert lt_item["sevKey"] == "alert"
    assert (
        lt_item["desc"] == "Logprob averages moved 1.5 nats from the reference period."
    )
    assert len(lt_item["trace"]) > 0
    assert lt_item["model"] == "a"
    assert lt_item["provider"] == "p"


def test_feed_includes_b3it_item_from_view_transition(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 6)]
    _write_lt_endpoint(
        root, "m2fa23p", "m/a", "p", dates=dates, changes=[], drift=[0.1] * 5
    )
    _write_b3it_with_transition(root, "m2fb23q", "m/b", "q", status="monitoring")
    (root / "data" / "changes.json").write_text(json.dumps([]))
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


def test_b3it_only_retired_endpoint_gets_retired_status(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 6)]
    _write_lt_endpoint(
        root, "m2fa23p", "m/a", "p", dates=dates, changes=[], drift=[0.1] * 5
    )
    _write_b3it_with_transition(
        root,
        "m2fb23q",
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

    ov = _build_overview(root)
    ep = next(e for e in ov["endpoints"] if e["slug"] == "m2fb23q")
    assert ep["methods"] == ["b3it"]
    assert ep["status"] == "retired"
    assert len(ep["trace"]) > 0
    assert ov["stats"]["active"] == 1  # only the still-monitoring LT endpoint


def test_providers_include_zero_change_providers(fake_site):
    root = fake_site
    long_span = [
        (datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=d)).strftime(
            "%Y-%m-%dT00:00:00Z"
        )
        for d in range(0, 220)
    ]
    _write_lt_endpoint(
        root,
        "m2fc23zeroprov",
        "m/c",
        "zeroprov",
        dates=long_span,
        changes=[],
        drift=[0.0] * len(long_span),
    )
    ov = _build_overview(root)
    prov_names = {p["name"] for p in ov["providers"]}
    assert "zeroprov" in prov_names
    zp = next(p for p in ov["providers"] if p["name"] == "zeroprov")
    assert zp["n_changes"] == 0
    assert 0.0 <= zp["conf"] <= 1.0


def test_stats_counts_match_endpoint_and_changes_lists(fake_site):
    ov = _build_overview(fake_site)
    assert ov["stats"]["endpoints"] == len(ov["endpoints"])
    assert ov["stats"]["lt_endpoints"] == 1
    assert ov["stats"]["b3it_endpoints"] == 1
    assert ov["stats"]["spend_cumulative"] == 1.23
