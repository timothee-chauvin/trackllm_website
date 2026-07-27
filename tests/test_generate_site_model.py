import json
from datetime import datetime, timezone
from pathlib import Path

from trackllm_website.bi.phase_2 import save_results
from trackllm_website.bi.state import EndpointBIState, Epoch
from trackllm_website.config import Endpoint
from trackllm_website.generate_site.model import build_model_views
from trackllm_website.util import slugify


def _write_lt_endpoint(root: Path, slug: str, model: str, provider: str, *, dates, changes, drift):
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
    _write_lt_endpoint(
        root,
        "m2fa23p1",
        "m/a",
        "p1",
        dates=dates_a,
        changes=[{"index": 15, "sigma": 12.0}],
        drift=[0.1] * 15 + [1.2] * 5,
    )
    _write_lt_endpoint(
        root, "m2fa23p2", "m/a", "p2", dates=dates_b, changes=[], drift=[0.2] * len(dates_b)
    )

    views = build_model_views(root)
    modelslug = slugify("m/a")
    assert modelslug in views
    view = views[modelslug]

    assert view["model"] == "m/a"
    assert view["org"] == "m"
    assert view["n_providers"] == 2
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
    _write_lt_endpoint(root, "m2fa23p1", "m/a", "p1", dates=dates, changes=[], drift=[0.1] * 5)
    _write_b3it_with_transition(root, "m2fa23p2", "m/a", "p2")

    views = build_model_views(root)
    view = views[slugify("m/a")]
    assert view["n_providers"] == 2

    b3_ep = next(e for e in view["endpoints"] if e["provider"] == "p2")
    assert b3_ep["lt"] is None
    assert b3_ep["b3it"] is not None
    assert b3_ep["b3it"]["tv"], "expected a non-empty tv series"
    assert b3_ep["b3it"]["changes"], "expected a detected transition"
    assert b3_ep["b3it"]["changes"][0]["peakTV"] > 0
    assert b3_ep["n_changes"] == len(b3_ep["b3it"]["changes"])


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

    views = build_model_views(root)
    view = views[slugify("m/a")]
    ep = view["endpoints"][0]
    assert ep["methods"] == ["lt"]
    assert ep["lt"] is None
    assert ep["first"] is None and ep["last"] is None
    assert ep["n_changes"] == 0
