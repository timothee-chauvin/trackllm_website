import json
import pytest
import shutil
from pathlib import Path

from trackllm_website.generate_site.render import render_site


def _scaffold(website: Path):
    # copy real templates + style so rendering matches production
    src = Path("website")
    (website / "templates").mkdir(parents=True)
    for t in (src / "templates").glob("*.j2"):
        shutil.copy(t, website / "templates" / t.name)
    (website / "style.css").write_text((src / "style.css").read_text())
    ep = website / "data" / "lt" / "m2fa23p" / "default"
    ep.mkdir(parents=True)
    ep_info = {"prompt": "hi", "endpoint": {"model": "m/a", "provider": "p"}}
    (ep / "info.json").write_text(json.dumps(ep_info))
    md = ep / "2026-06"
    md.mkdir()
    (md / "queries.json").write_text(json.dumps([["24 10:00:00", 0]]))


def test_render_site_produces_index_and_endpoint(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path)
    index = (tmp_path / "index.html").read_text()
    assert "m/a" in index
    assert (tmp_path / "endpoints" / "m2fa23p.html").exists()


def test_render_emits_changes_and_unified_index(tmp_path):
    _scaffold(tmp_path)
    import json

    (tmp_path / "data" / "lt" / "lt_changes.json").write_text(
        json.dumps(
            {
                "m2fa23p": [
                    {
                        "endpoint": "m2fa23p",
                        "index": 3,
                        "date": "2026-06-20T00:00:00Z",
                        "sigma": 9.0,
                        "first_detected": "2026-06-21T00:00:00Z",
                    }
                ]
            }
        )
    )

    render_site(tmp_path)
    assert (tmp_path / "data" / "changes.json").exists()
    index = (tmp_path / "index.html").read_text()
    assert "m/a" in index  # endpoint row
    assert "2026-06-20" in index  # change feed entry


def test_render_emits_b3it_json_and_b3it_only_page(tmp_path):
    _scaffold(tmp_path)
    sd = tmp_path / "data" / "b3it" / "state"
    sd.mkdir(parents=True)
    state = {
        "endpoint": {
            "api": "openrouter",
            "model": "b/x",
            "provider": "q",
            "cost": [0.1, 0.2],
            "max_logprobs": None,
        },
        "status": "monitoring",
        "retired": None,
        "epochs": [
            {
                "start": "2026-06-01T00:00:00Z",
                "border_inputs": [],
                "reference": {},
                "end": None,
            }
        ],
    }
    (sd / "b2fx23q.json").write_text(json.dumps(state))

    render_site(tmp_path)
    assert (tmp_path / "data" / "b3it" / "b2fx23q" / "b3it.json").exists()
    assert (tmp_path / "endpoints" / "b2fx23q.html").exists()

    b3it_data = json.loads(
        (tmp_path / "data" / "b3it" / "b2fx23q" / "b3it.json").read_text()
    )
    assert b3it_data["status"] == "monitoring"

    page_html = (tmp_path / "endpoints" / "b2fx23q.html").read_text()
    # The model must appear in the visible header (h1/title), not just the manifest JSON script tag.
    assert "<h1>" in page_html
    assert (
        "b/x" in page_html.split("</h1>")[0].split("<h1>")[-1]
        or "b/x" in page_html.split("</title>")[0].split("<title>")[-1]
    )


def test_render_emits_spend(tmp_path):
    _scaffold(tmp_path)
    import json

    sp = tmp_path / "data" / "spend" / "m2fa23p"
    sp.mkdir(parents=True)
    (sp / "2026-06.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-06-24T00:00:00Z",
                "kind": "lt",
                "cost": 0.05,
                "n_queries": 1,
                "n_errors": 0,
            }
        )
        + "\n"
    )
    from trackllm_website.generate_site.render import render_site

    render_site(tmp_path)
    assert (tmp_path / "data" / "spend.json").exists()
    assert (tmp_path / "spend.html").exists()
    assert "spend" in (tmp_path / "index.html").read_text().lower()

    # Assert spend data renders with correct cost value
    spend_html = (tmp_path / "spend.html").read_text()
    assert "$0.0500" in spend_html, "Cost should render as $0.0500 (4 decimal places)"
    assert "m2fa23p" in spend_html, "Endpoint slug should appear in spend table"

    # Assert emitted spend.json has expected cumulative cost
    spend_data = json.loads((tmp_path / "data" / "spend.json").read_text())
    assert spend_data["cumulative"]["lt"] == pytest.approx(0.05)
