import json
import pytest
import shutil
from pathlib import Path

from trackllm_website.generate_site.render import render_site
from trackllm_website.util import slugify


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
    # index.html is now a static shell; the directory is populated client-side from overview.json
    assert 'id="dirBody"' in index
    assert (tmp_path / "endpoints" / "m2fa23p.html").exists()
    overview = json.loads((tmp_path / "data" / "overview.json").read_text())
    assert "a" in {e["model"] for e in overview["endpoints"]}


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
    # index.html is a static shell now; the directory + feed are populated client-side
    # from overview.json / changes.json rather than server-rendered into index.html.
    changes = json.loads((tmp_path / "data" / "changes.json").read_text())
    assert any("2026-06-20" in c["date"] for c in changes)  # change feed entry
    overview = json.loads((tmp_path / "data" / "overview.json").read_text())
    assert "a" in {e["model"] for e in overview["endpoints"]}  # endpoint row


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


def _spend_line(kind, cost):
    return json.dumps(
        {
            "timestamp": "2026-06-24T00:00:00Z",
            "kind": kind,
            "cost": cost,
            "n_queries": 1,
            "n_errors": 0,
        }
    )


def test_render_emits_spend(tmp_path):
    _scaffold(tmp_path)
    sp = tmp_path / "data" / "spend" / "m2fa23p"
    sp.mkdir(parents=True)
    (sp / "2026-06.jsonl").write_text(_spend_line("lt", 0.05) + "\n")
    # an all-error endpoint: real data ($0 billed), must not render as "no data"
    zp = tmp_path / "data" / "spend" / "zero2fcost23ep"
    zp.mkdir(parents=True)
    (zp / "2026-06.jsonl").write_text(_spend_line("lt", 0.0) + "\n")

    render_site(tmp_path)
    assert (tmp_path / "data" / "spend.json").exists()
    assert (tmp_path / "spend.html").exists()
    assert "spend" in (tmp_path / "index.html").read_text().lower()

    # Assert spend data renders with correct cost value
    spend_html = (tmp_path / "spend.html").read_text()
    assert "$0.0500" in spend_html, "Cost should render as $0.0500 (4 decimal places)"
    assert "m2fa23p" in spend_html, "Endpoint slug should appear in spend table"

    # Zero-billed group renders as $0.0000 (lt cell + total), not as the no-data dash
    zero_row = next(r for r in spend_html.split("<tr>") if "zero2fcost23ep" in r)
    assert zero_row.count("$0.0000") == 2

    # Assert emitted spend.json has expected cumulative cost
    spend_data = json.loads((tmp_path / "data" / "spend.json").read_text())
    assert spend_data["cumulative"]["lt"] == pytest.approx(0.05)


def test_render_endpoint_page_context_for_multi_provider_model(tmp_path):
    _scaffold(tmp_path)
    # second endpoint for the same model ("m/a") but a different provider, so
    # the model is served by 2 providers.
    ep2 = tmp_path / "data" / "lt" / "m2fa23p2" / "default"
    ep2.mkdir(parents=True)
    ep2_info = {"prompt": "hi", "endpoint": {"model": "m/a", "provider": "p2"}}
    (ep2 / "info.json").write_text(json.dumps(ep2_info))
    md2 = ep2 / "2026-06"
    md2.mkdir()
    (md2 / "queries.json").write_text(json.dumps([["24 10:00:00", 0]]))

    render_site(tmp_path)

    model_slug = slugify("m/a")
    page = (tmp_path / "endpoints" / "m2fa23p.html").read_text()

    # nav_prefix="../" applied to a nav/crumb link
    assert 'href="../index.html"' in page
    # compare banner links to the model page with the model's own slug
    assert f'href="../models/{model_slug}.html"' in page
    # n_providers count text, sourced from view["n_providers"]
    assert "served by 2 providers" in page
