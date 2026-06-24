import json
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
    from trackllm_website.generate_site.render import render_site

    render_site(tmp_path)
    assert (tmp_path / "data" / "b3it" / "b2fx23q" / "b3it.json").exists()
    assert (tmp_path / "endpoints" / "b2fx23q.html").exists()
