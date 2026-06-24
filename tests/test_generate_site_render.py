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
