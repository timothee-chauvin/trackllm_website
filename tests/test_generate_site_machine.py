"""The markdown mirror and the Atom feeds (generate_site/machine.py)."""

import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from conftest import empty_status_inputs, write_lt_endpoint
from trackllm_website.generate_site.machine import links
from trackllm_website.generate_site.render import render_site

ATOM = "{http://www.w3.org/2005/Atom}"
DATES = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 25)]
CHANGE_DATE = "2026-06-20T00:00:00Z"


def _scaffold(website: Path) -> None:
    src = Path("website")
    shutil.copytree(src / "templates", website / "templates")
    (website / "style.css").write_text((src / "style.css").read_text())
    write_lt_endpoint(
        website, "m2fa23p", "m/a", "p", dates=DATES, changes=[], drift=[0.1] * 24
    )
    write_lt_endpoint(
        website, "m2fb23q", "m/b", "q", dates=DATES, changes=[], drift=[0.1] * 24
    )
    (website / "data" / "lt" / "lt_changes.json").write_text(
        json.dumps(
            {
                "m2fa23p": [
                    {
                        "endpoint": "m2fa23p",
                        "index": 19,
                        "date": CHANGE_DATE,
                        "sigma": 9.0,
                        "first_detected": "2026-06-21T00:00:00Z",
                    }
                ]
            }
        )
    )


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("site")
    _scaffold(root)
    render_site(root, None, empty_status_inputs())
    return root


def _md_links(text: str) -> list[str]:
    return [
        h
        for h in re.findall(r"\]\(([^)\s]+)\)", text)
        if not h.startswith(("http://", "https://", "#"))
    ]


def test_every_html_page_has_a_markdown_twin(site: Path):
    pages = [p for p in site.rglob("*.html") if "templates" not in p.parts]
    assert pages
    for html in pages:
        assert html.with_suffix(".md").exists(), html


def test_every_markdown_link_resolves(site: Path):
    for md in site.rglob("*.md"):
        if "templates" in md.parts or "data" in md.parts:
            continue
        for href in _md_links(md.read_text()):
            target = (md.parent / href.split("#")[0]).resolve()
            assert target.exists(), f"{md.relative_to(site)}: {href}"


def test_markdown_header_names_html_feed_and_json(site: Path):
    md = (site / "endpoints" / "m2fa23p.md").read_text()
    assert "https://www.trackllm.net/endpoints/m2fa23p.html" in md
    assert "../feeds/endpoints/m2fa23p.xml" in md
    assert "../data/models/m2fa.json" in md
    assert "trackllm_data" in md
    # the chart is a link to its series, not a table
    assert "Full series:" in md
    # the detected change is listed
    assert "| 2026-06-20 | LT |" in md


def test_html_declares_its_twins(site: Path):
    html = (site / "endpoints" / "m2fa23p.html").read_text()
    assert 'rel="alternate" type="text/markdown" href="../endpoints/m2fa23p.md"' in html
    assert 'href="../feeds/endpoints/m2fa23p.xml"' in html
    assert 'data-goatcounter-click="subscribe/endpoint/m2fa23p"' in html
    index = (site / "index.html").read_text()
    assert 'id="machine"' in index
    assert 'href="feeds/all.xml"' in index


def _entries(site: Path, path: str) -> list[ET.Element]:
    root = ET.parse(site / path).getroot()
    assert root.tag == f"{ATOM}feed"
    for tag in ("title", "id", "updated"):
        assert root.find(f"{ATOM}{tag}") is not None, tag
    return root.findall(f"{ATOM}entry")


def test_feeds_are_scoped_exactly(site: Path):
    assert len(_entries(site, "feeds/all.xml")) == 1
    assert len(_entries(site, "feeds/endpoints/m2fa23p.xml")) == 1
    assert len(_entries(site, "feeds/endpoints/m2fb23q.xml")) == 0
    assert len(_entries(site, "feeds/models/m2fa.xml")) == 1
    assert len(_entries(site, "feeds/models/m2fb.xml")) == 0
    assert len(_entries(site, "feeds/providers/p.xml")) == 1
    assert len(_entries(site, "feeds/providers/q.xml")) == 0
    assert len(_entries(site, "feeds/orgs/m.xml")) == 1


def test_feed_entry_links_endpoint_page_markdown_and_json(site: Path):
    (entry,) = _entries(site, "feeds/all.xml")
    hrefs = {
        link.get("type"): link.get("href") for link in entry.findall(f"{ATOM}link")
    }
    assert hrefs["text/html"] == "https://www.trackllm.net/endpoints/m2fa23p.html"
    assert hrefs["text/markdown"] == "https://www.trackllm.net/endpoints/m2fa23p.md"
    assert hrefs["application/json"] == "https://www.trackllm.net/data/models/m2fa.json"
    assert entry.find(f"{ATOM}updated").text == CHANGE_DATE
    assert "m/a @ p: LT change" in entry.find(f"{ATOM}title").text


def test_stale_outputs_are_pruned(site: Path):
    stale_md = site / "endpoints" / "gone.md"
    stale_feed = site / "feeds" / "models" / "gone.xml"
    stale_md.write_text("x")
    stale_feed.write_text("x")
    render_site(site, None, empty_status_inputs())
    assert not stale_md.exists()
    assert not stale_feed.exists()


def test_links_top_level_pages_only_index_has_a_feed():
    assert links("index", "", ["data/overview.json"])["feed"] == "feeds/all.xml"
    assert links("changes", "", [])["feed"] is None
    assert links("model", "x", [])["feed"] == "feeds/models/x.xml"
