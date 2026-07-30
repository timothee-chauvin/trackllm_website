import json
import pytest
import shutil
from datetime import datetime, timezone
from pathlib import Path

from conftest import (
    b3it_slug,
    catalog_entry,
    empty_status_inputs,
    write_b3it_series,
    write_b3it_state,
    write_lt_endpoint,
    write_month_dir,
)
from trackllm_website.config import Endpoint
from trackllm_website.generate_site.render import render_site
from trackllm_website.update_endpoints import LTFailure, LTFailureCache
from trackllm_website.util import slugify

DATES = [f"2026-06-{d:02d}T00:00:00Z" for d in range(20, 25)]


def _lt_endpoint(website: Path, slug: str, model: str, provider: str):
    write_lt_endpoint(
        website,
        slug,
        model,
        provider,
        dates=DATES,
        changes=[],
        drift=[0.1] * len(DATES),
    )


def _scaffold(website: Path):
    # copy real templates + style so rendering matches production
    src = Path("website")
    (website / "templates").mkdir(parents=True)
    for t in (src / "templates").glob("*.j2"):
        shutil.copy(t, website / "templates" / t.name)
    (website / "style.css").write_text((src / "style.css").read_text())
    # a series, not just a directory: an endpoint with no observations is not
    # rendered at all (tracked.py)
    _lt_endpoint(website, "m2fa23p", "m/a", "p")


def test_render_site_raises_when_data_dir_missing(tmp_path):
    # a missing data dir must fail the build, not print-and-deploy an empty site
    with pytest.raises(FileNotFoundError, match="does not exist"):
        render_site(tmp_path, None, empty_status_inputs())


def test_render_site_produces_index_and_endpoint(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path, None, empty_status_inputs())
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

    render_site(tmp_path, None, empty_status_inputs())
    # index.html is a static shell now; the directory + feed are populated client-side
    # from overview.json / changes.json rather than server-rendered into index.html.
    changes = json.loads((tmp_path / "data" / "changes.json").read_text())
    assert any("2026-06-20" in c["date"] for c in changes)  # change feed entry
    overview = json.loads((tmp_path / "data" / "overview.json").read_text())
    assert "a" in {e["model"] for e in overview["endpoints"]}  # endpoint row


def test_render_emits_b3it_json_and_b3it_only_page(tmp_path):
    _scaffold(tmp_path)
    write_b3it_series(
        tmp_path,
        "b/x",
        "q",
        status="monitoring",
        retired=None,
        month="2026-06",
        tokens=["A"] * 10,
    )
    assert b3it_slug("b/x", "q") == "b2fx23q"

    render_site(tmp_path, None, empty_status_inputs())
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

    render_site(tmp_path, None, empty_status_inputs())
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
    _lt_endpoint(tmp_path, "m2fa23p2", "m/a", "p2")

    render_site(tmp_path, None, empty_status_inputs())

    model_slug = slugify("m/a")
    page = (tmp_path / "endpoints" / "m2fa23p.html").read_text()

    # nav_prefix="../" applied to a nav/crumb link
    assert 'href="../index.html"' in page
    # compare banner links to the model page with the model's own slug
    assert f'href="../models/{model_slug}.html"' in page
    # endpoint count text, sourced from view["n_endpoints"] -- the banner counts
    # serving endpoints, which is not the same thing as provider companies
    assert "tracked on 2 endpoints" in page


def test_render_emits_provider_pages_and_data(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path, None, empty_status_inputs())
    view = json.loads((tmp_path / "data" / "providers" / "p.json").read_text())
    assert view["name"] == "p"
    assert view["n_endpoints"] == 1
    assert (tmp_path / "providers" / "p.html").exists()
    assert 'id="providerData"' in (tmp_path / "providers" / "p.html").read_text()


def test_overview_providers_are_base_provider_rows(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path, None, empty_status_inputs())
    overview = json.loads((tmp_path / "data" / "overview.json").read_text())
    (row,) = overview["providers"]
    assert row["name"] == "p"
    assert row["slug"] == "p"
    assert "lt_ci" in row


def test_render_emits_changes_page(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path, None, empty_status_inputs())
    page = json.loads((tmp_path / "data" / "changes_page.json").read_text())
    assert set(page) == {"stats", "items", "months", "top_endpoints"}
    assert (tmp_path / "changes.html").exists()
    assert 'id="log"' in (tmp_path / "changes.html").read_text()


def test_nav_links_to_changes(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path, None, empty_status_inputs())
    assert 'href="changes.html"' in (tmp_path / "index.html").read_text()


def test_render_emits_methodology_page(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path, None, empty_status_inputs())
    page = (tmp_path / "methodology.html").read_text()
    # both papers and the blog post must be reachable from the page
    assert "arxiv.org/abs/2512.03816" in page
    assert "arxiv.org/abs/2602.11083" in page
    assert "tchauvin.com/change-detection-llm-apis" in page
    assert 'href="methodology.html"' in (tmp_path / "index.html").read_text()


def test_favicon_link_is_relative_to_page_depth(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path, None, empty_status_inputs())
    model_slug = slugify("m/a")
    assert 'href="favicon.svg"' in (tmp_path / "index.html").read_text()
    assert (
        'href="../favicon.svg"' in (tmp_path / "endpoints" / "m2fa23p.html").read_text()
    )
    assert (
        'href="../favicon.svg"'
        in (tmp_path / "models" / f"{model_slug}.html").read_text()
    )


def test_endpoint_page_links_to_its_provider(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path, None, empty_status_inputs())
    html = (tmp_path / "endpoints" / "m2fa23p.html").read_text()
    assert 'href="../providers/p.html"' in html


def _head(html: str) -> str:
    """The endpoint page's title block -- everything the crumb links can't cover."""
    return html.split('<div class="head">')[1].split("</div>\n    <div")[0]


def test_endpoint_head_links_model_provider_and_org(tmp_path):
    """The h1 names a model, the @ names a provider, the trailing tag names an org:
    each is a page, so each is a link."""
    _scaffold(tmp_path)
    render_site(tmp_path, None, empty_status_inputs())
    head = _head((tmp_path / "endpoints" / "m2fa23p.html").read_text())
    assert f'href="../models/{slugify("m/a")}.html"' in head
    assert 'href="../providers/p.html"' in head
    assert f'href="../orgs/{slugify("m")}.html"' in head


def test_endpoint_and_model_crumbs_link_the_org(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path, None, empty_status_inputs())
    org_href = f'href="../orgs/{slugify("m")}.html"'
    for path in ("endpoints/m2fa23p.html", f"models/{slugify('m/a')}.html"):
        crumb = (tmp_path / path).read_text().split('<div class="crumb">')[1]
        assert org_href in crumb.split("</div>")[0], path


def test_render_emits_org_pages(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path, None, empty_status_inputs())
    page = (tmp_path / "orgs" / f"{slugify('m')}.html").read_text()
    assert f'href="../models/{slugify("m/a")}.html"' in page
    assert "<h1>m</h1>" in page


def test_org_pages_are_rewritten_from_scratch(tmp_path):
    """A model that leaves the fleet must not leave a stale org page behind."""
    _scaffold(tmp_path)
    orgs = tmp_path / "orgs"
    orgs.mkdir()
    (orgs / "gone.html").write_text("stale")
    render_site(tmp_path, None, empty_status_inputs())
    assert not (orgs / "gone.html").exists()


def test_endpoint_with_nothing_to_show_gets_a_status_page_if_ever_tracked(tmp_path):
    """A never-tracked endpoint that already left the catalog stays absent. One we
    tracked (a BI state file) keeps an explained page and directory row -- with a
    status instead of a chart -- but stays out of the tracked-fleet stats."""
    _scaffold(tmp_path)
    # an LT endpoint whose queries all errored (never observed, no catalog entry),
    # and a B3IT one retired before its first post-reference batch
    dead_lt = tmp_path / "data" / "lt" / "m2fa23dead" / "default"
    dead_lt.mkdir(parents=True)
    (dead_lt / "info.json").write_text(
        json.dumps({"prompt": "hi", "endpoint": {"model": "m/a", "provider": "dead"}})
    )
    write_month_dir(dead_lt, "2026-06", [["24 10:00:00", "e0"]])
    write_b3it_state(tmp_path, "m/a", "gone", status="retired")

    render_site(tmp_path, None, empty_status_inputs())

    assert not (tmp_path / "endpoints" / "m2fa23dead.html").exists()
    gone_slug = b3it_slug("m/a", "gone")
    assert (tmp_path / "endpoints" / f"{gone_slug}.html").exists()

    overview = json.loads((tmp_path / "data" / "overview.json").read_text())
    assert {e["slug"] for e in overview["endpoints"]} == {"m2fa23p", gone_slug}
    gone_row = next(e for e in overview["endpoints"] if e["slug"] == gone_slug)
    assert gone_row["methods"] == [] and gone_row["headline"] == "retired"
    assert overview["stats"]["endpoints"] == 1

    model = json.loads(
        (tmp_path / "data" / "models" / f"{slugify('m/a')}.json").read_text()
    )
    assert [e["provider"] for e in model["endpoints"] if e["methods"]] == ["p"]
    assert model["n_endpoints"] == 1 and model["n_providers"] == 1


def _manifest(html: str) -> dict:
    return json.loads(html.split('id="manifest">')[1].split("</script>")[0])


def test_manifest_escapes_hostile_names_and_error_details(tmp_path):
    """Provider names and probe-failure reasons are attacker-influenced; a
    </script> inside them must not break out of the manifest JSON block."""
    _scaffold(tmp_path)
    hostile = "p</script><script>alert(1)</script>"
    inputs = empty_status_inputs()
    inputs.catalog = [catalog_entry("m/evil", hostile)]
    inputs.lt_failures = LTFailureCache(
        failures=[
            LTFailure(
                model="m/evil",
                provider=hostile,
                reason=hostile,
                last_seen=datetime(2026, 7, 1, tzinfo=timezone.utc),
            )
        ]
    )

    render_site(tmp_path, None, inputs)

    page = (tmp_path / "endpoints" / f"{slugify(f'm/evil#{hostile}')}.html").read_text()
    block = page.split('id="manifest">')[1].split("</script>")[0]
    assert "<" not in block
    json.loads(block)
    assert "<script>alert(1)</script>" not in page


def test_render_emits_status_pages_for_catalog_endpoints(tmp_path):
    _scaffold(tmp_path)
    inputs = empty_status_inputs()
    inputs.endpoints_lt = [
        Endpoint(api="openrouter", model="m/a", provider="p", cost=(1, 2))
    ]
    inputs.catalog = [
        catalog_entry("m/a", "p"),
        catalog_entry(
            "openai/gpt-5.4",
            "openai",
            supports_temperature=False,
            supports_logprobs=False,
        ),
    ]
    inputs.bi_cache.add_bad_temperature(
        Endpoint(
            api="openrouter", model="openai/gpt-5.4", provider="openai", cost=(1, 2)
        )
    )

    render_site(tmp_path, None, inputs)

    slug = slugify("openai/gpt-5.4#openai")
    page = (tmp_path / "endpoints" / f"{slug}.html").read_text()
    # the manifest carries only what endpoint.js reads; status + metadata are
    # rendered server-side
    assert _manifest(page) == {"slug": slug}
    assert 'class="badge st st-untrackable"' in page
    assert '<b class="st-name">no logprobs</b>' in page
    assert "$1.00 in · $2.00 out" in page

    tracked_page = (tmp_path / "endpoints" / "m2fa23p.html").read_text()
    assert _manifest(tracked_page) == {"slug": "m2fa23p"}
    assert '<b class="st-name">tracked</b>' in tracked_page

    model = json.loads(
        (tmp_path / "data" / "models" / f"{slugify('openai/gpt-5.4')}.json").read_text()
    )
    assert model["status_summary"] == "0 of 1 endpoint trackable"

    overview = json.loads((tmp_path / "data" / "overview.json").read_text())
    row = next(e for e in overview["endpoints"] if e["slug"] == slug)
    assert row["headline"] == "untrackable" and row["reason"]


def test_untracked_endpoint_page_renders_card_not_chart(tmp_path):
    """An untracked page is the status card + catalog metadata: no chart mount, no
    endpoint.js, and no link to a provider page that was never generated."""
    _scaffold(tmp_path)
    inputs = empty_status_inputs()
    inputs.endpoints_lt = [
        Endpoint(api="openrouter", model="m/a", provider="p", cost=(1, 2))
    ]
    inputs.catalog = [
        catalog_entry("m/a", "p"),
        catalog_entry(
            "openai/gpt-5.4",
            "openai",
            supports_temperature=False,
            supports_logprobs=False,
        ),
    ]
    inputs.bi_cache.add_bad_temperature(
        Endpoint(
            api="openrouter", model="openai/gpt-5.4", provider="openai", cost=(1, 2)
        )
    )

    render_site(tmp_path, None, inputs)

    page = (
        tmp_path / "endpoints" / f"{slugify('openai/gpt-5.4#openai')}.html"
    ).read_text()
    assert "status-methods" in page and "meta-grid" in page
    assert 'id="mainchart"' not in page and "js/endpoint.js" not in page
    assert 'class="badge st st-untrackable"' in page
    # openai has no tracked endpoint, hence no provider page: named, never linked
    assert 'href="../providers/openai.html"' not in page

    tracked = (tmp_path / "endpoints" / "m2fa23p.html").read_text()
    assert "status-methods" in tracked and 'id="mainchart"' in tracked
    assert 'href="../providers/p.html"' in tracked


def test_a_dead_lt_series_does_not_hide_a_live_b3it_one(tmp_path):
    """The endpoint survives on its B3IT series; only the empty lt badge goes."""
    _scaffold(tmp_path)
    slug = b3it_slug("m/a", "half")
    half = tmp_path / "data" / "lt" / slug / "default"
    half.mkdir(parents=True)
    (half / "info.json").write_text(
        json.dumps({"prompt": "hi", "endpoint": {"model": "m/a", "provider": "half"}})
    )
    write_month_dir(half, "2026-06", [["24 10:00:00", "e0"]])
    write_b3it_series(
        tmp_path,
        "m/a",
        "half",
        status="monitoring",
        retired=None,
        month="2026-06",
        tokens=["A"] * 10,
    )

    render_site(tmp_path, None, empty_status_inputs())

    assert (tmp_path / "endpoints" / f"{slug}.html").exists()
    overview = json.loads((tmp_path / "data" / "overview.json").read_text())
    row = next(e for e in overview["endpoints"] if e["slug"] == slug)
    assert row["methods"] == ["b3it"]


def test_spend_rows_only_link_endpoints_that_have_a_page(tmp_path):
    """Spend covers every slug we were ever billed for -- discovery probes and
    endpoints that never produced a series included. Those have no page, so the
    slug is named but not linked."""
    _scaffold(tmp_path)
    for slug in ("m2fa23p", "probe2fonly23q"):
        d = tmp_path / "data" / "spend" / slug
        d.mkdir(parents=True)
        (d / "2026-06.jsonl").write_text(_spend_line("lt", 0.05) + "\n")

    render_site(tmp_path, None, empty_status_inputs())

    html = (tmp_path / "spend.html").read_text()
    assert 'href="endpoints/m2fa23p.html"' in html
    assert "probe2fonly23q" in html
    assert 'href="endpoints/probe2fonly23q.html"' not in html


def test_methodology_links_each_paper_from_its_own_section(tmp_path):
    """The paper belongs beside the method it describes, not only in Read more."""
    _scaffold(tmp_path)
    render_site(tmp_path, None, empty_status_inputs())
    page = (tmp_path / "methodology.html").read_text()
    lt_section, b3it_section, read_more = (
        page.split("Black-box border input tracking")[0],
        page.split("Black-box border input tracking")[1].split("Read more")[0],
        page.split("Read more")[1],
    )
    assert "arxiv.org/abs/2512.03816" in lt_section
    assert "arxiv.org/abs/2602.11083" in b3it_section
    assert "arxiv.org/abs/2512.03816" in read_more
    assert "arxiv.org/abs/2602.11083" in read_more
