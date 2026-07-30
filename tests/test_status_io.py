"""The IO side of statuses: loading the committed catalog and deriving the
site-wide status map (statuses + name/catalog lookups) that render.py stamps
into every page."""

from datetime import datetime, timezone

from conftest import catalog_entry, empty_status_inputs
from trackllm_website.bi.state import EndpointBIState
from trackllm_website.config import Endpoint
from trackllm_website.generate_site.lt import EndpointInfo
from trackllm_website.generate_site.status_io import (
    load_catalog,
    lt_stalled_slugs,
    resolve_site_statuses,
)
from trackllm_website.storage import MonthlyData, ResponseError
from trackllm_website.util import slugify

CATALOG_YAML = """\
endpoints_catalog:
- model: ai21/jamba-large-1.7
  provider: ai21/fp8
  cost:
  - 2.0
  - 8.0
  created: '2025-08-08T16:03:40+00:00'
  supports_temperature: true
  supports_logprobs: false
  free: false
- model: org/free-model
  provider: p
  cost:
  - 0.0
  - 0.0
  created: null
  supports_temperature: null
  supports_logprobs: null
  free: true
"""


def test_load_catalog_parses_the_committed_shape(tmp_path):
    path = tmp_path / "endpoints_catalog.yaml"
    path.write_text(CATALOG_YAML)
    entries = load_catalog(path)
    assert len(entries) == 2
    e = entries[0]
    assert e.slug == slugify("ai21/jamba-large-1.7#ai21/fp8")
    assert e.cost == (2.0, 8.0)
    assert e.created == datetime(2025, 8, 8, 16, 3, 40, tzinfo=timezone.utc)
    assert e.supports_logprobs is False
    free = entries[1]
    assert free.free and free.created is None and free.supports_temperature is None


def test_load_catalog_missing_file_is_empty(tmp_path):
    assert load_catalog(tmp_path / "nope.yaml") == []


def test_resolve_site_statuses_builds_union_lookups():
    inputs = empty_status_inputs()
    inputs.catalog = [catalog_entry("org/a", "p")]
    lt_info = EndpointInfo(
        model="org/b", provider="q", slug=slugify("org/b#q"), prompts=[]
    )
    state = EndpointBIState(
        endpoint=Endpoint(api="openrouter", model="org/c", provider="r", cost=(1, 2)),
        status="monitoring",
        retired=None,
        epochs=[],
    )
    site = resolve_site_statuses(
        inputs, {lt_info.slug: lt_info}, set(), {state.slug: state}
    )

    expected = {slugify("org/a#p"), lt_info.slug, state.slug}
    assert set(site.statuses) == expected
    assert set(site.names) == expected
    assert site.names[lt_info.slug] == ("org/b", "q")
    assert site.names[state.slug] == ("org/c", "r")
    assert set(site.entries) == {slugify("org/a#p")}
    # the historical LT-only endpoint reads as stalled, the BI one as monitoring
    assert site.statuses[lt_info.slug].lt == "stalled"
    assert site.statuses[state.slug].bi == "monitoring"


def _write_error_months(lt_dir, endpoint: Endpoint, n_errors: int):
    d = lt_dir / slugify(f"{endpoint.model}#{endpoint.provider}") / "prompt"
    err = ResponseError(http_code=500, message="boom")
    dt = datetime(2026, 6, 15, tzinfo=timezone.utc)
    MonthlyData(
        year=2026, month=6, logprob_responses=[], error_responses=[(dt, err)] * n_errors
    ).serialize(d / "2026-06")


def test_lt_stalled_slugs_flags_only_all_error_observed_endpoints(tmp_path):
    stalled = Endpoint(api="openrouter", model="org/dead", provider="p", cost=(1, 2))
    thin = Endpoint(api="openrouter", model="org/thin", provider="p", cost=(1, 2))
    unobserved = Endpoint(api="openrouter", model="org/gone", provider="p", cost=(1, 2))
    # is_stalled needs config.api.abandon_after all-error latest queries (100)
    _write_error_months(tmp_path, stalled, 100)
    _write_error_months(tmp_path, thin, 5)
    _write_error_months(tmp_path, unobserved, 100)

    observed = {
        slugify("org/dead#p"),
        slugify("org/thin#p"),
    }
    out = lt_stalled_slugs(tmp_path, [stalled, thin, unobserved], observed)
    assert out == {slugify("org/dead#p")}
