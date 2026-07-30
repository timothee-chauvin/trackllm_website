"""The IO side of statuses: load the committed inputs, derive the site-wide map.

status.py is pure; this module is its caller. load_status_inputs reads the
committed snapshots (catalog, both vetting caches, tracked-endpoint lists,
selection policy), and resolve_site_statuses turns them plus the build's own
observations into the slug -> status map and the name/catalog lookups that
render.py threads into every generator.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from trackllm_website.bi.selection import SelectionPolicy, load_policy
from trackllm_website.bi.state import EndpointBIState
from trackllm_website.bi.vetting import EndpointCache
from trackllm_website.config import Endpoint, config, root
from trackllm_website.generate_site.lt import EndpointInfo
from trackllm_website.generate_site.status import (
    CatalogEntry,
    EndpointStatus,
    resolve_statuses,
)
from trackllm_website.storage import ResultsStorage
from trackllm_website.update_endpoints import (
    ENDPOINTS_CACHE_BI_PATH,
    ENDPOINTS_CACHE_LT_PATH,
    ENDPOINTS_CATALOG_PATH,
    LTFailureCache,
)
from trackllm_website.util import slugify


@dataclass
class StatusInputs:
    """The committed files behind resolve_statuses, parsed."""

    catalog: list[CatalogEntry]
    endpoints_lt: list[Endpoint]
    endpoints_bi: list[Endpoint]
    bi_cache: EndpointCache
    lt_failures: LTFailureCache
    policy: SelectionPolicy
    max_cost_mtok: float


def load_catalog(path: Path) -> list[CatalogEntry]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "endpoints_catalog" not in data:
        # A present-but-misshapen file is a writer regression, not an empty
        # catalog: returning [] here would silently drop every untracked page.
        raise ValueError(f"{path} has no endpoints_catalog mapping")
    return [CatalogEntry(**e) for e in data["endpoints_catalog"] or []]


def load_status_inputs() -> StatusInputs:
    return StatusInputs(
        catalog=load_catalog(ENDPOINTS_CATALOG_PATH),
        endpoints_lt=config.endpoints_lt,
        endpoints_bi=config.endpoints_bi,
        bi_cache=EndpointCache.load(ENDPOINTS_CACHE_BI_PATH),
        lt_failures=LTFailureCache.load(ENDPOINTS_CACHE_LT_PATH),
        policy=load_policy(root / config.bi.selection_path),
        max_cost_mtok=config.api.max_cost_mtok,
    )


def lt_stalled_slugs(
    lt_dir: Path, endpoints_lt: list[Endpoint], observed: set[str]
) -> set[str]:
    """Observed LT endpoints whose latest queries all errored (storage.is_stalled)."""
    storage = ResultsStorage(lt_dir)
    return {
        slug
        for e in endpoints_lt
        if (slug := slugify(f"{e.model}#{e.provider}")) in observed
        and storage.is_stalled(e)
    }


@dataclass
class SiteStatuses:
    """The status map plus the lookups pages stamp alongside it."""

    statuses: dict[str, EndpointStatus]
    entries: dict[str, CatalogEntry]  # slug -> catalog entry, for page metadata
    names: dict[str, tuple[str, str]]  # slug -> (model, provider)


def resolve_site_statuses(
    inputs: StatusInputs,
    lt_by_slug: dict[str, EndpointInfo],
    lt_stalled: set[str],
    bi_states: dict[str, EndpointBIState],
) -> SiteStatuses:
    """lt_by_slug is the observed LT fleet (has an lt_scores series); bi_states is
    every state file, series or not -- together with the catalog they define the
    universe of endpoints the site explains."""
    statuses = resolve_statuses(
        catalog=inputs.catalog,
        endpoints_lt=inputs.endpoints_lt,
        lt_observed=set(lt_by_slug),
        lt_stalled=lt_stalled,
        endpoints_bi=inputs.endpoints_bi,
        bi_cache=inputs.bi_cache,
        bi_states=bi_states,
        policy=inputs.policy,
        lt_failures=inputs.lt_failures,
        max_cost_mtok=inputs.max_cost_mtok,
    )
    entries = {e.slug: e for e in inputs.catalog}
    names = {slug: (e.model, e.provider) for slug, e in entries.items()}
    for slug, state in bi_states.items():
        names.setdefault(slug, (state.endpoint.model, state.endpoint.provider))
    for slug, info in lt_by_slug.items():
        names.setdefault(slug, (info.model, info.provider))
    return SiteStatuses(statuses=statuses, entries=entries, names=names)
