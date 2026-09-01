"""Why every endpoint is or isn't tracked, resolved purely from committed files.

`tracked.py` decides who gets charts; this module decides what every page says.
Its universe is the catalog snapshot plus every endpoint we ever tracked (LT
observations or a BI state file), so a delisted endpoint we monitored stays
explained instead of vanishing. Everything here is derivation over parsed
inputs the caller loads — no file IO, no network, no re-run of selection
(monitoring/retired come from state files; selection needs the live popularity
feed).

Statuses are per-method: LT and BI are independent (grok-4.5 is LT-tracked and
BI-too-expensive), and one derived headline summarizes the endpoint. All
user-facing status text lives in STATUS_COPY; templates never invent wording.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from trackllm_website.bi.selection import SelectionPolicy, _matches_any
from trackllm_website.bi.state import EndpointBIState
from trackllm_website.bi.vetting import EndpointCache
from trackllm_website.config import Endpoint
from trackllm_website.update_endpoints import LTFailureCache
from trackllm_website.util import slugify

STATUS_COPY: dict[str, str] = {
    "tracked": "This endpoint is actively tracked.",
    "stalled": "We tracked this endpoint's logprobs, but its recent queries all failed, so tracking stopped.",
    "probe_failed": "This endpoint claims logprob support, but our probe could not obtain usable logprobs.",
    "no_logprobs": "This endpoint does not return logprobs, so logprob tracking is impossible.",
    "monitoring": "This endpoint is actively monitored through its border inputs.",
    "retired:no_bis": "Monitoring was retired: we could not find enough border inputs for this endpoint.",
    "retired:unreachable": "Monitoring was retired: the endpoint stopped answering our queries.",
    "retired:delisted": "Monitoring was retired: the endpoint left the OpenRouter catalog.",
    "retired:stalled": "Monitoring was retired: the endpoint stopped yielding usable samples.",
    "retired:reinit_timeout": "Monitoring was retired: re-initialization after a detected change repeatedly ran out of time.",
    "retired:too_expensive": "Monitoring was retired: a single query costs more than our per-query guard allows.",
    "retired:budget": "Monitoring was retired: projected monthly spend ran over the budget cap.",
    "bad_temperature": "This API rejects or ignores the temperature parameter, so T=0 sampling is impossible — presumably to prevent distillation.",
    "liar": "This endpoint bills more than its advertised price implies, so we refuse to fund it.",
    "excluded": "Our selection policy explicitly excludes this endpoint.",
    "not_selected": "This endpoint vetted fine, but the selection budget went to more popular models.",
    "too_expensive": "This endpoint costs more than our tracking budget allows.",
    "free_excluded": "Free endpoints are excluded from tracking: their routing and rate limits are too unstable.",
    "pending": "This endpoint has not been evaluated for tracking yet.",
    "retired": "We tracked this endpoint, but tracking has since been retired.",
    "untrackable": "This endpoint claims neither temperature control nor logprobs, so no tracking method can work — presumably to prevent distillation.",
    "errors_out": "Our probes of this endpoint error out.",
    "unprobeable:batch": "Batch endpoints only answer asynchronous batch queries, so our synchronous probes cannot vet them.",
    "unprobeable:flaky": "Our vetting probes of this endpoint persistently fail.",
}

ERRORS_OUT = frozenset(
    {"liar", "probe_failed", "retired:unreachable", "unprobeable:flaky"}
)
# retired:unreachable is deliberately absent: it reads as "errors out", and the
# retired headline would otherwise shadow errors_out entirely.
_RETIRED_HEADLINE = frozenset(
    {
        "retired:no_bis",
        "retired:delisted",
        "retired:stalled",
        "retired:reinit_timeout",
        "retired:budget",
    }
)

HEADLINE_ORDER = [
    "tracked",
    "retired",
    "untrackable",
    "too_expensive",
    "not_selected",
    "errors_out",
    "pending",
    "free_excluded",
]


class CatalogEntry(BaseModel):
    """One endpoints_catalog.yaml entry, as parsed by the caller."""

    model: str
    provider: str
    cost: tuple[float, float]
    created: datetime | None
    supports_temperature: bool | None
    supports_logprobs: bool | None
    free: bool

    @property
    def slug(self) -> str:
        return slugify(f"{self.model}#{self.provider}")

    def as_endpoint(self) -> Endpoint:
        return Endpoint(
            api="openrouter", model=self.model, provider=self.provider, cost=self.cost
        )

    def as_meta(self) -> dict:
        """Catalog metadata an untracked endpoint page shows instead of a chart."""
        return {
            "cost": list(self.cost),
            "created": self.created.isoformat() if self.created else None,
            "supports_temperature": self.supports_temperature,
            "supports_logprobs": self.supports_logprobs,
            "free": self.free,
        }


@dataclass
class EndpointStatus:
    lt: str
    bi: str
    headline: str
    lt_detail: str | None
    bi_detail: str | None


def headline_for(lt: str, bi: str) -> str:
    """The one-word summary of an endpoint, first match wins.

    lt=stalled joins the retired group (it is LT's form of retirement), and
    bi=excluded joins not_selected (both are policy decisions, not failures);
    neither has a headline of its own in the taxonomy.
    """
    if lt == "tracked" or bi == "monitoring":
        return "tracked"
    if lt == "stalled" or bi in _RETIRED_HEADLINE:
        return "retired"
    if lt == "no_logprobs" and bi == "bad_temperature":
        return "untrackable"
    if bi == "unprobeable:batch":  # async-only: blocks LT and BI alike
        return "untrackable"
    if lt == "too_expensive" or bi in ("too_expensive", "retired:too_expensive"):
        return "too_expensive"
    if bi in ("not_selected", "excluded"):
        return "not_selected"
    if lt in ERRORS_OUT or bi in ERRORS_OUT:
        return "errors_out"
    if "pending" in (lt, bi):
        return "pending"
    if "free_excluded" in (lt, bi):
        return "free_excluded"
    return "pending"


def dominant_headline(headlines: Iterable[str]) -> str:
    """The strongest headline in the chain's priority order (a model-level badge)."""
    return min(headlines, key=HEADLINE_ORDER.index)


def _headline_of(status: str) -> str | None:
    """What this one method status alone contributes to the headline; None for
    statuses (no_logprobs, bad_temperature) that only matter jointly."""
    if status in ("tracked", "monitoring"):
        return "tracked"
    if status == "stalled" or status in _RETIRED_HEADLINE:
        return "retired"
    if status in ("too_expensive", "retired:too_expensive"):
        return "too_expensive"
    if status == "unprobeable:batch":
        return "untrackable"
    if status in ("not_selected", "excluded"):
        return "not_selected"
    if status in ERRORS_OUT:
        return "errors_out"
    if status in ("pending", "free_excluded"):
        return status
    return None


def one_line_reason(st: EndpointStatus) -> str:
    """The fleet row's single line: the copy of the method status that drove the
    headline (with its recorded detail), or the headline's own copy when the
    headline is a joint conclusion (untrackable)."""
    for status, detail in ((st.lt, st.lt_detail), (st.bi, st.bi_detail)):
        if _headline_of(status) == st.headline:
            copy = STATUS_COPY[status]
            return f"{copy.rstrip('.')} ({detail})." if detail else copy
    return STATUS_COPY[st.headline]


def status_json(st: EndpointStatus) -> dict:
    """The status object every page JSON carries; templates only echo it."""
    return {
        "lt": st.lt,
        "bi": st.bi,
        "headline": st.headline,
        "ltCopy": STATUS_COPY[st.lt],
        "biCopy": STATUS_COPY[st.bi],
        "ltDetail": st.lt_detail,
        "biDetail": st.bi_detail,
        "reason": one_line_reason(st),
    }


def _slug(model: str, provider: str) -> str:
    return slugify(f"{model}#{provider}")


def _lt_status(
    slug: str,
    entry: CatalogEntry | None,
    lt_slugs: set[str],
    lt_observed: set[str],
    lt_stalled: set[str],
    failure_by_slug: dict[str, str],
    max_cost_mtok: float,
) -> tuple[str, str | None]:
    if slug in lt_observed:
        if slug in lt_slugs and slug not in lt_stalled:
            return "tracked", None
        return "stalled", None
    if slug in failure_by_slug:
        return "probe_failed", failure_by_slug[slug]
    if entry is not None:
        if entry.supports_logprobs is False:
            return "no_logprobs", None
        if sum(entry.cost) >= max_cost_mtok:
            return "too_expensive", None
        if entry.free:
            return "free_excluded", None
    return "pending", None


def _bi_status(
    slug: str,
    entry: CatalogEntry | None,
    endpoint: Endpoint | None,
    state: EndpointBIState | None,
    bucket_by_slug: dict[str, tuple[str, str | None]],
    bi_slugs: set[str],
    policy: SelectionPolicy,
) -> tuple[str, str | None]:
    if state is not None:
        if state.status == "monitoring":
            return "monitoring", None
        return (
            f"retired:{state.retired.reason}",
            f"since {state.retired.since.date().isoformat()}",
        )
    if slug in bucket_by_slug:
        return bucket_by_slug[slug]
    if endpoint is not None and _matches_any(endpoint, policy.exclude):
        return "excluded", None
    if slug in bi_slugs:
        return "not_selected", None
    if entry is not None and entry.free:
        return "free_excluded", None
    return "pending", None


def resolve_statuses(
    catalog: list[CatalogEntry],
    endpoints_lt: list[Endpoint],
    lt_observed: set[str],
    lt_stalled: set[str],
    endpoints_bi: list[Endpoint],
    bi_cache: EndpointCache,
    bi_states: dict[str, EndpointBIState],
    policy: SelectionPolicy,
    lt_failures: LTFailureCache,
    max_cost_mtok: float,
) -> dict[str, EndpointStatus]:
    """Statuses for the union of the catalog and previously-tracked endpoints.

    lt_observed / lt_stalled are slug sets the caller derives from lt_scores
    presence and ResultsStorage.is_stalled; everything else is parsed committed
    files. Endpoints that existed but were never tracked leave with the catalog.
    """
    entry_by_slug = {e.slug: e for e in catalog}
    lt_slugs = {_slug(e.model, e.provider) for e in endpoints_lt}
    bi_slugs = {_slug(e.model, e.provider) for e in endpoints_bi}
    failure_by_slug = {
        _slug(f.model, f.provider): f.reason for f in lt_failures.failures
    }
    # liars processed last so they win, matching EndpointCache.bucket_of
    bucket_by_slug: dict[str, tuple[str, str | None]] = {
        _slug(entry.endpoint.model, entry.endpoint.provider): (
            f"unprobeable:{entry.reason}",
            entry.detail,
        )
        for entry in bi_cache.unprobeable
    }
    bucket_by_slug |= {
        _slug(e.model, e.provider): (bucket, None)
        for bucket, endpoints in (
            ("bad_temperature", bi_cache.bad_temperature),
            ("too_expensive", bi_cache.too_expensive),
            ("liar", bi_cache.liars),
        )
        for e in endpoints
    }

    statuses = {}
    for slug in sorted(entry_by_slug.keys() | lt_observed | bi_states.keys()):
        entry = entry_by_slug.get(slug)
        state = bi_states.get(slug)
        endpoint = (
            entry.as_endpoint()
            if entry is not None
            else (state.endpoint if state is not None else None)
        )
        lt, lt_detail = _lt_status(
            slug,
            entry,
            lt_slugs,
            lt_observed,
            lt_stalled,
            failure_by_slug,
            max_cost_mtok,
        )
        bi, bi_detail = _bi_status(
            slug, entry, endpoint, state, bucket_by_slug, bi_slugs, policy
        )
        statuses[slug] = EndpointStatus(
            lt=lt,
            bi=bi,
            headline=headline_for(lt, bi),
            lt_detail=lt_detail,
            bi_detail=bi_detail,
        )
    return statuses
