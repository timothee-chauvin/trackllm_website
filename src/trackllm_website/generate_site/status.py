"""Why every endpoint is or isn't tracked, resolved purely from committed files.

`tracked.py` decides who gets charts; this module decides what every page says.
Its universe is the catalog snapshot plus every endpoint we ever tracked (LT
observations or a BI state file), so an endpoint we monitored stays explained
after it leaves the catalog or our selection instead of vanishing. Everything
here is derivation over parsed inputs the caller loads — no file IO, no network,
no re-run of selection (monitoring/retired come from state files; selection
needs the live popularity feed).

Statuses are per-method: LT and BI are independent (grok-4.5 is LT-tracked and
BI-too-expensive). One derived headline summarizes the endpoint for badges and
counts, and `headlines` keeps every headline either method contributes, so the
directory's status chips can show an endpoint under each thing that happened to
it. All user-facing status text lives in STATUS_COPY; templates never invent
wording.
"""

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from trackllm_website.bi.selection import SelectionPolicy, _matches_any
from trackllm_website.bi.state import EndpointBIState
from trackllm_website.bi.vetting import EndpointCache
from trackllm_website.config import Endpoint, config
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
    # The stored reason literal says "delisted", but the lifecycle retires under it
    # whenever a monitored endpoint stays out of the selected set for the grace
    # period -- the selection policy (bi_selection.toml: flagships, popularity,
    # provider coverage, budget) dropping it, not only the catalog losing it.
    "retired:delisted": (
        "Monitoring was retired: the endpoint dropped out of our B3IT selection "
        "(flagships, the most popular models, the cheapest endpoints per provider, "
        f"within budget) for {config.bi.reinit.deselection_grace_days} days."
    ),
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

# api.py raises the first form on a query billed over config.api.max_cost_per_query
# and vetting records the second; the caches store them verbatim, but a guard trip
# is a budget decision, not an error.
_GUARD_TRIP = re.compile(r"\$[\d.]+/query > \$[\d.]+/query guard|^too_expensive: \$")
# A stored detail is a raw error string, often a JSON blob (nested and truncated).
# The reader-facing detail is the innermost message it carries.
_DETAIL_PREFIX = re.compile(r"^(error: |(plain|cached): |openrouter#\S+: )+")
_HTTP_CODE = re.compile(r"^(\d{3}) |\"code\"\s*:\s*(\d{3})")
_LEADING_CODE = re.compile(
    r"^\d{1,3} "
)  # "plain: 400 ...", or "plain: 0 ..." for no HTTP reply
_TRACE_ID = re.compile(r"\s*trace_id:.*$")
_JSON_MESSAGE = re.compile(r"\"(?:message|msg|raw)\"\s*:\s*\"([^\"{][^\"]*)\"")
_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
_GENERIC_MESSAGES = frozenset({"Provider returned error"})
_TIMEOUT = re.compile(r"^Timeout after [\d.]+s$")
_NO_USAGE = re.compile(r"^No usage in response")
DETAIL_MAX_CHARS = 120


def is_guard_trip(detail: str | None) -> bool:
    return bool(detail and _GUARD_TRIP.search(detail))


def humanize_detail(detail: str | None) -> str | None:
    """One readable line from a stored probe error, or None when there is none."""
    if not detail:
        return None
    text = _DETAIL_PREFIX.sub("", detail.strip())
    code_match = _HTTP_CODE.search(text)
    code = next((c for c in code_match.groups() if c), None) if code_match else None
    text = _LEADING_CODE.sub("", text)
    if _TIMEOUT.match(text):
        return "the request timed out"
    if _NO_USAGE.match(text):
        return "the response carried no usage or cost data"
    # unescape the nested JSON strings providers wrap their errors in
    while (flat := text.replace('\\"', '"').replace("\\n", " ")) != text:
        text = flat
    text = _UNICODE_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
    messages = [m for m in _JSON_MESSAGE.findall(text) if m not in _GENERIC_MESSAGES]
    if messages:
        text = messages[-1]
    elif text.startswith("{") or text.startswith('"'):
        text = "the provider returned an error"
    text = " ".join(_TRACE_ID.sub("", text).replace("\\", "").split())
    if len(text) > DETAIL_MAX_CHARS:
        text = text[:DETAIL_MAX_CHARS].rsplit(" ", 1)[0] + "…"
    return f"HTTP {code}: {text}" if code else text


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
    headlines: list[str]  # every headline either method contributes, chain order
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
    """A model-level badge: tracked if any endpoint is, retired if any was, else
    the reason most of its endpoints share (ties broken by the chain's order) --
    one untrackable endpoint must not label a model whose other fifteen are
    merely too expensive."""
    counts = Counter(headlines)
    for h in ("tracked", "retired"):
        if counts[h]:
            return h
    return max(counts, key=lambda h: (counts[h], -HEADLINE_ORDER.index(h)))


def headline_breakdown(headlines: Iterable[str]) -> str:
    """Every headline with its count, most common first: "11 tracked · 7 retired"."""
    counts = Counter(headlines)
    ordered = sorted(
        counts.items(), key=lambda kv: (-kv[1], HEADLINE_ORDER.index(kv[0]))
    )
    return " · ".join(f"{n} {h.replace('_', ' ')}" for h, n in ordered)


def _headlines_of(status: str) -> list[str]:
    """What this one method status alone contributes. Every retirement is
    "retired", and some carry a second finding besides (retired:too_expensive is
    also too expensive, retired:unreachable also errors out); statuses that only
    matter jointly (no_logprobs, bad_temperature) contribute nothing."""
    found = []
    if status == "stalled" or status.startswith("retired:"):
        found.append("retired")
    if status in ("tracked", "monitoring"):
        found.append("tracked")
    elif status in ("too_expensive", "retired:too_expensive"):
        found.append("too_expensive")
    elif status == "unprobeable:batch":
        found.append("untrackable")
    elif status in ("not_selected", "excluded"):
        found.append("not_selected")
    elif status in ERRORS_OUT:
        found.append("errors_out")
    elif status in ("pending", "free_excluded"):
        found.append(status)
    return found


def headlines_for(lt: str, bi: str) -> list[str]:
    """Every headline the endpoint carries, in HEADLINE_ORDER. A tracked endpoint
    is just that: what its other method could not do is not held against it. An
    untracked one carries what each method contributes on its own plus the joint
    headline (untrackable); "pending" is the absence of a verdict, so it only
    stays when nothing else was found."""
    headline = headline_for(lt, bi)
    if headline == "tracked":
        return [headline]
    found = set(_headlines_of(lt)) | set(_headlines_of(bi)) | {headline}
    if len(found) > 1:
        found.discard("pending")
    return [h for h in HEADLINE_ORDER if h in found]


def one_line_reason(st: EndpointStatus) -> str:
    """The fleet row's single line: the copy of every method status behind the
    endpoint's headlines (with its recorded detail), the one that drove the
    dominant headline first; or the headline's own copy when the headline is a
    joint conclusion (untrackable)."""
    methods = [(st.lt, st.lt_detail), (st.bi, st.bi_detail)]
    parts: list[str] = []
    covered: set[str] = set()
    for status, detail in sorted(
        methods, key=lambda sd: st.headline not in _headlines_of(sd[0])
    ):
        # one sentence per headline: tracked+monitoring is not said twice
        new = set(_headlines_of(status)) & set(st.headlines) - covered
        if new:
            covered |= new
            copy = STATUS_COPY[status]
            parts.append(f"{copy.rstrip('.')} ({detail})." if detail else copy)
    return " ".join(parts) or STATUS_COPY[st.headline]


def status_json(st: EndpointStatus) -> dict:
    """The status object every page JSON carries; templates only echo it."""
    return {
        "lt": st.lt,
        "bi": st.bi,
        "headline": st.headline,
        "headlines": st.headlines,
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
        if is_guard_trip(failure_by_slug[slug]):
            return "too_expensive", None
        return "probe_failed", humanize_detail(failure_by_slug[slug])
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
            ("too_expensive", None)
            if is_guard_trip(entry.detail)
            else (f"unprobeable:{entry.reason}", humanize_detail(entry.detail))
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
            headlines=headlines_for(lt, bi),
            lt_detail=lt_detail,
            bi_detail=bi_detail,
        )
    return statuses
