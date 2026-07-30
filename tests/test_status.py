"""The status resolver: every catalog / previously-tracked endpoint gets a
per-method (lt, bi) status plus one headline, derived purely from committed
snapshots."""

from datetime import datetime, timezone

import pytest

from trackllm_website.bi.selection import SelectionPolicy
from trackllm_website.bi.state import EndpointBIState, RetiredInfo
from trackllm_website.bi.vetting import EndpointCache
from trackllm_website.config import Endpoint
from trackllm_website.generate_site.status import (
    STATUS_COPY,
    CatalogEntry,
    EndpointStatus,
    dominant_headline,
    headline_for,
    one_line_reason,
    resolve_statuses,
    status_json,
)
from trackllm_website.update_endpoints import LTFailure, LTFailureCache
from trackllm_website.util import slugify

NOW = datetime(2026, 7, 30, tzinfo=timezone.utc)
MAX_COST_MTOK = 30.0
POLICY = SelectionPolicy(
    budget_per_month=1.0, max_endpoint_cost=0.5, exclude=["*image*"], rules=[]
)


def slug(model: str, provider: str) -> str:
    return slugify(f"{model}#{provider}")


def entry(model: str, provider: str, **overrides) -> CatalogEntry:
    base = dict(
        model=model,
        provider=provider,
        cost=(1.0, 2.0),
        created=NOW,
        supports_temperature=True,
        supports_logprobs=True,
        free=False,
    )
    return CatalogEntry(**{**base, **overrides})


def ep(model: str, provider: str) -> Endpoint:
    return Endpoint(api="openrouter", model=model, provider=provider, cost=(1, 2))


def bi_state(model: str, provider: str, reason: str | None) -> EndpointBIState:
    retired = (
        None
        if reason is None
        else RetiredInfo(reason=reason, since=NOW, last_recheck=NOW)
    )
    return EndpointBIState(
        endpoint=ep(model, provider),
        status="monitoring" if reason is None else "retired",
        retired=retired,
        epochs=[],
    )


def empty_cache() -> EndpointCache:
    return EndpointCache(liars=[], too_expensive=[], bad_temperature=[])


def resolve(**overrides):
    base = dict(
        catalog=[],
        endpoints_lt=[],
        lt_observed=set(),
        lt_stalled=set(),
        endpoints_bi=[],
        bi_cache=empty_cache(),
        bi_states={},
        policy=POLICY,
        lt_failures=LTFailureCache(failures=[]),
        max_cost_mtok=MAX_COST_MTOK,
    )
    return resolve_statuses(**{**base, **overrides})


S = slug("org/m", "p")


class TestLTStatus:
    def test_tracked(self):
        statuses = resolve(
            catalog=[entry("org/m", "p")],
            endpoints_lt=[ep("org/m", "p")],
            lt_observed={S},
        )
        assert statuses[S].lt == "tracked"
        assert statuses[S].headline == "tracked"

    def test_stalled_flag(self):
        statuses = resolve(
            catalog=[entry("org/m", "p")],
            endpoints_lt=[ep("org/m", "p")],
            lt_observed={S},
            lt_stalled={S},
        )
        assert statuses[S].lt == "stalled"

    def test_stalled_when_dropped_from_list(self):
        statuses = resolve(catalog=[entry("org/m", "p")], lt_observed={S})
        assert statuses[S].lt == "stalled"

    def test_probe_failed_with_reason_detail(self):
        failures = LTFailureCache(
            failures=[
                LTFailure(
                    model="org/m",
                    provider="p",
                    reason="returned 5 logprobs, expected 20",
                    last_seen=NOW,
                )
            ]
        )
        statuses = resolve(catalog=[entry("org/m", "p")], lt_failures=failures)
        assert statuses[S].lt == "probe_failed"
        assert statuses[S].lt_detail == "returned 5 logprobs, expected 20"

    def test_no_logprobs(self):
        statuses = resolve(catalog=[entry("org/m", "p", supports_logprobs=False)])
        assert statuses[S].lt == "no_logprobs"

    def test_unknown_logprobs_claim_is_pending(self):
        statuses = resolve(catalog=[entry("org/m", "p", supports_logprobs=None)])
        assert statuses[S].lt == "pending"

    def test_too_expensive_at_threshold(self):
        statuses = resolve(catalog=[entry("org/m", "p", cost=(10.0, 20.0))])
        assert statuses[S].lt == "too_expensive"

    def test_free_excluded(self):
        statuses = resolve(catalog=[entry("org/m", "p", cost=(0.0, 0.0), free=True)])
        assert statuses[S].lt == "free_excluded"

    def test_pending(self):
        statuses = resolve(catalog=[entry("org/m", "p")])
        assert statuses[S].lt == "pending"


class TestBIStatus:
    def test_monitoring(self):
        statuses = resolve(
            catalog=[entry("org/m", "p")],
            bi_states={S: bi_state("org/m", "p", None)},
        )
        assert statuses[S].bi == "monitoring"
        assert statuses[S].headline == "tracked"

    def test_retired_with_since_detail(self):
        statuses = resolve(
            catalog=[entry("org/m", "p")],
            bi_states={S: bi_state("org/m", "p", "delisted")},
        )
        assert statuses[S].bi == "retired:delisted"
        assert statuses[S].bi_detail == "since 2026-07-30"

    def test_state_wins_over_cache_bucket(self):
        cache = empty_cache()
        cache.add_liar(ep("org/m", "p"))
        statuses = resolve(
            catalog=[entry("org/m", "p")],
            bi_cache=cache,
            bi_states={S: bi_state("org/m", "p", None)},
        )
        assert statuses[S].bi == "monitoring"

    @pytest.mark.parametrize(
        "bucket,expected",
        [
            ("add_liar", "liar"),
            ("add_too_expensive", "too_expensive"),
            ("add_bad_temperature", "bad_temperature"),
        ],
    )
    def test_cache_buckets(self, bucket, expected):
        cache = empty_cache()
        getattr(cache, bucket)(ep("org/m", "p"))
        statuses = resolve(catalog=[entry("org/m", "p")], bi_cache=cache)
        assert statuses[S].bi == expected

    def test_excluded_by_policy_glob(self):
        e = entry("org/image-gen", "p")
        statuses = resolve(catalog=[e])
        assert statuses[e.slug].bi == "excluded"

    def test_not_selected(self):
        statuses = resolve(
            catalog=[entry("org/m", "p")], endpoints_bi=[ep("org/m", "p")]
        )
        assert statuses[S].bi == "not_selected"

    def test_free_excluded(self):
        statuses = resolve(catalog=[entry("org/m", "p", cost=(0.0, 0.0), free=True)])
        assert statuses[S].bi == "free_excluded"

    def test_pending(self):
        statuses = resolve(catalog=[entry("org/m", "p")])
        assert statuses[S].bi == "pending"


class TestHeadline:
    @pytest.mark.parametrize(
        "lt,bi,expected",
        [
            # grok case: LT-tracked wins over BI-too-expensive
            ("tracked", "too_expensive", "tracked"),
            ("pending", "monitoring", "tracked"),
            ("tracked", "retired:no_bis", "tracked"),
            ("stalled", "pending", "retired"),
            ("pending", "retired:delisted", "retired"),
            ("pending", "retired:no_bis", "retired"),
            ("pending", "retired:stalled", "retired"),
            # retired outranks errors_out only for non-unreachable reasons
            ("stalled", "retired:unreachable", "retired"),
            ("pending", "retired:unreachable", "errors_out"),
            ("no_logprobs", "bad_temperature", "untrackable"),
            ("no_logprobs", "too_expensive", "too_expensive"),
            ("too_expensive", "pending", "too_expensive"),
            ("pending", "not_selected", "not_selected"),
            ("probe_failed", "pending", "errors_out"),
            ("pending", "liar", "errors_out"),
            ("no_logprobs", "pending", "pending"),
            ("pending", "bad_temperature", "pending"),
            ("free_excluded", "free_excluded", "free_excluded"),
            ("no_logprobs", "free_excluded", "free_excluded"),
            # policy-excluded reads as a selection decision, not an error
            ("no_logprobs", "excluded", "not_selected"),
        ],
    )
    def test_priority_chain(self, lt, bi, expected):
        assert headline_for(lt, bi) == expected


class TestResolveUnion:
    def test_historical_bi_only_endpoint_kept(self):
        # tracked once, gone from the catalog: still on the site, as retired
        statuses = resolve(bi_states={S: bi_state("org/m", "p", "delisted")})
        assert statuses[S].bi == "retired:delisted"
        assert statuses[S].lt == "pending"
        assert statuses[S].headline == "retired"

    def test_historical_lt_only_endpoint_kept(self):
        statuses = resolve(lt_observed={S})
        assert statuses[S].lt == "stalled"
        assert statuses[S].bi == "pending"
        assert statuses[S].headline == "retired"

    def test_never_tracked_delisted_endpoint_absent(self):
        statuses = resolve(catalog=[entry("org/m", "p")])
        assert slug("org/other", "p") not in statuses

    def test_untrackable_gpt5_like(self):
        cache = empty_cache()
        cache.add_bad_temperature(ep("openai/gpt-5.4", "openai"))
        statuses = resolve(
            catalog=[
                entry(
                    "openai/gpt-5.4",
                    "openai",
                    supports_temperature=False,
                    supports_logprobs=False,
                )
            ],
            bi_cache=cache,
        )
        s = statuses[slug("openai/gpt-5.4", "openai")]
        assert (s.lt, s.bi, s.headline) == (
            "no_logprobs",
            "bad_temperature",
            "untrackable",
        )

    def test_free_endpoint(self):
        e = entry("org/m:free", "p", cost=(0.0, 0.0), free=True)
        statuses = resolve(catalog=[e])
        s = statuses[e.slug]
        assert (s.lt, s.bi, s.headline) == (
            "free_excluded",
            "free_excluded",
            "free_excluded",
        )


class TestReasonAndJson:
    def test_reason_is_the_driving_methods_copy_with_detail(self):
        st = EndpointStatus(
            lt="probe_failed",
            bi="pending",
            headline="errors_out",
            lt_detail="error: 404",
            bi_detail=None,
        )
        assert one_line_reason(st) == (
            "This endpoint claims logprob support, but our probe could not "
            "obtain usable logprobs (error: 404)."
        )

    def test_reason_retired_carries_the_since_detail(self):
        st = EndpointStatus(
            lt="pending",
            bi="retired:delisted",
            headline="retired",
            lt_detail=None,
            bi_detail="since 2026-07-30",
        )
        assert one_line_reason(st) == (
            "Monitoring was retired: the endpoint left the OpenRouter catalog "
            "(since 2026-07-30)."
        )

    def test_reason_falls_back_to_headline_copy_for_joint_conclusions(self):
        st = EndpointStatus(
            lt="no_logprobs",
            bi="bad_temperature",
            headline="untrackable",
            lt_detail=None,
            bi_detail=None,
        )
        assert one_line_reason(st) == STATUS_COPY["untrackable"]

    def test_status_json_shape(self):
        st = EndpointStatus(
            lt="tracked",
            bi="monitoring",
            headline="tracked",
            lt_detail=None,
            bi_detail=None,
        )
        assert status_json(st) == {
            "lt": "tracked",
            "bi": "monitoring",
            "headline": "tracked",
            "ltCopy": STATUS_COPY["tracked"],
            "biCopy": STATUS_COPY["monitoring"],
            "ltDetail": None,
            "biDetail": None,
            "reason": STATUS_COPY["tracked"],
        }

    def test_dominant_headline_follows_priority(self):
        assert dominant_headline(["pending", "untrackable"]) == "untrackable"
        assert dominant_headline(["untrackable", "tracked"]) == "tracked"
        assert dominant_headline(["free_excluded"]) == "free_excluded"

    def test_catalog_entry_as_meta(self):
        e = entry("org/m", "p", cost=(2.0, 8.0), supports_logprobs=False)
        assert e.as_meta() == {
            "cost": [2.0, 8.0],
            "created": NOW.isoformat(),
            "supports_temperature": True,
            "supports_logprobs": False,
            "free": False,
        }


class TestCopy:
    def test_every_status_has_copy(self):
        lt_statuses = {
            "tracked",
            "stalled",
            "probe_failed",
            "no_logprobs",
            "too_expensive",
            "free_excluded",
            "pending",
        }
        bi_statuses = {
            "monitoring",
            "retired:no_bis",
            "retired:unreachable",
            "retired:delisted",
            "retired:stalled",
            "bad_temperature",
            "too_expensive",
            "liar",
            "excluded",
            "not_selected",
            "free_excluded",
            "pending",
        }
        headlines = {
            "tracked",
            "retired",
            "untrackable",
            "too_expensive",
            "not_selected",
            "errors_out",
            "pending",
            "free_excluded",
        }
        assert lt_statuses | bi_statuses | headlines <= STATUS_COPY.keys()

    def test_copy_is_one_sentence_each(self):
        for status, sentence in STATUS_COPY.items():
            assert sentence.strip(), status

    def test_bad_temperature_wording(self):
        sentence = STATUS_COPY["bad_temperature"]
        assert "temperature" in sentence
        assert "T=0" in sentence
        assert "distillation" in sentence
