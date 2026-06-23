import asyncio
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import aiohttp
import requests
import yaml
from pydantic import BaseModel

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.common import TOO_EXPENSIVE, resolve_strategies
from trackllm_website.bi.popularity import fetch_popular_models_safe
from trackllm_website.bi.reinit import reinit
from trackllm_website.bi.selection import (
    exceeds_ceiling,
    load_policy,
    select_monitoring_targets,
)
from trackllm_website.bi.state import EndpointBIState, RetiredInfo, load_all_states
from trackllm_website.bi.vetting import EndpointCache, should_recheck, vet_endpoint
from trackllm_website.config import Endpoint, config, logger, root
from trackllm_website.spend import Spend, append_entry, track
from trackllm_website.storage import ResultsStorage
from trackllm_website.util import (
    atomic_write_bytes,
    gather_with_concurrency,
    gather_with_concurrency_streaming,
    slugify,
)

ENDPOINTS_CACHE_BI_PATH = root / "endpoints_cache_bi.yaml"


def cache_too_expensive_from_probe(
    failed: dict[str, list[str]], probed: list[Endpoint], cache: EndpointCache
) -> None:
    """Route endpoints that discover_strategy short-circuited on cost (errors led by
    TOO_EXPENSIVE) into the too_expensive cache, so they are not re-probed next run."""
    by_key = {str(e): e for e in probed}
    for key, errors in failed.items():
        if errors and errors[0] == TOO_EXPENSIVE and key in by_key:
            cache.add_too_expensive(by_key[key])


def save_endpoints_bi(endpoints: list[Endpoint]) -> None:
    """Dump BI candidate endpoints (with measured cost_per_request) to endpoints_bi.yaml.

    Sorted by (total advertised cost, api, model, provider). Task 10's cost-preview
    reuses this exact writer.
    """
    sorted_endpoints = sorted(
        endpoints, key=lambda e: (sum(e.cost), e.api, e.model, e.provider)
    )
    output_data = {
        "endpoints_bi": [
            {
                "api": e.api,
                "model": e.model,
                "provider": e.provider,
                "cost": list(e.cost),
                "cost_per_request": e.cost_per_request,
                "created": e.created.isoformat() if e.created else None,
            }
            for e in sorted_endpoints
        ]
    }
    atomic_write_bytes(
        config.endpoints_yaml_path_bi,
        yaml.dump(output_data, default_flow_style=False, sort_keys=False).encode(),
    )
    logger.info(
        f"Updated {config.endpoints_yaml_path_bi} with {len(sorted_endpoints)} endpoints"
    )


def merge_goods(
    prior_goods: list[Endpoint],
    freshly_good: list[Endpoint],
    cache: EndpointCache,
) -> list[Endpoint]:
    """Merge this run's good endpoints with prior goods that flaked transiently.

    A prior good that didn't vet as a candidate this run AND isn't in a reject
    bucket either has flaked transiently (network / 5xx / un-priceable). We carry
    it forward with its PRIOR cost_per_request so a flaky API day doesn't shrink
    the monitored set. Endpoints explicitly moved to liar / too_expensive are
    excluded (they live in the cache now). Fresh measurements win on overlap.
    """
    fresh_set = set(freshly_good)
    carried = [
        e
        for e in prior_goods
        if e not in fresh_set
        and not cache.is_cached(e)
        and e.cost_per_request is not None
    ]
    return freshly_good + carried


async def get_model_endpoints(
    session,
    model_id,
    created: datetime | None,
    supports_temperature: bool,
    logprob_filter: bool = False,
) -> list[Endpoint]:
    """Fetch endpoints for a model.

    Args:
        session: aiohttp session
        model_id: OpenRouter model ID
        created: model release date from the /models list, stamped onto each Endpoint
        supports_temperature: whether the model's /models supported_parameters lists
            temperature; stamped onto each Endpoint for the vetting temp pre-filter
        logprob_filter: If True, only return endpoints that claim to support logprobs
    """
    url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
    try:
        async with session.get(url) as response:
            data = await response.json()
            model_endpoints = data["data"]["endpoints"]
            filtered_endpoints = []
            for endpoint in model_endpoints:
                if logprob_filter and not (
                    "logprobs" in endpoint["supported_parameters"]
                    and "top_logprobs" in endpoint["supported_parameters"]
                ):
                    continue
                endpoint_data = Endpoint(
                    api="openrouter",
                    model=model_id,
                    provider=endpoint["tag"],
                    cost=(
                        float(
                            (
                                Decimal(endpoint["pricing"]["prompt"]) * 1_000_000
                            ).normalize()
                        ),
                        float(
                            (
                                Decimal(endpoint["pricing"]["completion"]) * 1_000_000
                            ).normalize()
                        ),
                    ),
                    created=created,
                    supports_temperature=supports_temperature,
                )
                filtered_endpoints.append(endpoint_data)
            return filtered_endpoints
    except Exception as e:
        logger.error(f"Error fetching endpoints for {model_id}: {e}")
        return []


async def get_endpoints(
    logprob_filter: bool, max_cost_mtok: float | None
) -> list[Endpoint]:
    """Get all endpoints for all models.

    Args:
        logprob_filter: If True, only return endpoints that claim to support logprobs
        max_cost_mtok: If not None, only keep endpoints whose combined input+output
            cost is below this cap (in $/Mtok)
    """
    response = requests.get("https://openrouter.ai/api/v1/models")
    models = response.json()["data"]
    model_ids = [model["id"] for model in models]
    created_by_id = {
        model["id"]: datetime.fromtimestamp(model["created"], tz=timezone.utc)
        for model in models
        if model.get("created") is not None
    }
    supports_temperature_by_id = {
        model["id"]: "temperature" in (model.get("supported_parameters") or [])
        for model in models
    }
    all_endpoints = []
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_model_endpoints(
                session,
                model_id,
                created_by_id.get(model_id),
                supports_temperature_by_id.get(model_id, True),
                logprob_filter=logprob_filter,
            )
            for model_id in model_ids
        ]
        async for result in gather_with_concurrency_streaming(
            config.api.max_workers, *tasks
        ):
            all_endpoints.extend(result)

    filtered_endpoints = all_endpoints
    if max_cost_mtok is not None:
        filtered_endpoints = [
            e for e in filtered_endpoints if e.cost[0] + e.cost[1] < max_cost_mtok
        ]

    if config.api.openrouter_avoid_free_endpoints:
        filtered_endpoints = [
            e for e in filtered_endpoints if e.cost[0] + e.cost[1] > 0
        ]

    log_msg = (
        f"Found {len(all_endpoints)} {'endpoints claiming logprobs support' if logprob_filter else 'total endpoints'}, "
        f"keeping {len(filtered_endpoints)}"
    )
    if max_cost_mtok is not None:
        log_msg += f" within max cost of ${max_cost_mtok}/Mtok"
    if config.api.openrouter_avoid_free_endpoints:
        log_msg += " and excluding free endpoints"
    logger.info(log_msg)

    return filtered_endpoints


async def test_endpoint_logprobs(endpoint: Endpoint) -> Endpoint | None:
    """Test if an endpoint actually returns logprobs when queried with 'x'"""
    async with OpenRouterClient() as client:
        try:
            response = await client.query(endpoint, "x")
            if response.error or len(
                response.logprobs.logprobs
            ) != endpoint.get_max_logprobs(config):
                log_msg = f"{endpoint} logprob support: ❌"
                if response.error:
                    log_msg += f"\n{response.error}"
                else:
                    log_msg += f"\n{len(response.logprobs.logprobs)} logprobs, expected {endpoint.get_max_logprobs(config)}"
                logger.info(log_msg)
                return None
            logger.info(f"{endpoint} logprob support: ✅")
            return endpoint
        except Exception:
            logger.exception(f"Error testing logprobs for {endpoint}")
            return None


async def test_endpoints_logprobs(endpoints: Iterable[Endpoint]) -> list[Endpoint]:
    """Return the endpoints that actually return logprobs when queried with 'x'"""
    tasks = [test_endpoint_logprobs(e) for e in endpoints]
    return [
        e for e in await gather_with_concurrency(config.api.max_workers, *tasks) if e
    ]


async def update_endpoints_lt():
    """Update the LT endpoints by removing stalled endpoints and adding new ones."""
    storage = ResultsStorage(config.lt_dir)
    current_endpoints = config.endpoints_lt
    endpoints_to_keep = set([e for e in current_endpoints if not storage.is_stalled(e)])
    logger.info(
        f"Keeping {len(endpoints_to_keep)}/{len(current_endpoints)} non-stalled endpoints"
    )
    endpoints_claiming_logprobs = await get_endpoints(
        logprob_filter=True, max_cost_mtok=config.api.max_cost_mtok
    )

    # Update costs with latest values
    for endpoint in endpoints_to_keep:
        if endpoint in endpoints_claiming_logprobs:
            updated_endpoint = endpoints_claiming_logprobs[
                endpoints_claiming_logprobs.index(endpoint)
            ]
            if updated_endpoint.cost != endpoint.cost:
                logger.info(
                    f"Updating cost for {endpoint}: {endpoint.cost} -> {updated_endpoint.cost}"
                )
                endpoint.cost = updated_endpoint.cost

    endpoints_to_test = set(endpoints_claiming_logprobs) - endpoints_to_keep
    valid_endpoints = await test_endpoints_logprobs(endpoints_to_test)
    logger.info(f"Found {len(valid_endpoints)} valid new endpoints")
    final_endpoints = endpoints_to_keep | set(valid_endpoints)
    # Sort endpoints by total cost, then api, model and provider
    sorted_endpoints = sorted(
        final_endpoints, key=lambda e: (sum(e.cost), e.api, e.model, e.provider)
    )

    output_data = {"endpoints_lt": []}
    for e in sorted_endpoints:
        e_data = {
            "api": e.api,
            "model": e.model,
            "provider": e.provider,
            "cost": list(e.cost),
        }
        output_data["endpoints_lt"].append(e_data)

    with open(config.endpoints_yaml_path_lt, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Updated {config.endpoints_yaml_path_lt}")


def partition_temperature(
    endpoints: list[Endpoint],
) -> tuple[list[Endpoint], list[Endpoint]]:
    """(to_probe, to_skip): skip endpoints that explicitly declare no temperature
    support (supports_temperature is False); probe unknown (None) and True."""
    probe = [e for e in endpoints if e.supports_temperature is not False]
    skip = [e for e in endpoints if e.supports_temperature is False]
    return probe, skip


async def update_endpoints_bi() -> list[Endpoint]:
    """Refresh the BI candidate catalog via cost-based vetting.

    Pulls the full catalog (no cost cap), skips endpoints already in the bucketed
    cache (liars / too_expensive / bad_temperature), then vets the rest — known-good
    endpoints included, to refresh their measured cost_per_request since prices move.
    Each vet runs one real query and measures the billed cost; results are routed:
    candidate (kept, with measured cost; or cached too_expensive if over the ceiling),
    liar (cached), transient (skipped, retried next run). Writes endpoints_bi.yaml with
    measured cost_per_request and saves the cache.
    """
    prior_goods = config.endpoints_bi  # prior registry, carries prior cost_per_request
    all_endpoints = await get_endpoints(logprob_filter=False, max_cost_mtok=None)
    policy = load_policy(root / config.bi.selection_path)
    cache = EndpointCache.load(ENDPOINTS_CACHE_BI_PATH)
    now = datetime.now(tz=timezone.utc).replace(microsecond=0)

    # Periodically re-vet too_expensive / bad_temperature rejects: prices drop and
    # providers fix temperature, so clear those buckets to re-probe them this run.
    if should_recheck(cache, now, config.bi.reinit.recheck_days):
        n_cleared = len(cache.too_expensive) + len(cache.bad_temperature)
        cache.too_expensive = []
        cache.bad_temperature = []
        cache.last_recheck = now
        logger.info(f"Recheck due: cleared {n_cleared} too_expensive/bad_temperature")

    # Re-vet known-good and new endpoints alike (to refresh cost_per_request); skip
    # only those already in a reject bucket.
    to_vet = [e for e in all_endpoints if not cache.is_cached(e)]

    # Pre-filter models whose /models supported_parameters omits temperature: they
    # provably ignore T=0, so route them straight to bad_temperature without probing.
    to_vet, temp_skip = partition_temperature(to_vet)
    for endpoint in temp_skip:
        cache.add_bad_temperature(endpoint)
    logger.info(
        f"Skipping {len(temp_skip)} temperature-unsupported endpoints (temp:NO)"
    )

    logger.info(
        f"Vetting {len(to_vet)} of {len(all_endpoints)} endpoints "
        f"(skipping {len(all_endpoints) - len(to_vet)} cached rejects)"
    )

    probe_spend: dict[str, Spend] = {}
    async with OpenRouterClient(timeout=60.0) as probe_client:
        strategies, failed = await resolve_strategies(
            probe_client, to_vet, policy=policy, probe_spend=probe_spend
        )
    cache_too_expensive_from_probe(failed, to_vet, cache)
    logger.info(
        f"Resolved strategies for {len(strategies)} endpoints "
        f"({len(failed)} failed probing)"
    )

    async def vet_one(client: OpenRouterClient, endpoint: Endpoint) -> Endpoint | None:
        """Vet one endpoint; route to the cache or return it (kept good)."""
        strategy = strategies.get(str(endpoint))
        if strategy is None:
            return None  # couldn't resolve a strategy; skip (not cached)
        with track() as spend:
            res = await vet_endpoint(client, endpoint, strategy)
        spend.merge(probe_spend.get(str(endpoint), Spend()))
        append_entry(
            config.spend_dir,
            slugify(f"{endpoint.model}#{endpoint.provider}"),
            "vetting",
            spend,
            now,
        )
        if res.bucket == "candidate":
            endpoint.cost_per_request = res.cost_per_request
            if exceeds_ceiling(
                res.cost_per_request, endpoint.model, endpoint.provider, policy
            ):
                cache.add_too_expensive(endpoint)
                return None
            return endpoint
        if res.bucket == "liar":
            cache.add_liar(endpoint)
        return None  # liar (cached above) or transient (retry next run)

    client = OpenRouterClient()
    try:
        results = await gather_with_concurrency(
            config.api.max_workers,
            *(vet_one(client, e) for e in to_vet),
        )
    finally:
        await client.close()

    freshly_good = [e for e in results if e is not None]
    good = merge_goods(prior_goods, freshly_good, cache)
    logger.info(
        f"Vetting results: {len(freshly_good)} freshly good, "
        f"{len(good) - len(freshly_good)} carried (transient flakes), "
        f"{len(cache.liars)} liars, {len(cache.too_expensive)} too expensive"
    )

    save_endpoints_bi(good)
    cache.save(ENDPOINTS_CACHE_BI_PATH)
    return good


class LifecycleActions(BaseModel):
    onboard: list[Endpoint]
    recheck: list[EndpointBIState]
    delist: list[EndpointBIState]


def should_delist(state: EndpointBIState, now: datetime, grace_days: int) -> bool:
    """A monitoring endpoint that's no longer selected delists only after the grace
    period since it dropped out. Caller sets deselected_since when it leaves the set."""
    if state.deselected_since is None:
        return False
    return now - state.deselected_since >= timedelta(days=grace_days)


def select_lifecycle_actions(
    candidates: list[Endpoint],
    states: dict[str, EndpointBIState],
    now: datetime,
) -> LifecycleActions:
    """Pure selection of lifecycle actions; the caller performs them."""
    r = config.bi.reinit
    known = {s.slug for s in states.values()}
    candidate_set = set(candidates)

    onboard = [e for e in candidates if slugify(f"{e.model}#{e.provider}") not in known]
    # Never re-onboard endpoints retired for no_bis: they are alive but yield
    # too few border inputs, so a recheck just re-runs the full (~15k-query)
    # onboarding and fails again — pure waste. stalled endpoints recheck cheaply
    # (a dead endpoint fails strategy resolution and is skipped) and delisted
    # ones are already excluded by the candidate_set test until they return.
    recheck = [
        s
        for s in states.values()
        if s.status == "retired"
        and s.retired.reason != "no_bis"
        and s.endpoint in candidate_set
        and now - s.retired.last_recheck >= timedelta(days=r.recheck_days)
    ]
    # Deselected monitoring endpoints are only delisted once past the grace period;
    # the executor maintains deselected_since on the states beforehand.
    delist = [
        s
        for s in states.values()
        if s.status == "monitoring"
        and s.endpoint not in candidate_set
        and should_delist(s, now, r.deselection_grace_days)
    ]
    return LifecycleActions(onboard=onboard, recheck=recheck, delist=delist)


async def update_endpoints_bi_lifecycle(candidates: list[Endpoint]):
    """Onboard new candidates, delist vanished endpoints, re-check retired ones.

    Runs the budget policy over the vetted candidates first, so only the selected
    subset is monitored.
    """
    policy = load_policy(root / config.bi.selection_path)
    popular_models = fetch_popular_models_safe(config.bi.popularity.top_n)
    selected, _breakdown = select_monitoring_targets(candidates, policy, popular_models)
    logger.info(
        f"Selection: monitoring {len(selected)} of {len(candidates)} candidates"
    )

    cache = EndpointCache.load(ENDPOINTS_CACHE_BI_PATH)
    states = load_all_states(config.bi.state_dir)
    now = datetime.now(tz=timezone.utc).replace(microsecond=0)

    # Maintain the deselection clock before deciding actions: stamp the moment a
    # monitored endpoint leaves the selected set, and clear it if it returns.
    selected_set = set(selected)
    for state in states.values():
        if state.status != "monitoring":
            continue
        in_set = state.endpoint in selected_set
        if not in_set and state.deselected_since is None:
            state.deselected_since = now
            state.save(config.bi.state_dir)
        elif in_set and state.deselected_since is not None:
            state.deselected_since = None
            state.save(config.bi.state_dir)

    actions = select_lifecycle_actions(selected, states, now)
    logger.info(
        f"BI lifecycle: {len(actions.onboard)} to onboard, "
        f"{len(actions.recheck)} to re-check, {len(actions.delist)} to delist"
    )

    for state in actions.delist:
        epoch = state.current_epoch
        if epoch is not None:
            epoch.end = now
            epoch.end_reason = "gap"
        state.status = "retired"
        state.retired = RetiredInfo(reason="delisted", since=now, last_recheck=now)
        state.save(config.bi.state_dir)

    onboards = [(e, False) for e in actions.onboard]
    rechecks = [(s.endpoint, True) for s in actions.recheck]
    to_init = onboards + rechecks
    if not to_init:
        return

    probe_spend: dict[str, Spend] = {}
    async with OpenRouterClient(timeout=60.0) as probe_client:
        strategies, _ = await resolve_strategies(
            probe_client,
            [e for e, _ in to_init],
            policy=policy,
            probe_spend=probe_spend,
        )

    async def onboard_one(
        client: OpenRouterClient, endpoint: Endpoint, is_recheck: bool
    ) -> None:
        """Onboard or re-check one endpoint; failures are logged and swallowed
        so one bad endpoint never aborts the rest of the batch."""
        slug = slugify(f"{endpoint.model}#{endpoint.provider}")
        kind = "recheck" if is_recheck else "onboard"
        with track() as spend:
            try:
                state = states.get(slug)

                if str(endpoint) not in strategies:
                    # Bump last_recheck so a permanently-hidden-reasoning endpoint isn't
                    # re-probed daily. Fresh onboards have no state yet: skip silently.
                    if is_recheck and state is not None:
                        state.retired.last_recheck = now
                        state.save(config.bi.state_dir)
                    return

                # old_bis=[]: rechecks intentionally rediscover BIs from scratch,
                # since references from before the monitoring gap are stale.
                old_bis = []
                result = await asyncio.wait_for(
                    reinit(client, strategies[str(endpoint)], endpoint, old_bis, now),
                    timeout=config.bi.reinit.onboard_timeout_seconds,
                )
                if result.reason == "bad_temperature":
                    # T=0 is a no-op for this endpoint: cache it so it's excluded and
                    # not re-onboarded every run (rechecked on the cache schedule).
                    cache.add_bad_temperature(endpoint)
                    logger.warning(f"{endpoint}: cached bad_temperature (T=0 ignored)")
                    return
                if state is None:
                    state = EndpointBIState(
                        endpoint=endpoint, status="monitoring", epochs=[]
                    )
                if result.epoch is None:
                    state.status = "retired"
                    state.retired = RetiredInfo(
                        reason="no_bis", since=now, last_recheck=now
                    )
                else:
                    state.status = "monitoring"
                    state.retired = None
                    state.epochs.append(result.epoch)
                state.save(config.bi.state_dir)
            except asyncio.TimeoutError:
                hours = config.bi.reinit.onboard_timeout_seconds / 3600
                logger.warning(
                    f"{endpoint} onboarding exceeded {hours:.0f}h, will resume next run"
                )
            except Exception:
                logger.exception(f"BI onboarding failed for {endpoint}")
            finally:
                spend.merge(probe_spend.get(str(endpoint), Spend()))
                append_entry(config.spend_dir, slug, kind, spend, now)

    client = OpenRouterClient()
    try:
        await gather_with_concurrency(
            config.bi.reinit.onboard_concurrency,
            *(onboard_one(client, e, is_recheck) for e, is_recheck in to_init),
        )
    finally:
        await client.close()
    cache.save(ENDPOINTS_CACHE_BI_PATH)


async def main():
    await update_endpoints_lt()
    good = await update_endpoints_bi()
    await update_endpoints_bi_lifecycle(good)


if __name__ == "__main__":
    asyncio.run(main())
