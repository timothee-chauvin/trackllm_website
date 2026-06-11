import asyncio
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Literal

import aiohttp
import requests
import yaml
from pydantic import BaseModel

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.common import resolve_strategies
from trackllm_website.bi.reinit import reinit
from trackllm_website.bi.state import EndpointBIState, RetiredInfo, load_all_states
from trackllm_website.config import Endpoint, config, logger, root
from trackllm_website.storage import ResultsStorage
from trackllm_website.util import (
    gather_with_concurrency,
    gather_with_concurrency_streaming,
    slugify,
)

BAD_ENDPOINTS_BI_PATH = root / "bad_endpoints_bi.yaml"


class BadEndpointReason(BaseModel):
    token_usage: list[int] | None = None
    price_mismatch: str | None = None


class BadEndpoint(BaseModel):
    api: Literal["openrouter"]
    model: str
    provider: str | None = None
    reason: BadEndpointReason

    @classmethod
    def from_endpoint(
        cls, endpoint: Endpoint, reason: BadEndpointReason
    ) -> "BadEndpoint":
        return cls(
            api=endpoint.api,
            model=endpoint.model,
            provider=endpoint.provider,
            reason=reason,
        )

    def matches(self, endpoint: Endpoint) -> bool:
        return (
            self.api == endpoint.api
            and self.model == endpoint.model
            and self.provider == endpoint.provider
        )


def load_bad_endpoints_bi() -> list[BadEndpoint]:
    """Load the set of known bad BI endpoints from disk."""
    if not BAD_ENDPOINTS_BI_PATH.exists():
        return []
    with open(BAD_ENDPOINTS_BI_PATH) as f:
        data = yaml.safe_load(f) or {}
    return [
        BadEndpoint(
            api=e["api"],
            model=e["model"],
            provider=e.get("provider"),
            reason=BadEndpointReason(**e.get("reason", {})),
        )
        for e in data.get("bad_endpoints_bi", [])
    ]


def save_bad_endpoints_bi(bad_endpoints: Iterable[BadEndpoint]) -> None:
    """Save the list of bad BI endpoints to disk."""
    sorted_bad = sorted(
        bad_endpoints,
        key=lambda b: (b.api, b.model, b.provider or ""),
    )
    output_data = {
        "bad_endpoints_bi": [
            {
                "api": b.api,
                "model": b.model,
                "provider": b.provider,
                "reason": b.reason.model_dump(exclude_none=True),
            }
            for b in sorted_bad
        ]
    }
    with open(BAD_ENDPOINTS_BI_PATH, "w") as f:
        # default_flow_style=False for consistent dumping
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)


async def get_model_endpoints(
    session, model_id, logprob_filter: bool = False
) -> list[Endpoint]:
    """Fetch endpoints for a model.

    Args:
        session: aiohttp session
        model_id: OpenRouter model ID
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
                )
                filtered_endpoints.append(endpoint_data)
            return filtered_endpoints
    except Exception as e:
        logger.error(f"Error fetching endpoints for {model_id}: {e}")
        return []


async def get_endpoints(logprob_filter: bool = False) -> list[Endpoint]:
    """Get all endpoints for all models.

    Args:
        logprob_filter: If True, only return endpoints that claim to support logprobs
    """
    response = requests.get("https://openrouter.ai/api/v1/models")
    model_ids = [model["id"] for model in response.json()["data"]]
    all_endpoints = []
    async with aiohttp.ClientSession() as session:
        tasks = [
            get_model_endpoints(session, model_id, logprob_filter=logprob_filter)
            for model_id in model_ids
        ]
        async for result in gather_with_concurrency_streaming(
            config.api.max_workers, *tasks
        ):
            all_endpoints.extend(result)

    filtered_endpoints = [
        e for e in all_endpoints if e.cost[0] + e.cost[1] < config.api.max_cost_mtok
    ]

    if config.api.openrouter_avoid_free_endpoints:
        filtered_endpoints = [
            e for e in filtered_endpoints if e.cost[0] + e.cost[1] > 0
        ]

    log_msg = (
        f"Found {len(all_endpoints)} {'endpoints claiming logprobs support' if logprob_filter else 'total endpoints'}, "
        f"keeping {len(filtered_endpoints)} within max cost of ${config.api.max_cost_mtok}/Mtok"
    )
    if config.api.openrouter_avoid_free_endpoints:
        log_msg += " and excluding free endpoints"
    logger.info(log_msg)

    return filtered_endpoints


def compute_expected_cost(
    input_tokens: int, output_tokens: int, endpoint: Endpoint
) -> float:
    return (
        input_tokens * endpoint.cost[0] / 1e6 + output_tokens * endpoint.cost[1] / 1e6
    )


async def test_endpoint_token_usage(
    endpoint: Endpoint,
    price_tolerance: float = 0.01,
) -> tuple[Endpoint | None, BadEndpoint | None]:
    """Test if an endpoint uses acceptable token counts and correct pricing.

    Returns:
        (valid_endpoint, bad_endpoint) - at least one will be None.
    """
    async with OpenRouterClient() as client:
        try:
            response = await client.query(endpoint, "a", logprobs=False, temperature=0)
            if response.error:
                logger.info(
                    f"{endpoint} token test: ❌ (error: {response.error.message})"
                )
                return None, None  # Don't mark as bad on error, might be transient
            max_input_tokens = config.bi.max_input_tokens
            max_output_tokens = config.bi.max_output_tokens
            if response.input_tokens > max_input_tokens or (
                max_output_tokens is not None
                and response.output_tokens > max_output_tokens
            ):
                logger.info(
                    f"{endpoint} token test: ❌ "
                    f"(input={response.input_tokens}, output={response.output_tokens})"
                )
                bad = BadEndpoint.from_endpoint(
                    endpoint,
                    BadEndpointReason(
                        token_usage=[response.input_tokens, response.output_tokens]
                    ),
                )
                return None, bad

            # Verify pricing matches advertised cost using the generation endpoint
            expected_cost = compute_expected_cost(
                response.input_tokens, response.output_tokens, endpoint
            )
            if expected_cost > 0:
                if not response.generation_id:
                    logger.info(f"{endpoint} price test: ⏭️ (no generation_id)")
                    return None, None
                actual_cost = await client.get_generation_cost(response.generation_id)
                if actual_cost is None:
                    logger.info(
                        f"{endpoint} price test: ⏭️ (couldn't fetch generation cost for id {response.generation_id})"
                    )
                    return None, None
                if actual_cost > expected_cost * (1 + price_tolerance):
                    ratio = actual_cost / expected_cost
                    logger.info(
                        f"{endpoint} price test: ❌ "
                        f"(actual={actual_cost:.8f}, expected={expected_cost:.8f}, ratio={ratio:.2f})"
                    )
                    bad = BadEndpoint.from_endpoint(
                        endpoint,
                        BadEndpointReason(price_mismatch=f"{ratio:.2f}"),
                    )
                    return None, bad

            logger.info(
                f"{endpoint} token test: ✅ "
                f"(input={response.input_tokens}, output={response.output_tokens})"
            )
            return endpoint, None
        except Exception:
            logger.exception(f"Error testing token usage for {endpoint}")
            return None, None  # Don't mark as bad on exception, might be transient


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
    storage = ResultsStorage(config.data_dir)
    current_endpoints = config.endpoints_lt
    endpoints_to_keep = set([e for e in current_endpoints if not storage.is_stalled(e)])
    logger.info(
        f"Keeping {len(endpoints_to_keep)}/{len(current_endpoints)} non-stalled endpoints"
    )
    endpoints_claiming_logprobs = await get_endpoints(logprob_filter=True)

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


async def update_endpoints_bi():
    """Update the BI endpoints with all providers for all models.

    Filters out endpoints that use too many tokens (>10 input or >1 output).
    Bad endpoints are stored in bad_endpoints_bi.yaml to avoid re-testing.
    Good endpoints are stored in endpoints_bi.yaml and also skipped on re-runs.
    Only new endpoints (not in good or bad lists) are tested.
    """
    all_endpoints = await get_endpoints(logprob_filter=False)
    all_endpoints_set = set(all_endpoints)

    known_bad = load_bad_endpoints_bi()
    known_good = set(config.endpoints_bi)
    logger.info(
        f"Loaded {len(known_good)} known good endpoints, "
        f"{len(known_bad)} known bad endpoints"
    )

    # Update costs for known good endpoints if they're still in the API response
    still_good = []
    for good_endpoint in known_good:
        if good_endpoint in all_endpoints_set:
            updated = next(e for e in all_endpoints if e == good_endpoint)
            still_good.append(updated)
        else:
            still_good.append(good_endpoint)
    logger.info(f"Keeping all {len(still_good)} known good endpoints")

    # Filter out known bad and known good endpoints - only test new ones
    endpoints_to_test = [
        e
        for e in all_endpoints
        if not any(b.matches(e) for b in known_bad) and e not in known_good
    ]
    skipped_bad = len(
        [e for e in all_endpoints if any(b.matches(e) for b in known_bad)]
    )
    skipped_good = len([e for e in all_endpoints if e in known_good])
    logger.info(
        f"Testing {len(endpoints_to_test)} new endpoints "
        f"(skipping {skipped_bad} known bad, {skipped_good} known good)"
    )

    # Test token usage for each new endpoint
    tasks = [test_endpoint_token_usage(e) for e in endpoints_to_test]
    results = await gather_with_concurrency(config.api.max_workers, *tasks)

    newly_valid = [r[0] for r in results if r[0] is not None]
    new_bad = [r[1] for r in results if r[1] is not None]

    logger.info(
        f"Token test results: {len(newly_valid)} valid, "
        f"{len(new_bad)} new bad endpoints"
    )

    # Update bad endpoints list
    all_bad = known_bad + new_bad
    save_bad_endpoints_bi(all_bad)
    logger.info(f"Saved {len(all_bad)} total bad endpoints to {BAD_ENDPOINTS_BI_PATH}")

    # Combine still-good and newly-valid endpoints, deduplicating by identity
    all_good = list({e: e for e in still_good + newly_valid}.values())

    # Sort and save valid endpoints to endpoints_bi.yaml
    sorted_endpoints = sorted(
        all_good, key=lambda e: (sum(e.cost), e.api, e.model, e.provider)
    )

    output_data = {"endpoints_bi": []}
    for e in sorted_endpoints:
        output_data["endpoints_bi"].append(
            {
                "api": e.api,
                "model": e.model,
                "provider": e.provider,
                "cost": list(e.cost),
            }
        )

    with open(config.endpoints_yaml_path_bi, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    logger.info(
        f"Updated {config.endpoints_yaml_path_bi} with {len(sorted_endpoints)} endpoints"
    )


class LifecycleActions(BaseModel):
    onboard: list[Endpoint]
    recheck: list[EndpointBIState]
    delist: list[EndpointBIState]


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
    recheck = [
        s
        for s in states.values()
        if s.status == "retired"
        and s.endpoint in candidate_set
        and now - s.retired.last_recheck >= timedelta(days=r.recheck_days)
    ]
    delist = [
        s
        for s in states.values()
        if s.status == "monitoring" and s.endpoint not in candidate_set
    ]
    return LifecycleActions(onboard=onboard, recheck=recheck, delist=delist)


async def update_endpoints_bi_lifecycle():
    """Onboard new candidates, delist vanished endpoints, re-check retired ones."""
    states = load_all_states(config.bi.state_dir)
    candidates = config.endpoints_bi
    now = datetime.now(tz=timezone.utc).replace(microsecond=0)
    actions = select_lifecycle_actions(candidates, states, now)
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

    async with OpenRouterClient(timeout=60.0) as probe_client:
        strategies, _ = await resolve_strategies(probe_client, [e for e, _ in to_init])

    client = OpenRouterClient()
    try:
        for endpoint, is_recheck in to_init:
            slug = slugify(f"{endpoint.model}#{endpoint.provider}")
            state = states.get(slug)

            if str(endpoint) not in strategies:
                # Bump last_recheck so a permanently-hidden-reasoning endpoint isn't
                # re-probed daily. Fresh onboards have no state yet: skip silently.
                if is_recheck and state is not None:
                    state.retired.last_recheck = now
                    state.save(config.bi.state_dir)
                continue

            old_bis = []
            epoch = await reinit(
                client, strategies[str(endpoint)], endpoint, old_bis, now
            )
            if state is None:
                state = EndpointBIState(
                    endpoint=endpoint, status="monitoring", epochs=[]
                )
            if epoch is None:
                state.status = "retired"
                state.retired = RetiredInfo(
                    reason="no_bis", since=now, last_recheck=now
                )
            else:
                state.status = "monitoring"
                state.retired = None
                state.epochs.append(epoch)
            state.save(config.bi.state_dir)
    finally:
        await client.close()


async def main():
    await update_endpoints_lt()
    await update_endpoints_bi()
    await update_endpoints_bi_lifecycle()


if __name__ == "__main__":
    asyncio.run(main())
