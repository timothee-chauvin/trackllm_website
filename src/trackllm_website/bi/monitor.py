"""Daily BI monitor: sample border inputs, detect changes, trigger re-init."""

import asyncio
from datetime import datetime, timezone
from typing import Literal

import fire
from pydantic import BaseModel

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.analyze import load_phase2_results
from trackllm_website.bi.common import QueryStrategy, resolve_strategies
from trackllm_website.bi.detection import (
    adaptive_transitions,
    epoch_tv_series,
    is_unstable,
)
from trackllm_website.bi.phase_2 import (
    get_output_path,
    load_existing_results,
    save_results,
)
from trackllm_website.bi.reinit import reinit
from trackllm_website.bi.sampling import sample_prompts
from trackllm_website.bi.state import EndpointBIState, RetiredInfo, load_all_states
from trackllm_website.config import config, logger


class Decision(BaseModel):
    action: Literal["none", "reinit", "retire_stalled"]
    change_date: datetime | None = None
    unstable: bool = False


def _day_has_samples(results: dict, day: str) -> bool | None:
    """True/False if the day was queried with/without successes, None if not queried."""
    queried = False
    for batches in results.values():
        for ts, samples in batches.items():
            if ts[:10] == day:
                queried = True
                if samples:
                    return True
    return False if queried else None


def decide(state: EndpointBIState, results: dict, now: datetime) -> Decision:
    epoch = state.current_epoch
    if epoch is None:
        return Decision(action="none")

    epoch_results = {
        p: {
            ts: s
            for ts, s in results.get(p, {}).items()
            if datetime.fromisoformat(ts) >= epoch.start
        }
        for p in epoch.border_inputs
    }

    # Stall: the most recent stall_days queried days all had zero successes.
    # Checked before detection because a dead endpoint may still carry a pending
    # change event from before it went silent.
    recent_days = sorted(
        {ts[:10] for b in epoch_results.values() for ts in b}, reverse=True
    )[: config.bi.reinit.stall_days]
    if len(recent_days) >= config.bi.reinit.stall_days and all(
        _day_has_samples(epoch_results, day) is False for day in recent_days
    ):
        return Decision(action="retire_stalled")

    tv = epoch_tv_series(epoch.reference, epoch_results)
    events = adaptive_transitions(tv)
    if events:
        return Decision(
            action="reinit",
            change_date=datetime.fromisoformat(events[-1]),
            unstable=is_unstable(tv),
        )
    return Decision(action="none", unstable=is_unstable(tv))


async def run_endpoint(
    client: OpenRouterClient,
    strategy: QueryStrategy,
    state: EndpointBIState,
    now: datetime,
) -> None:
    epoch = state.current_epoch
    assert epoch is not None

    samples, _ = await sample_prompts(
        client,
        state.endpoint,
        strategy,
        epoch.border_inputs,
        config.bi.phase_2.queries_per_token,
    )
    path = get_output_path(state.endpoint, now.strftime("%Y-%m"))
    existing = load_existing_results(path)
    batch_key = now.replace(microsecond=0).isoformat()
    for prompt, prompt_samples in samples.items():
        existing.setdefault(prompt, {})[batch_key] = prompt_samples
    save_results(path, existing)

    results = load_phase2_results(config.bi.phase_2_dir / state.slug)
    decision = decide(state, results, now)

    if decision.action == "retire_stalled":
        epoch.end = now
        epoch.end_reason = "stalled"
        state.status = "retired"
        state.retired = RetiredInfo(reason="stalled", since=now, last_recheck=now)
        logger.warning(f"{state.endpoint}: retired (stalled)")
    elif decision.action == "reinit":
        epoch.end = now
        epoch.end_reason = "change_detected"
        epoch.change_date = decision.change_date
        epoch.params = config.bi.detection.model_dump()
        logger.warning(
            f"{state.endpoint}: change detected (onset {decision.change_date})"
        )
        new_epoch = await reinit(
            client, strategy, state.endpoint, epoch.border_inputs, now
        )
        if new_epoch is None:
            state.status = "retired"
            state.retired = RetiredInfo(reason="no_bis", since=now, last_recheck=now)
        else:
            state.epochs.append(new_epoch)
    state.save(config.bi.state_dir)


async def monitor() -> None:
    states = load_all_states(config.bi.state_dir)
    monitoring = [s for s in states.values() if s.status == "monitoring"]
    logger.info(f"Monitoring {len(monitoring)} endpoints")
    now = datetime.now(tz=timezone.utc).replace(microsecond=0)

    async with OpenRouterClient(timeout=60.0) as probe_client:
        strategies, _failed = await resolve_strategies(
            probe_client, [s.endpoint for s in monitoring]
        )

    client = OpenRouterClient()
    try:
        runnable = [s for s in monitoring if str(s.endpoint) in strategies]
        outcomes = await asyncio.gather(
            *(
                run_endpoint(client, strategies[str(s.endpoint)], s, now)
                for s in runnable
            ),
            return_exceptions=True,
        )
    finally:
        await client.close()

    for state, outcome in zip(runnable, outcomes):
        if isinstance(outcome, Exception):
            logger.warning(f"{state.endpoint}: monitor run failed: {outcome!r}")


if __name__ == "__main__":
    fire.Fire(monitor)
