"""Daily BI monitor: sample border inputs, detect changes, trigger re-init."""

import asyncio
from datetime import datetime, timezone
from typing import Literal

import fire
from pydantic import BaseModel

from trackllm_website.api import OpenRouterClient, QueryTooExpensive
from trackllm_website.bi.results import load_phase2_results
from trackllm_website.bi.common import (
    TOO_EXPENSIVE,
    QueryStrategy,
    probe_errors_unreachable,
    resolve_strategies,
)
from trackllm_website.bi.detection import adaptive_transitions, epoch_tv_series
from trackllm_website.bi.phase_2 import (
    get_output_path,
    load_existing_results,
    save_results,
)
from trackllm_website.bi.reinit import cleanup_onboarding_progress, reinit
from trackllm_website.bi.scan import changepoint_scan
from trackllm_website.bi.sampling import sample_prompts
from trackllm_website.bi.state import EndpointBIState, RetiredInfo, load_all_states
from trackllm_website.bi.digest import (
    MonitorReport,
    MonitorRow,
    send_monitoring_digest,
)
from trackllm_website.config import config, logger
from trackllm_website.spend import Spend, append_entry, track
from trackllm_website.util import gather_with_concurrency


class Decision(BaseModel):
    action: Literal["none", "reinit", "retire_stalled"]
    change_date: datetime | None = None
    detector: Literal["adaptive", "scan"] | None = None


def _retire_too_expensive(
    state: EndpointBIState,
    detail: str,
    spent: float,
    now: datetime,
    event_rows: list[MonitorRow] | None,
) -> None:
    """Immediate retirement on a guard trip: no grace, no streak (see
    QueryTooExpensive). The recheck schedule re-vets it; vetting's own guard
    keeps it out until the price actually drops."""
    epoch = state.current_epoch
    if epoch is not None:
        epoch.end = now
        epoch.end_reason = "too_expensive"
    state.status = "retired"
    state.retired = RetiredInfo(reason="too_expensive", since=now, last_recheck=now)
    state.save(config.bi.state_dir)
    # A guard trip mid-discovery leaves resumable phase-1 scratch that would
    # otherwise be committed daily; the endpoint is retired, so drop it.
    cleanup_onboarding_progress(state.endpoint)
    logger.warning(f"{state.endpoint}: retired (too expensive: {detail})")
    if event_rows is not None:
        event_rows.append(
            MonitorRow(
                state.endpoint.model,
                state.endpoint.provider,
                "retired_too_expensive",
                None,
                None,
                spent,
            )
        )


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

    epoch_results = epoch.filter_results(results)

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
            detector="adaptive",
        )
    # Young epochs are invisible to the adaptive rule (it needs ~9 batches of
    # baseline, then absorbs whatever level it finds); the changepoint scan
    # covers that window.
    scan_event = changepoint_scan(epoch_results)
    if scan_event is not None:
        return Decision(
            action="reinit",
            change_date=datetime.fromisoformat(scan_event.split_ts),
            detector="scan",
        )
    return Decision(action="none")


async def run_endpoint(
    client: OpenRouterClient,
    strategy: QueryStrategy,
    state: EndpointBIState,
    now: datetime,
    probe_spend: dict[str, Spend] | None = None,
    event_rows: list[MonitorRow] | None = None,
) -> None:
    epoch = state.current_epoch
    assert epoch is not None

    guard_exc: QueryTooExpensive | None = None
    with track() as monitor_spend:
        try:
            samples, n_errors = await sample_prompts(
                client,
                state.endpoint,
                strategy,
                epoch.border_inputs,
                config.bi.phase_2.queries_per_token,
                temperature=0.0,
            )
            path = get_output_path(state.endpoint, now.strftime("%Y-%m"))
            existing = load_existing_results(path)
            batch_key = now.replace(microsecond=0).isoformat()
            for prompt, prompt_samples in samples.items():
                existing.setdefault(prompt, {})[batch_key] = prompt_samples
            save_results(path, existing)
        except QueryTooExpensive as e:
            guard_exc = e
    if probe_spend is not None:
        monitor_spend.merge(probe_spend.get(str(state.endpoint), Spend()))
    append_entry(config.spend_dir, state.slug, "monitor", monitor_spend, now)
    if guard_exc is not None:
        _retire_too_expensive(
            state, str(guard_exc), monitor_spend.cost, now, event_rows
        )
        return

    # A batch where every query errored must show up in the digest on day 1, not
    # only as a "retired (stalled)" surprise stall_days later.
    n_queries = len(epoch.border_inputs) * config.bi.phase_2.queries_per_token
    if n_queries and n_errors >= n_queries:
        logger.warning(f"{state.endpoint}: all {n_queries} monitor queries errored")
        if event_rows is not None:
            event_rows.append(
                MonitorRow(
                    state.endpoint.model,
                    state.endpoint.provider,
                    "all_errors",
                    None,
                    None,
                    0.0,
                )
            )

    results = load_phase2_results(config.bi.phase_2_dir / state.slug)
    decision = decide(state, results, now)

    if decision.action == "retire_stalled":
        epoch.end = now
        epoch.end_reason = "stalled"
        state.status = "retired"
        state.retired = RetiredInfo(reason="stalled", since=now, last_recheck=now)
        logger.warning(f"{state.endpoint}: retired (stalled)")
        if event_rows is not None:
            event_rows.append(
                MonitorRow(
                    state.endpoint.model,
                    state.endpoint.provider,
                    "retired_stalled",
                    None,
                    None,
                    0.0,
                )
            )
    elif decision.action == "reinit":
        logger.warning(
            f"{state.endpoint}: change detected (onset {decision.change_date})"
        )
        change_date_str = (
            decision.change_date.date().isoformat() if decision.change_date else None
        )

        # The epoch is closed only once the re-init's outcome is known, so that the
        # timeout path can persist the streak below without also persisting a
        # half-closed epoch.
        def close_epoch() -> None:
            epoch.end = now
            epoch.end_reason = "change_detected"
            epoch.change_date = decision.change_date
            detector_cfg = (
                config.bi.scan if decision.detector == "scan" else config.bi.detection
            )
            epoch.params = {"detector": decision.detector, **detector_cfg.model_dump()}

        timed_out = False
        with track() as reinit_spend:
            try:
                result = await asyncio.wait_for(
                    reinit(client, strategy, state.endpoint, epoch.border_inputs, now),
                    # A re-init is a full discovery run (~15k queries) and it happens
                    # inside the daily monitor job, so one hanging endpoint would stall
                    # the whole workflow. Its own deadline, not the onboarding one:
                    # this job's budget is 300 minutes (see config.toml).
                    timeout=config.bi.monitor.reinit_timeout_seconds,
                )
            except asyncio.TimeoutError:
                timed_out = True
            except QueryTooExpensive as e:
                guard_exc = e
        append_entry(config.spend_dir, state.slug, "reinit", reinit_spend, now)
        if guard_exc is not None:
            _retire_too_expensive(
                state, str(guard_exc), reinit_spend.cost, now, event_rows
            )
            return
        if timed_out:
            state.reinit_timeout_streak += 1
            if state.reinit_timeout_streak >= config.bi.monitor.max_reinit_timeouts:
                # Every retry re-detects the same change and burns the full timeout
                # again (deepseek-v4-flash@fireworks, rate-limited upstream, did this
                # daily): give up and retire. The recheck schedule re-onboards it
                # from scratch later. Giving up is a decision, not a failure.
                close_epoch()
                state.status = "retired"
                state.retired = RetiredInfo(
                    reason="reinit_timeout", since=now, last_recheck=now
                )
                logger.warning(
                    f"{state.endpoint}: retired after "
                    f"{state.reinit_timeout_streak} consecutive re-init timeouts"
                )
                if event_rows is not None:
                    event_rows.append(
                        MonitorRow(
                            state.endpoint.model,
                            state.endpoint.provider,
                            "retired_reinit_timeout",
                            change_date_str,
                            None,
                            reinit_spend.cost,
                        )
                    )
                state.save(config.bi.state_dir)
                return
            # Persist only the streak: the epoch stays open, so the next daily run
            # re-detects the change and retries (resuming phase-1 progress).
            state.save(config.bi.state_dir)
            if event_rows is not None:
                event_rows.append(
                    MonitorRow(
                        state.endpoint.model,
                        state.endpoint.provider,
                        "reinit_timeout",
                        change_date_str,
                        None,
                        reinit_spend.cost,
                    )
                )
            logger.warning(
                f"{state.endpoint}: re-init exceeded "
                f"{config.bi.monitor.reinit_timeout_seconds}s (streak "
                f"{state.reinit_timeout_streak}/{config.bi.monitor.max_reinit_timeouts})"
            )
            return
        state.reinit_timeout_streak = 0
        close_epoch()
        if event_rows is not None:
            event_rows.append(
                MonitorRow(
                    state.endpoint.model,
                    state.endpoint.provider,
                    event="reonboarded" if result.epoch else "reonboard_no_bis",
                    change_date=change_date_str,
                    n_bis_after=(
                        len(result.epoch.border_inputs) if result.epoch else None
                    ),
                    spent=reinit_spend.cost,
                )
            )
        if result.epoch is None:
            # The temperature gate runs only on discovery (old_bis empty); a monitor
            # reinit always has old_bis, so reason is no_bis here. Retire either way.
            state.status = "retired"
            state.retired = RetiredInfo(reason="no_bis", since=now, last_recheck=now)
        else:
            state.epochs.append(result.epoch)
    # State is saved once a re-init's outcome is settled (or immediately for the
    # other actions). A crash mid-reinit persists nothing; the next daily run
    # idempotently re-detects the change and retries (facts-vs-derivations design:
    # state files record only committed facts).
    state.save(config.bi.state_dir)


async def monitor() -> MonitorReport:
    states = load_all_states(config.bi.state_dir)
    monitoring = [s for s in states.values() if s.status == "monitoring"]
    logger.info(f"Monitoring {len(monitoring)} endpoints")
    now = datetime.now(tz=timezone.utc).replace(microsecond=0)

    probe_spend: dict[str, Spend] = {}
    async with OpenRouterClient(timeout=60.0) as probe_client:
        strategies, failed = await resolve_strategies(
            probe_client, [s.endpoint for s in monitoring], probe_spend=probe_spend
        )

    client = OpenRouterClient()
    event_rows: list[MonitorRow] = []
    failures: list[str] = []

    # An endpoint dropped by resolve_strategies is never sampled: no batch means
    # the stall detector never advances, so without a digest row it would sit
    # in "monitoring" forever, invisibly. The row recurs daily until resolved.
    for state in monitoring:
        key = str(state.endpoint)
        if key in strategies:
            continue
        # A probe that tripped the cost guard would otherwise be re-probed (and
        # re-billed) daily forever: retire now, like any other guard trip.
        errors = failed.get(key, [])
        if errors and errors[0] == TOO_EXPENSIVE:
            spend = probe_spend.get(key, Spend())
            append_entry(config.spend_dir, state.slug, "monitor", spend, now)
            _retire_too_expensive(
                state, "; ".join(errors[1:]) or errors[0], spend.cost, now, event_rows
            )
            continue
        # All probes 404'd and the catalog already dropped the endpoint: the
        # provider is gone from OpenRouter's routing, so retire it now rather
        # than fail the run daily for the rest of the deselection grace period.
        # If it comes back, the catalog re-lists it and the recheck schedule
        # resurrects it. All-404 on a still-listed endpoint is an anomaly, not
        # a known removal: no retirement, a red digest row every day it lasts.
        if state.deselected_since is not None and probe_errors_unreachable(
            failed.get(key, [])
        ):
            epoch = state.current_epoch
            if epoch is not None:
                epoch.end = now
                epoch.end_reason = "unreachable"
            state.status = "retired"
            state.retired = RetiredInfo(
                reason="unreachable", since=now, last_recheck=now
            )
            state.save(config.bi.state_dir)
            logger.warning(f"{key}: retired (unreachable, all probes 404)")
            event_rows.append(
                MonitorRow(
                    state.endpoint.model,
                    state.endpoint.provider,
                    "retired_unreachable",
                    None,
                    None,
                    0.0,
                )
            )
            continue
        logger.error(f"{key}: no strategy resolved ({failed.get(key)}); skipped")
        event_rows.append(
            MonitorRow(
                state.endpoint.model,
                state.endpoint.provider,
                "probes_failed",
                None,
                None,
                0.0,
            )
        )

    # Absolute, so the probe above and every wave of endpoints eat into the same
    # budget: what matters is when the *job* ends, not how long each part took.
    deadline = (
        asyncio.get_running_loop().time() + config.bi.monitor.job_deadline_seconds
    )

    def cut_off(state: EndpointBIState, why: str) -> None:
        logger.error(f"{state.endpoint}: {why}")
        event_rows.append(
            MonitorRow(
                state.endpoint.model,
                state.endpoint.provider,
                "deadline_cutoff",
                None,
                None,
                0.0,
            )
        )

    async def run_isolated(state: EndpointBIState) -> None:
        if asyncio.get_running_loop().time() >= deadline:
            cut_off(state, "job deadline passed, not started")
            return
        try:
            # Nested so that only the job deadline expiring reaches the outer
            # handler: an exception escaping run_endpoint is a different event —
            # a bug, the only thing that still fails the workflow.
            async with asyncio.timeout_at(deadline):
                try:
                    await run_endpoint(
                        client,
                        strategies[str(state.endpoint)],
                        state,
                        now,
                        probe_spend,
                        event_rows=event_rows,
                    )
                except Exception:
                    logger.exception(f"Monitor run failed for {state.endpoint}")
                    failures.append(str(state.endpoint))
                    event_rows.append(
                        MonitorRow(
                            state.endpoint.model,
                            state.endpoint.provider,
                            "error",
                            None,
                            None,
                            0.0,
                        )
                    )
        except TimeoutError:
            # Nothing is persisted beyond the batch run_endpoint already saved, so
            # the next run re-detects whatever this one was in the middle of.
            cut_off(state, "cut off by the job deadline")

    try:
        runnable = [s for s in monitoring if str(s.endpoint) in strategies]
        await gather_with_concurrency(
            config.bi.monitor.max_concurrent_endpoints,
            *(run_isolated(s) for s in runnable),
        )
    finally:
        await client.close()

    return MonitorReport(
        date=now.date().isoformat(),
        rows=event_rows,
        n_endpoints=len(monitoring),
        failures=failures,
    )


def main() -> None:
    report = asyncio.run(monitor())
    send_monitoring_digest(report, config.spend_dir)
    # Fail the run (after saving and sending the digest) only for exceptions that
    # escaped run_endpoint: a failure email means a bug. Diagnosed conditions
    # (re-init timeouts, unresolved probes, deadline cutoffs) are digest rows.
    if report.failures:
        raise RuntimeError(
            f"monitor run failed unexpectedly for {len(report.failures)} "
            f"endpoint(s): {', '.join(report.failures)}"
        )


if __name__ == "__main__":
    fire.Fire(main)
