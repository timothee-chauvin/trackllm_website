import asyncio
from datetime import datetime, timezone
from pathlib import Path

import orjson

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi import monitor as monitor_mod
from trackllm_website.bi.common import PlainStrategy
from trackllm_website.bi.migrate_state import migrate_endpoint
from trackllm_website.bi.monitor import Decision, decide, monitor, run_endpoint
from trackllm_website.config import Endpoint
from trackllm_website.storage import Response
from trackllm_website.util import slugify

FIXTURES = Path("tests/fixtures/phase_2")
ENDPOINT = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))


def open_state_from_fixture(slug: str):
    results = orjson.loads((FIXTURES / slug / "data.json").read_bytes())
    state = migrate_endpoint(ENDPOINT, results)
    state.status = "monitoring"
    state.retired = None
    epoch = state.epochs[0]
    epoch.end = None
    epoch.end_reason = None
    return state, results


def test_change_detected_closes_epoch():
    state, results = open_state_from_fixture(
        "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8"
    )
    decision = decide(state, results, datetime(2026, 2, 15, tzinfo=timezone.utc))
    assert decision.action == "reinit"
    assert decision.change_date.date().isoformat() == "2026-01-24"
    assert decision.detector == "adaptive"


def test_early_change_caught_by_scan():
    # hy3 @ atlas-cloud: model swap at batch 3, invisible to the adaptive rule
    state, results = open_state_from_fixture("tencent2fhy323atlas-cloud2ffp8")
    decision = decide(state, results, datetime(2026, 7, 28, tzinfo=timezone.utc))
    assert decision.action == "reinit"
    assert decision.detector == "scan"
    assert decision.change_date.date().isoformat() == "2026-07-20"


def test_scan_split_below_tv_threshold_is_not_a_change(monkeypatch):
    # A permutation-significant split that moves TV by less than abs_delta is
    # the "visually nonexistent change": the epoch must stay open.
    from trackllm_website.bi.scan import ScanEvent

    state, results = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    split = sorted({ts for b in results.values() for ts in b})[3]
    monkeypatch.setattr(
        monitor_mod,
        "changepoint_scan",
        lambda *_: ScanEvent(split_ts=split, p_value=0.001),
    )
    decision = decide(state, results, datetime(2026, 2, 15, tzinfo=timezone.utc))
    assert decision.action == "none"


def test_stable_endpoint_no_action():
    state, results = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    decision = decide(state, results, datetime(2026, 2, 15, tzinfo=timezone.utc))
    assert decision.action == "none"


def test_stalled_endpoint_retired():
    # mistral-7b together: all queries error after 2026-02-25
    state, results = open_state_from_fixture(
        "mistralai2fmistral-7b-instruct-v0.323together"
    )
    decision = decide(state, results, datetime(2026, 3, 10, tzinfo=timezone.utc))
    assert decision.action == "retire_stalled"


def test_no_current_epoch_no_action():
    state, results = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    state.epochs[0].end = datetime(2026, 2, 1, tzinfo=timezone.utc)
    decision = decide(state, results, datetime(2026, 2, 15, tzinfo=timezone.utc))
    assert decision == Decision(action="none")


def test_run_endpoint_reinit_retires_and_persists(tmp_path, monkeypatch):
    state, results = open_state_from_fixture(
        "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8"
    )
    epoch = state.current_epoch
    now = datetime(2026, 2, 15, tzinfo=timezone.utc)

    daily_batch = {bi: [(now.isoformat(), "tok")] for bi in epoch.border_inputs}
    monthly_path = tmp_path / "monthly.json"

    async def fake_sample_prompts(*args, **kwargs):
        return daily_batch, 0

    async def fake_reinit(*args, **kwargs):
        from trackllm_website.bi.reinit import ReinitResult

        return ReinitResult(epoch=None, reason="no_bis")  # retired "no_bis"

    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)
    # run_endpoint now writes a spend-ledger line; keep it in tmp, not the repo.
    monkeypatch.setattr(
        type(monitor_mod.config), "spend_dir", property(lambda self: tmp_path / "spend")
    )
    state_dir = monitor_mod.config.bi.state_dir
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)
    monkeypatch.setattr(monitor_mod, "reinit", fake_reinit)
    monkeypatch.setattr(monitor_mod, "get_output_path", lambda ep, ym: monthly_path)
    monkeypatch.setattr(monitor_mod, "load_phase2_results", lambda d: results)

    async def go():
        async with OpenRouterClient() as client:
            await run_endpoint(client, PlainStrategy(), state, now)

    asyncio.run(go())

    # (a) old epoch closed with change_detected and params recorded
    assert epoch.end == now
    assert epoch.end_reason == "change_detected"
    assert epoch.params is not None
    # (b) state retired with reason no_bis
    assert state.status == "retired"
    assert state.retired is not None and state.retired.reason == "no_bis"
    # (c) daily batch merged into the monthly file
    written = orjson.loads(monthly_path.read_bytes())
    a_bi = epoch.border_inputs[0]
    assert written[a_bi][now.isoformat()] == [[now.isoformat(), "tok"]]
    # (d) state file written into the tmp state dir
    assert (state_dir / f"{state.slug}.json").exists()


def test_all_errors_batch_lands_in_event_rows(tmp_path, monkeypatch):
    """A batch where every query errored must reach the run summary, not vanish."""
    state, results = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    epoch = state.current_epoch
    now = datetime(2026, 2, 15, tzinfo=timezone.utc)
    n_queries = (
        len(epoch.border_inputs) * monitor_mod.config.bi.phase_2.queries_per_token
    )

    async def fake_sample_prompts(*args, **kwargs):
        return {bi: [] for bi in epoch.border_inputs}, n_queries

    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        type(monitor_mod.config), "spend_dir", property(lambda self: tmp_path / "spend")
    )
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)
    monkeypatch.setattr(
        monitor_mod, "get_output_path", lambda ep, ym: tmp_path / "monthly.json"
    )
    monkeypatch.setattr(monitor_mod, "load_phase2_results", lambda d: results)

    event_rows = []

    async def go():
        async with OpenRouterClient() as client:
            await run_endpoint(
                client, PlainStrategy(), state, now, event_rows=event_rows
            )

    asyncio.run(go())
    assert any(r.event == "all_errors" for r in event_rows)


def _reinit_case(tmp_path, monkeypatch, reinit_impl):
    """Wire run_endpoint up to a change-detecting fixture with a fake re-init."""
    state, results = open_state_from_fixture(
        "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8"
    )
    epoch = state.current_epoch
    now = datetime(2026, 2, 15, tzinfo=timezone.utc)

    async def fake_sample_prompts(*args, **kwargs):
        return {bi: [(now.isoformat(), "tok")] for bi in epoch.border_inputs}, 0

    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        type(monitor_mod.config), "spend_dir", property(lambda self: tmp_path / "spend")
    )
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)
    monkeypatch.setattr(monitor_mod, "reinit", reinit_impl)
    monkeypatch.setattr(
        monitor_mod, "get_output_path", lambda ep, ym: tmp_path / "monthly.json"
    )
    monkeypatch.setattr(monitor_mod, "load_phase2_results", lambda d: results)
    monkeypatch.setattr(monitor_mod.config.bi.monitor, "reinit_timeout_seconds", 0.05)

    event_rows = []

    def go():
        async def _go():
            async with OpenRouterClient() as client:
                await run_endpoint(
                    client, PlainStrategy(), state, now, event_rows=event_rows
                )

        asyncio.run(_go())

    return state, now, event_rows, go


async def _hanging_reinit(*args, **kwargs):
    await asyncio.sleep(10)
    raise AssertionError("re-init should have been cancelled by the timeout")


def _saved_state(state):
    from trackllm_website.bi.state import EndpointBIState

    path = monitor_mod.config.bi.state_dir / f"{state.slug}.json"
    return EndpointBIState.model_validate_json(path.read_bytes())


def test_reinit_timeout_is_a_digest_row_not_a_failure(tmp_path, monkeypatch):
    """A hanging change-triggered re-init (full discovery is ~15k queries) must not
    stall the daily monitor job: it is bounded, and only the timeout streak is
    persisted — the epoch stays open so the next run retries. It is a diagnosed
    condition with its own resolution path (retirement at max_reinit_timeouts), so
    it is reported in the digest, not as a workflow failure."""
    state, now, event_rows, go = _reinit_case(tmp_path, monkeypatch, _hanging_reinit)

    go()  # must not raise: a failed workflow is reserved for bugs

    assert any(r.event == "reinit_timeout" for r in event_rows)
    saved = _saved_state(state)
    assert saved.reinit_timeout_streak == 1
    assert saved.status == "monitoring"
    # epoch still open: the next daily run re-detects the change and retries
    assert saved.epochs[-1].end is None
    # the spend of the abandoned re-init is still recorded
    assert (tmp_path / "spend").exists()


def test_reinit_timeout_streak_hits_threshold_and_retires(tmp_path, monkeypatch):
    """An endpoint whose re-init times out every day would otherwise re-detect the
    same change and burn the full timeout forever: after max_reinit_timeouts, give
    up and retire it (the recheck schedule re-onboards it from scratch later)."""
    state, now, event_rows, go = _reinit_case(tmp_path, monkeypatch, _hanging_reinit)
    state.reinit_timeout_streak = monitor_mod.config.bi.monitor.max_reinit_timeouts - 1

    go()  # no TimeoutError: giving up is a decision, not a failure

    assert [r.event for r in event_rows] == ["retired_reinit_timeout"]
    saved = _saved_state(state)
    assert saved.status == "retired"
    assert saved.retired.reason == "reinit_timeout"
    # the detected change is recorded as the closed epoch's end, as facts
    assert saved.epochs[-1].end_reason == "change_detected"
    assert saved.epochs[-1].change_date is not None


def test_successful_reinit_resets_timeout_streak(tmp_path, monkeypatch):
    from trackllm_website.bi.reinit import ReinitResult
    from trackllm_website.bi.state import Epoch

    now = datetime(2026, 2, 15, tzinfo=timezone.utc)

    async def ok_reinit(*args, **kwargs):
        return ReinitResult(
            epoch=Epoch(start=now, border_inputs=["bi"], reference={"bi": []}),
            reason="ok",
        )

    state, now, event_rows, go = _reinit_case(tmp_path, monkeypatch, ok_reinit)
    state.reinit_timeout_streak = 1

    go()

    saved = _saved_state(state)
    assert saved.reinit_timeout_streak == 0
    assert saved.status == "monitoring"
    assert saved.epochs[-2].end_reason == "change_detected"
    assert saved.epochs[-1].end is None


class FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def test_unresolved_strategy_is_a_digest_row(monkeypatch):
    """An endpoint resolve_strategies drops must not be silently skipped: no batch
    is written, so the stall detector never advances and it would idle forever. A
    probes_failed digest row keeps it visible every day it recurs, without paging
    a workflow failure for a condition that is not a bug."""
    state, _ = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    report = _monitor_unresolved(monkeypatch, state, ["plain: 429 rate limited"])
    assert report.failures == []
    assert [r.event for r in report.rows] == ["probes_failed"]


def _monitor_unresolved(monkeypatch, state, errors):
    async def fake_resolve(client, endpoints, **kwargs):
        return {}, {str(state.endpoint): errors}

    monkeypatch.setattr(monitor_mod, "load_all_states", lambda d: {state.slug: state})
    monkeypatch.setattr(monitor_mod, "resolve_strategies", fake_resolve)
    monkeypatch.setattr(monitor_mod, "OpenRouterClient", FakeClient)
    return asyncio.run(monitor())


PROBE_404 = 'plain: 404 {"message":"No allowed providers are available"}'


def test_unreachable_deselected_endpoint_is_retired_not_failed(tmp_path, monkeypatch):
    """All probes 404 and the catalog already dropped the endpoint: the provider is
    gone from OpenRouter's routing. Retire it instead of failing the run daily for
    the rest of the 30-day deselection grace period (runs #46/#47)."""
    state, _ = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    state.deselected_since = datetime(2026, 7, 29, tzinfo=timezone.utc)
    epoch = state.current_epoch
    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)

    report = _monitor_unresolved(monkeypatch, state, [PROBE_404, PROBE_404])

    assert report.failures == []
    assert state.status == "retired"
    assert state.retired.reason == "unreachable"
    assert epoch.end is not None and epoch.end_reason == "unreachable"
    assert [r.event for r in report.rows] == ["retired_unreachable"]
    assert (monitor_mod.config.bi.state_dir / f"{state.slug}.json").exists()


def test_unreachable_but_still_listed_endpoint_is_not_retired(monkeypatch):
    """All-404 probes on an endpoint the catalog still lists is an anomaly, not a
    known removal: no retirement, but a red digest row every day until the catalog
    deselects it (or the provider comes back)."""
    state, _ = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    assert state.deselected_since is None

    report = _monitor_unresolved(monkeypatch, state, [PROBE_404])

    assert report.failures == []
    assert [r.event for r in report.rows] == ["probes_failed"]
    assert state.status == "monitoring"


def test_transient_probe_failure_is_not_a_retirement(monkeypatch):
    """429s/timeouts are transient even on a deselected endpoint: not proof the
    provider is gone, so no retirement — a digest row, retried tomorrow."""
    state, _ = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    state.deselected_since = datetime(2026, 7, 29, tzinfo=timezone.utc)

    report = _monitor_unresolved(monkeypatch, state, ["plain: 429 rate limited"])

    assert report.failures == []
    assert [r.event for r in report.rows] == ["probes_failed"]
    assert state.status == "monitoring"


def _monitor_one_endpoint(monkeypatch, state, run_endpoint_impl):
    async def fake_resolve(client, endpoints, **kwargs):
        return {str(state.endpoint): PlainStrategy()}, {}

    monkeypatch.setattr(monitor_mod, "load_all_states", lambda d: {state.slug: state})
    monkeypatch.setattr(monitor_mod, "resolve_strategies", fake_resolve)
    monkeypatch.setattr(monitor_mod, "OpenRouterClient", FakeClient)
    monkeypatch.setattr(monitor_mod, "run_endpoint", run_endpoint_impl)
    return asyncio.run(monitor())


class FakeQueryClient(OpenRouterClient):
    """Answers instantly, except for endpoints that never answer at all.

    A real subclass because run_endpoint's signature is beartype-enforced; no
    __init__ chaining, so no session and no API key.
    """

    def __init__(self):
        self.hangs: set[str] = set()

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, endpoint, prompt, **kwargs):
        if str(endpoint) in self.hangs:
            await asyncio.sleep(30)
        return Response(
            date=datetime(2026, 2, 15, tzinfo=timezone.utc),
            endpoint=endpoint,
            prompt=prompt,
            content="tok",
            cost=0.0,
        )


def _monitor_endpoints(monkeypatch, tmp_path, states, hangs):
    """Run monitor() over `states` with a fake API, one query per border input.

    Only the network is faked: run_endpoint, sample_prompts and decide are the real
    ones, so a cut-off has to unwind the sampler the same way it does in production.
    """
    client = FakeQueryClient()
    client.hangs = {str(s.endpoint) for s in states if s.slug in hangs}

    async def fake_resolve(probe_client, endpoints, **kwargs):
        return {str(ep): PlainStrategy() for ep in endpoints}, {}

    monkeypatch.setattr(
        monitor_mod, "load_all_states", lambda d: {s.slug: s for s in states}
    )
    monkeypatch.setattr(monitor_mod, "resolve_strategies", fake_resolve)
    monkeypatch.setattr(monitor_mod, "OpenRouterClient", lambda *a, **kw: client)
    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        type(monitor_mod.config), "spend_dir", property(lambda self: tmp_path / "spend")
    )
    monkeypatch.setattr(
        monitor_mod,
        "get_output_path",
        lambda ep, ym: tmp_path / f"{slugify(str(ep))}.json",
    )
    monkeypatch.setattr(monitor_mod.config.bi.phase_2, "queries_per_token", 1)
    monkeypatch.setattr(
        monitor_mod.config.bi.phase_2, "requests_per_second_per_endpoint", 1000
    )
    return asyncio.run(monitor())


def test_deadline_cutoff_keeps_the_work_the_run_already_did(tmp_path, monkeypatch):
    """The reason the deadline exists: whatever finished must still be on disk for
    the commit step. A hung endpoint costs its own batch, not everyone else's."""
    done, results = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    done.endpoint = Endpoint(
        api="openrouter", model="m/done", provider="p", cost=(1, 1)
    )
    hung, _ = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    hung.endpoint = Endpoint(
        api="openrouter", model="m/hung", provider="p", cost=(1, 1)
    )

    monkeypatch.setattr(monitor_mod, "load_phase2_results", lambda d: results)
    monkeypatch.setattr(monitor_mod.config.bi.monitor, "job_deadline_seconds", 1.0)
    report = _monitor_endpoints(monkeypatch, tmp_path, [done, hung], hangs={hung.slug})

    assert report.failures == []
    assert [(r.model, r.event) for r in report.rows] == [("m/hung", "deadline_cutoff")]

    # the finished endpoint's batch and state survived the cut-off
    batch = orjson.loads(
        (tmp_path / f"{slugify(str(done.endpoint))}.json").read_bytes()
    )
    a_bi = done.current_epoch.border_inputs[0]
    (samples,) = batch[a_bi].values()  # exactly one batch, the one just written
    assert [tok for _, tok in samples] == ["tok"]
    assert (monitor_mod.config.bi.state_dir / f"{done.slug}.json").exists()
    # the hung one persisted nothing: tomorrow's run repeats it from scratch
    assert not (monitor_mod.config.bi.state_dir / f"{hung.slug}.json").exists()


def test_reinit_timeout_is_not_reported_as_a_deadline_cutoff(tmp_path, monkeypatch):
    """The two deadlines are different events: one endpoint's re-init giving up must
    not read as the whole job running out of time."""
    state, results = open_state_from_fixture(
        "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8"
    )

    async def slow_reinit(*args, **kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(monitor_mod, "load_phase2_results", lambda d: results)
    monkeypatch.setattr(monitor_mod, "reinit", slow_reinit)
    monkeypatch.setattr(monitor_mod.config.bi.monitor, "reinit_timeout_seconds", 0.05)
    monkeypatch.setattr(monitor_mod.config.bi.monitor, "job_deadline_seconds", 30.0)
    report = _monitor_endpoints(monkeypatch, tmp_path, [state], hangs=set())

    assert report.failures == []
    assert [r.event for r in report.rows] == ["reinit_timeout"]


def test_job_deadline_cuts_a_hanging_endpoint_short(monkeypatch):
    """The job must end itself before the workflow's hard timeout: a kill there is
    reported as "cancelled", which skips the commit step and throws away every
    endpoint's saved batch."""
    state, _ = open_state_from_fixture("openai2fgpt-4o-mini23azure")

    async def hanging_run_endpoint(*args, **kwargs):
        await asyncio.sleep(30)
        raise AssertionError("should have been cut off by the job deadline")

    monkeypatch.setattr(monitor_mod.config.bi.monitor, "job_deadline_seconds", 0.05)
    report = _monitor_one_endpoint(monkeypatch, state, hanging_run_endpoint)

    assert report.failures == []
    assert [r.event for r in report.rows] == ["deadline_cutoff"]


def test_endpoints_left_after_the_deadline_are_not_started(monkeypatch):
    """Once the deadline has passed, queueing more work would only push the job
    further past it: report the endpoint instead of running it."""
    state, _ = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    started = []

    async def fake_run_endpoint(client, strategy, endpoint_state, *args, **kwargs):
        started.append(str(endpoint_state.endpoint))

    monkeypatch.setattr(monitor_mod.config.bi.monitor, "job_deadline_seconds", 0.0)
    report = _monitor_one_endpoint(monkeypatch, state, fake_run_endpoint)

    assert started == []
    assert report.failures == []
    assert [r.event for r in report.rows] == ["deadline_cutoff"]
