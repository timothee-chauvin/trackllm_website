import asyncio
from datetime import datetime, timezone
from pathlib import Path

import orjson
import pytest

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


def test_reinit_timeout_is_a_reported_failure(tmp_path, monkeypatch):
    """A hanging change-triggered re-init (full discovery is ~15k queries) must not
    stall the daily monitor job: it is bounded, reported, and persists nothing."""
    state, results = open_state_from_fixture(
        "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8"
    )
    epoch = state.current_epoch
    now = datetime(2026, 2, 15, tzinfo=timezone.utc)

    async def fake_sample_prompts(*args, **kwargs):
        return {bi: [(now.isoformat(), "tok")] for bi in epoch.border_inputs}, 0

    async def slow_reinit(*args, **kwargs):
        await asyncio.sleep(10)
        raise AssertionError("re-init should have been cancelled by the timeout")

    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(
        type(monitor_mod.config), "spend_dir", property(lambda self: tmp_path / "spend")
    )
    state_dir = monitor_mod.config.bi.state_dir
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)
    monkeypatch.setattr(monitor_mod, "reinit", slow_reinit)
    monkeypatch.setattr(
        monitor_mod, "get_output_path", lambda ep, ym: tmp_path / "monthly.json"
    )
    monkeypatch.setattr(monitor_mod, "load_phase2_results", lambda d: results)
    monkeypatch.setattr(monitor_mod.config.bi.monitor, "reinit_timeout_seconds", 0.05)

    event_rows = []

    async def go():
        async with OpenRouterClient() as client:
            await run_endpoint(
                client, PlainStrategy(), state, now, event_rows=event_rows
            )

    with pytest.raises(TimeoutError):  # run_isolated turns this into a report failure
        asyncio.run(go())

    assert any(r.event == "reinit_timeout" for r in event_rows)
    # nothing persisted: the next daily run re-detects the change and retries
    assert not (state_dir / f"{state.slug}.json").exists()
    # the spend of the abandoned re-init is still recorded
    assert (tmp_path / "spend").exists()


class FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    async def close(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def test_unresolved_strategy_is_a_reported_failure(monkeypatch):
    """An endpoint resolve_strategies drops must not be silently skipped: no batch
    is written, so the stall detector never advances and it would idle forever."""
    state, _ = open_state_from_fixture("openai2fgpt-4o-mini23azure")

    async def fake_resolve(client, endpoints, **kwargs):
        return {}, {str(state.endpoint): ["plain: 429 rate limited"]}

    monkeypatch.setattr(monitor_mod, "load_all_states", lambda d: {state.slug: state})
    monkeypatch.setattr(monitor_mod, "resolve_strategies", fake_resolve)
    monkeypatch.setattr(monitor_mod, "OpenRouterClient", FakeClient)

    report = asyncio.run(monitor())
    assert report.failures == [str(state.endpoint)]


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

    assert report.failures == [str(hung.endpoint)]
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

    assert report.failures == [str(state.endpoint)]
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

    assert report.failures == [str(state.endpoint)]
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
    assert report.failures == [str(state.endpoint)]
    assert [r.event for r in report.rows] == ["deadline_cutoff"]
