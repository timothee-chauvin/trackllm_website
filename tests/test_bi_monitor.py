import asyncio
from datetime import datetime, timezone
from pathlib import Path

import orjson

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi import monitor as monitor_mod
from trackllm_website.bi.common import PlainStrategy
from trackllm_website.bi.migrate_state import migrate_endpoint
from trackllm_website.bi.monitor import Decision, decide, run_endpoint
from trackllm_website.config import Endpoint

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


def test_stable_endpoint_no_action():
    state, results = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    decision = decide(state, results, datetime(2026, 2, 15, tzinfo=timezone.utc))
    assert decision.action == "none"
    assert decision.unstable is False


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
        return None  # no new BIs -> retired with reason "no_bis"

    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)
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
