"""Tests: MonitorReport built from run_endpoint/monitor events."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import orjson

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi import monitor as monitor_mod
from trackllm_website.bi.common import PlainStrategy
from trackllm_website.bi.digest import MonitorReport, MonitorRow
from trackllm_website.bi.migrate_state import migrate_endpoint
from trackllm_website.bi.monitor import run_endpoint
from trackllm_website.config import Endpoint, config
from trackllm_website.spend import record_query

FIXTURES = Path("tests/fixtures/phase_2")
ENDPOINT = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))
NOW = datetime(2026, 2, 15, tzinfo=timezone.utc)


def _state_from_fixture(slug: str):
    results = orjson.loads((FIXTURES / slug / "data.json").read_bytes())
    state = migrate_endpoint(ENDPOINT, results)
    state.status = "monitoring"
    state.retired = None
    epoch = state.epochs[0]
    epoch.end = None
    epoch.end_reason = None
    return state, results


def _wire_patches(monkeypatch, tmp_path, monthly_path, results):
    monkeypatch.setattr(type(config), "spend_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(monitor_mod, "get_output_path", lambda ep, ym: monthly_path)
    monkeypatch.setattr(monitor_mod, "load_phase2_results", lambda d: results)


# --- run_endpoint tests ---


def test_reinit_with_epoch_appends_reonboarded_row(monkeypatch, tmp_path):
    state, results = _state_from_fixture(
        "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8"
    )
    epoch = state.current_epoch
    monthly_path = tmp_path / "monthly.json"
    daily_batch = {bi: [(NOW.isoformat(), "tok")] for bi in epoch.border_inputs}
    n_bis = len(epoch.border_inputs)

    async def fake_sample_prompts(*args, **kwargs):
        return daily_batch, 0

    async def fake_reinit(*args, **kwargs):
        from trackllm_website.bi.reinit import ReinitResult
        from trackllm_website.bi.state import Epoch

        new_epoch = Epoch(
            start=NOW,
            border_inputs=epoch.border_inputs[:],
            reference={},
        )
        record_query(0.30, False)
        return ReinitResult(epoch=new_epoch, reason="ok")

    _wire_patches(monkeypatch, tmp_path, monthly_path, results)
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)
    monkeypatch.setattr(monitor_mod, "reinit", fake_reinit)

    event_rows: list[MonitorRow] = []

    async def go():
        async with OpenRouterClient() as client:
            await run_endpoint(
                client, PlainStrategy(), state, NOW, event_rows=event_rows
            )

    asyncio.run(go())

    assert len(event_rows) == 1
    row = event_rows[0]
    assert row.event == "reonboarded"
    assert row.change_date == "2026-01-24"
    assert row.n_bis_after == n_bis
    assert row.model == ENDPOINT.model
    assert row.provider == ENDPOINT.provider


def test_reinit_no_epoch_appends_reonboard_no_bis_row(monkeypatch, tmp_path):
    state, results = _state_from_fixture(
        "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8"
    )
    epoch = state.current_epoch
    monthly_path = tmp_path / "monthly.json"
    daily_batch = {bi: [(NOW.isoformat(), "tok")] for bi in epoch.border_inputs}

    async def fake_sample_prompts(*args, **kwargs):
        return daily_batch, 0

    async def fake_reinit(*args, **kwargs):
        from trackllm_website.bi.reinit import ReinitResult

        record_query(0.20, False)
        return ReinitResult(epoch=None, reason="no_bis")

    _wire_patches(monkeypatch, tmp_path, monthly_path, results)
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)
    monkeypatch.setattr(monitor_mod, "reinit", fake_reinit)

    event_rows: list[MonitorRow] = []

    async def go():
        async with OpenRouterClient() as client:
            await run_endpoint(
                client, PlainStrategy(), state, NOW, event_rows=event_rows
            )

    asyncio.run(go())

    assert len(event_rows) == 1
    row = event_rows[0]
    assert row.event == "reonboard_no_bis"
    assert row.change_date == "2026-01-24"
    assert row.n_bis_after is None


def test_retire_stalled_appends_retired_stalled_row(monkeypatch, tmp_path):
    state, results = _state_from_fixture(
        "mistralai2fmistral-7b-instruct-v0.323together"
    )
    epoch = state.current_epoch
    monthly_path = tmp_path / "monthly.json"
    daily_batch = {bi: [(NOW.isoformat(), "tok")] for bi in epoch.border_inputs}

    async def fake_sample_prompts(*args, **kwargs):
        return daily_batch, 0

    now_stall = datetime(2026, 3, 10, tzinfo=timezone.utc)
    _wire_patches(monkeypatch, tmp_path, monthly_path, results)
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)

    event_rows: list[MonitorRow] = []

    async def go():
        async with OpenRouterClient() as client:
            await run_endpoint(
                client, PlainStrategy(), state, now_stall, event_rows=event_rows
            )

    asyncio.run(go())

    assert len(event_rows) == 1
    row = event_rows[0]
    assert row.event == "retired_stalled"
    assert row.change_date is None
    assert row.n_bis_after is None
    assert row.spent == 0.0


def test_action_none_appends_no_row(monkeypatch, tmp_path):
    state, results = _state_from_fixture("openai2fgpt-4o-mini23azure")
    epoch = state.current_epoch
    monthly_path = tmp_path / "monthly.json"
    daily_batch = {bi: [(NOW.isoformat(), "tok")] for bi in epoch.border_inputs}

    async def fake_sample_prompts(*args, **kwargs):
        return daily_batch, 0

    _wire_patches(monkeypatch, tmp_path, monthly_path, results)
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)

    event_rows: list[MonitorRow] = []

    async def go():
        async with OpenRouterClient() as client:
            await run_endpoint(
                client, PlainStrategy(), state, NOW, event_rows=event_rows
            )

    asyncio.run(go())

    assert event_rows == []


def test_event_rows_none_does_not_crash(monkeypatch, tmp_path):
    """Passing event_rows=None (default) should not crash."""
    state, results = _state_from_fixture("openai2fgpt-4o-mini23azure")
    epoch = state.current_epoch
    monthly_path = tmp_path / "monthly.json"
    daily_batch = {bi: [(NOW.isoformat(), "tok")] for bi in epoch.border_inputs}

    async def fake_sample_prompts(*args, **kwargs):
        return daily_batch, 0

    _wire_patches(monkeypatch, tmp_path, monthly_path, results)
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)

    async def go():
        async with OpenRouterClient() as client:
            await run_endpoint(client, PlainStrategy(), state, NOW)

    asyncio.run(go())  # must not raise


# --- monitor() integration test ---


def test_monitor_returns_report_with_n_endpoints(monkeypatch, tmp_path):
    """monitor() returns a MonitorReport with n_endpoints matching monitoring states."""
    from trackllm_website.bi.monitor import monitor

    state, results = _state_from_fixture("openai2fgpt-4o-mini23azure")

    monkeypatch.setattr(type(config), "spend_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(monitor_mod, "load_all_states", lambda d: {"ep": state})

    async def fake_resolve_strategies(client, endpoints, probe_spend=None):
        return {str(state.endpoint): PlainStrategy()}, []

    async def fake_run_endpoint(
        client, strategy, s, now, probe_spend=None, event_rows=None
    ):
        pass  # no events → empty rows

    monkeypatch.setattr(monitor_mod, "resolve_strategies", fake_resolve_strategies)
    monkeypatch.setattr(monitor_mod, "run_endpoint", fake_run_endpoint)
    # Prevent actual client connections
    monkeypatch.setattr(
        monitor_mod,
        "gather_with_concurrency",
        lambda n, *coros: asyncio.gather(*coros),
    )

    report = asyncio.run(monitor())

    assert isinstance(report, MonitorReport)
    assert report.n_endpoints == 1
    assert report.rows == []
    assert report.date == datetime.now(tz=timezone.utc).date().isoformat()
