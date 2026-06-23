"""Tests: spend ledger entries written by run_endpoint (monitor + reinit buckets)."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import orjson

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi import monitor as monitor_mod
from trackllm_website.bi.common import PlainStrategy
from trackllm_website.bi.migrate_state import migrate_endpoint
from trackllm_website.bi.monitor import run_endpoint
from trackllm_website.config import Endpoint, config
from trackllm_website.spend import cumulative_by_kind, record_query

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


def _wire_common_patches(monkeypatch, tmp_path, monthly_path, results):
    monkeypatch.setattr(type(config), "spend_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(monitor_mod.config.bi, "data_dir", tmp_path)
    monkeypatch.setattr(monitor_mod, "get_output_path", lambda ep, ym: monthly_path)
    monkeypatch.setattr(monitor_mod, "load_phase2_results", lambda d: results)


def test_monitor_writes_monitor_spend(monkeypatch, tmp_path):
    state, results = _state_from_fixture("openai2fgpt-4o-mini23azure")
    epoch = state.current_epoch
    monthly_path = tmp_path / "monthly.json"
    daily_batch = {bi: [(NOW.isoformat(), "tok")] for bi in epoch.border_inputs}

    async def fake_sample_prompts(*args, **kwargs):
        record_query(0.05, False)
        return daily_batch, 0

    _wire_common_patches(monkeypatch, tmp_path, monthly_path, results)
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)

    async def go():
        async with OpenRouterClient() as client:
            await run_endpoint(client, PlainStrategy(), state, NOW)

    asyncio.run(go())

    by_kind = cumulative_by_kind(tmp_path)
    assert "monitor" in by_kind
    assert by_kind["monitor"] == 0.05
    assert "reinit" not in by_kind


def test_reinit_writes_both_spend_lines(monkeypatch, tmp_path):
    state, results = _state_from_fixture(
        "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8"
    )
    epoch = state.current_epoch
    monthly_path = tmp_path / "monthly.json"
    daily_batch = {bi: [(NOW.isoformat(), "tok")] for bi in epoch.border_inputs}

    async def fake_sample_prompts(*args, **kwargs):
        record_query(0.10, False)
        return daily_batch, 0

    async def fake_reinit(*args, **kwargs):
        from trackllm_website.bi.reinit import ReinitResult

        record_query(0.25, False)
        return ReinitResult(epoch=None, reason="no_bis")

    _wire_common_patches(monkeypatch, tmp_path, monthly_path, results)
    monkeypatch.setattr(monitor_mod, "sample_prompts", fake_sample_prompts)
    monkeypatch.setattr(monitor_mod, "reinit", fake_reinit)

    async def go():
        async with OpenRouterClient() as client:
            await run_endpoint(client, PlainStrategy(), state, NOW)

    asyncio.run(go())

    by_kind = cumulative_by_kind(tmp_path)
    assert by_kind.get("monitor", 0) == 0.10
    assert by_kind.get("reinit", 0) == 0.25
