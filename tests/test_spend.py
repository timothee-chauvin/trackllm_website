import asyncio
from datetime import datetime, timezone
from pathlib import Path

from trackllm_website.spend import (
    Spend,
    append_entry,
    cumulative_by_kind,
    record_query,
    track,
)


def test_record_query_noop_without_bucket():
    record_query(0.5, is_error=False)  # no active bucket → must not raise


def test_track_accumulates_cost_and_counts():
    with track() as s:
        record_query(0.10, is_error=False)
        record_query(0.0, is_error=True)
        record_query(0.05, is_error=False)
    assert abs(s.cost - 0.15) < 1e-9
    assert s.n_queries == 3
    assert s.n_errors == 1


def test_billed_error_still_counts_cost():
    # A billed-but-errored response (e.g. "No logprobs returned"): tokens were
    # generated and charged, so its cost must be counted, and it is an error.
    with track() as s:
        record_query(0.07, is_error=True)
    assert abs(s.cost - 0.07) < 1e-9
    assert s.n_queries == 1
    assert s.n_errors == 1


def test_track_propagates_into_child_tasks():
    async def child():
        record_query(0.02, is_error=False)

    async def run():
        with track() as s:
            await asyncio.gather(child(), child(), child())
        return s

    s = asyncio.run(run())
    assert s.n_queries == 3
    assert abs(s.cost - 0.06) < 1e-9


def test_partial_spend_survives_cancellation():
    async def slow():
        record_query(0.04, is_error=False)
        await asyncio.sleep(10)

    async def run():
        with track() as s:
            try:
                await asyncio.wait_for(slow(), timeout=0.05)
            except asyncio.TimeoutError:
                pass
        return s

    s = asyncio.run(run())
    assert s.cost == 0.04  # spend recorded before cancellation is retained


def test_append_and_cumulative_round_trip(tmp_path):
    now = datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc)
    append_entry(
        tmp_path, "slugA", "onboard", Spend(cost=1.0, n_queries=10, n_errors=1), now
    )
    append_entry(tmp_path, "slugA", "monitor", Spend(cost=0.5, n_queries=5), now)
    append_entry(tmp_path, "slugB", "monitor", Spend(cost=0.25, n_queries=2), now)
    f = tmp_path / "slugA" / "2026-06.jsonl"
    assert len(f.read_text().strip().splitlines()) == 2  # appended, not overwritten
    cum = cumulative_by_kind(tmp_path)
    assert abs(cum["onboard"] - 1.0) < 1e-9
    assert abs(cum["monitor"] - 0.75) < 1e-9


def test_append_entry_is_nonfatal_and_logs_on_write_error(tmp_path, caplog):
    # Make the per-slug dir creation fail by planting a FILE where the dir goes.
    (tmp_path / "slugX").write_text("not a directory")
    now = datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc)
    with caplog.at_level("ERROR"):
        # Must NOT raise: a ledger write failure can't abort the primary run.
        append_entry(tmp_path, "slugX", "monitor", Spend(cost=1.0, n_queries=1), now)
    assert "spend ledger write failed" in caplog.text  # logged loudly, not silent


def test_spend_dir_property(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    from trackllm_website.config import Config

    cfg = Config()
    assert cfg.spend_dir == cfg.data_dir / "spend" == Path("website/data/spend")
