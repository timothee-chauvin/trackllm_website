# PR3 — Universal Spend Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record actual `$` spent for every request (LT and B3IT) into an append-only per-endpoint JSONL ledger at `website/data/spend/`, so PR4's B3IT digest emails can report real spend.

**Architecture:** A `Spend` accumulator held in a `contextvars.ContextVar` (new `spend.py`). `OpenRouterClient.query` adds each response's cost + counts to the active bucket if one is set (no-op otherwise). Callers open a per-endpoint/activity bucket with a `track()` context manager, then write one ledger line via `append_entry`. Context-scoping means the hook captures every query path (including `phase_1a`'s own client) and survives the 3h-timeout cancellation (the bucket is owned by the caller, not the cancelled coroutine). LT's `main.py` already aggregates per-endpoint cost, so it writes `kind=lt` lines directly without the contextvar.

**Tech Stack:** Python 3.13, pydantic, contextvars, asyncio, orjson, pytest, uv.

## Global Constraints

- Ledger path: `website/data/spend/{slug}/{YYYY-MM}.jsonl`; `config.spend_dir = config.data_dir / "spend"` (a new property; `config.data_dir` stays `website/data`).
- `slug = slugify(f"{endpoint.model}#{endpoint.provider}")` (the existing scheme — reuse `trackllm_website.util.slugify`).
- One JSON object per line: `{"timestamp": <iso8601 str>, "kind": <str>, "cost": <float>, "n_queries": <int>, "n_errors": <int>}`.
- `kind ∈ {"lt","vetting","onboard","recheck","reinit","monitor"}`.
- The `query` hook MUST be a no-op when no bucket is active (LT main path writes its own lines; tests and ad-hoc scripts must be unaffected).
- Append, never rewrite: open files in append mode and write one line; do not read-modify-write the whole file.
- Use `prek run --files <changed .py>` before each commit, then `git commit --no-verify`.
- Tests: `uv run pytest`. `config` is module-level `Config()` at import → set `OPENROUTER_API_KEY` (monkeypatch or env).
- Do not add silent error handling that hides failures (project rule).

---

### Task 1: `spend.py` core + `config.spend_dir`

**Files:**
- Create: `src/trackllm_website/spend.py`
- Modify: `src/trackllm_website/config.py` (add `spend_dir` property to `Config`, next to `lt_dir`)
- Test: `tests/test_spend.py` (create)

**Interfaces produced (used by later tasks):**
- `class Spend` — mutable accumulator with fields `cost: float = 0.0`, `n_queries: int = 0`, `n_errors: int = 0`.
- `record_query(cost: float, is_error: bool) -> None` — adds to the active bucket if set, else no-op.
- `track() -> contextmanager` yielding a fresh `Spend`, set as the active bucket for the `with` body (and inherited by child asyncio tasks created within it).
- `class SpendEntry(BaseModel)` — `timestamp: datetime`, `kind: str`, `cost: float`, `n_queries: int`, `n_errors: int`.
- `append_entry(spend_dir: Path, slug: str, kind: str, spend: Spend, now: datetime) -> None` — appends one JSONL line to `spend_dir/{slug}/{YYYY-MM}.jsonl` (creating dirs), `YYYY-MM` from `now`.
- `cumulative_by_kind(spend_dir: Path) -> dict[str, float]` — sums `cost` across all `spend_dir/**/*.jsonl` grouped by `kind`.
- `config.spend_dir -> Path` (= `config.data_dir / "spend"`).

- [ ] **Step 1: Write failing tests**

Create `tests/test_spend.py`:

```python
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from trackllm_website.spend import (
    Spend, append_entry, cumulative_by_kind, record_query, track,
)


def test_record_query_noop_without_bucket():
    record_query(0.5, is_error=False)  # no active bucket → must not raise


def test_track_accumulates_cost_and_counts():
    with track() as s:
        record_query(0.10, is_error=False)
        record_query(0.0, is_error=True)
        record_query(0.05, is_error=False)
    assert s.cost == 0.15
    assert s.n_queries == 3
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
    append_entry(tmp_path, "slugA", "onboard", Spend(cost=1.0, n_queries=10, n_errors=1), now)
    append_entry(tmp_path, "slugA", "monitor", Spend(cost=0.5, n_queries=5), now)
    append_entry(tmp_path, "slugB", "monitor", Spend(cost=0.25, n_queries=2), now)
    f = tmp_path / "slugA" / "2026-06.jsonl"
    assert len(f.read_text().strip().splitlines()) == 2  # appended, not overwritten
    cum = cumulative_by_kind(tmp_path)
    assert abs(cum["onboard"] - 1.0) < 1e-9
    assert abs(cum["monitor"] - 0.75) < 1e-9


def test_spend_dir_property(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    from trackllm_website.config import Config
    cfg = Config()
    assert cfg.spend_dir == cfg.data_dir / "spend" == Path("website/data/spend")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `OPENROUTER_API_KEY=dummy uv run pytest tests/test_spend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trackllm_website.spend'`.

- [ ] **Step 3: Implement `spend.py`**

Create `src/trackllm_website/spend.py`:

```python
"""Actual-spend ledger: a context-scoped accumulator and per-endpoint JSONL store.

The accumulator lives in a ContextVar so OpenRouterClient.query can add each
response's cost without threading a parameter through every call path, and so a
cancelled (timed-out) coroutine still leaves its partial spend in the caller-owned
bucket. The ledger records only facts (money spent), one line per logical activity.
"""

import contextvars
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import orjson
from pydantic import BaseModel


@dataclass
class Spend:
    cost: float = 0.0
    n_queries: int = 0
    n_errors: int = 0


_active: contextvars.ContextVar[Spend | None] = contextvars.ContextVar(
    "active_spend", default=None
)


def record_query(cost: float, is_error: bool) -> None:
    """Add one query's outcome to the active bucket, if any (else no-op).

    cost is always added (it is the actually-billed amount: 0.0 for true errors,
    non-zero for billed-but-errored responses like "No logprobs returned");
    n_errors counts error responses separately.
    """
    bucket = _active.get()
    if bucket is None:
        return
    bucket.n_queries += 1
    bucket.cost += cost
    if is_error:
        bucket.n_errors += 1


@contextmanager
def track() -> Iterator[Spend]:
    """Open a fresh Spend bucket as the active accumulator for the with-body.

    Child asyncio tasks created within inherit it (ContextVar copy-on-task).
    The yielded Spend stays readable after the block, including after a caught
    cancellation/timeout inside it.
    """
    bucket = Spend()
    token = _active.set(bucket)
    try:
        yield bucket
    finally:
        _active.reset(token)


class SpendEntry(BaseModel):
    timestamp: datetime
    kind: str
    cost: float
    n_queries: int
    n_errors: int


def append_entry(
    spend_dir: Path, slug: str, kind: str, spend: Spend, now: datetime
) -> None:
    entry = SpendEntry(
        timestamp=now,
        kind=kind,
        cost=spend.cost,
        n_queries=spend.n_queries,
        n_errors=spend.n_errors,
    )
    path = spend_dir / slug / f"{now:%Y-%m}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write(orjson.dumps(entry.model_dump(mode="json")) + b"\n")


def cumulative_by_kind(spend_dir: Path) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    if not spend_dir.exists():
        return dict(totals)
    for f in spend_dir.glob("*/*.jsonl"):
        for line in f.read_bytes().splitlines():
            if not line.strip():
                continue
            rec = orjson.loads(line)
            totals[rec["kind"]] += rec["cost"]
    return dict(totals)
```

In `src/trackllm_website/config.py`, add to `class Config` next to `lt_dir`:

```python
    @property
    def spend_dir(self) -> Path:
        return self.data_dir / "spend"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `OPENROUTER_API_KEY=dummy uv run pytest tests/test_spend.py -v`
Expected: PASS (all 6).

- [ ] **Step 5: Commit**

```bash
prek run --files src/trackllm_website/spend.py src/trackllm_website/config.py tests/test_spend.py >/dev/null 2>&1
git add src/trackllm_website/spend.py src/trackllm_website/config.py tests/test_spend.py
git commit --no-verify -m "feat(spend): context-scoped spend accumulator + JSONL ledger + config.spend_dir"
```

---

### Task 2: Hook `OpenRouterClient.query` to the active bucket

**Files:**
- Modify: `src/trackllm_website/api.py` (`query`, around lines 183-235)
- Test: `tests/test_spend_api_hook.py` (create)

**Interfaces:**
- Consumes: `spend.record_query` (Task 1).
- Produces: every `OpenRouterClient.query` call records its result into the active bucket (cost on success, error count on error), once per call.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spend_api_hook.py` (mock the network at `_make_request` so no real calls):

```python
import asyncio

from trackllm_website.api import OpenRouterClient
from trackllm_website.config import Endpoint
from trackllm_website.spend import track
from trackllm_website.storage import Response, ResponseError


def _ep():
    return Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))


def _resp(cost, error=None):
    return Response(
        date=__import__("datetime").datetime.now(tz=__import__("datetime").timezone.utc),
        endpoint=_ep(), prompt="x", content="y", logprobs=None, cost=cost,
        input_tokens=1, output_tokens=1, reasoning_tokens=0, reasoning_content=None,
        generation_id="g", error=error,
    )


def test_query_records_cost_and_errors_into_bucket(monkeypatch):
    seq = [_resp(0.10), _resp(0.0, ResponseError(http_code=500, message="boom")), _resp(0.20)]

    async def fake_make_request(self, *a, **k):
        return seq.pop(0)

    monkeypatch.setattr(OpenRouterClient, "_make_request", fake_make_request)

    async def run():
        client = OpenRouterClient()
        try:
            with track() as s:
                await asyncio.gather(
                    client.query(_ep(), "x"), client.query(_ep(), "x"), client.query(_ep(), "x")
                )
            return s
        finally:
            await client.close()

    s = asyncio.run(run())
    assert s.n_queries == 3
    assert s.n_errors == 1
    assert abs(s.cost - 0.30) < 1e-9


def test_query_noop_without_bucket(monkeypatch):
    async def fake_make_request(self, *a, **k):
        return _resp(0.10)
    monkeypatch.setattr(OpenRouterClient, "_make_request", fake_make_request)

    async def run():
        client = OpenRouterClient()
        try:
            return await client.query(_ep(), "x")  # no bucket → must not raise
        finally:
            await client.close()

    assert asyncio.run(run()).cost == 0.10
```

(If `_make_request`'s real signature differs, match it; the point is to bypass the network and return canned `Response`s. If `query` retries on error responses, set the error `Response` to a terminal/non-retried form so the sequence is consumed once per `query` — inspect `query`/`retry_with_exponential_backoff` and adjust the canned responses so each `query()` consumes exactly one.)

- [ ] **Step 2: Run test to verify it fails**

Run: `OPENROUTER_API_KEY=dummy uv run pytest tests/test_spend_api_hook.py -v`
Expected: FAIL — bucket stays at 0 (no hook yet).

- [ ] **Step 3: Add the hook in `query`**

In `src/trackllm_website/api.py`, `import` at top: `from trackllm_website.spend import record_query`. In `query`, capture the result and record once before returning. Replace the structure so both the success path and the `except` error-Response path funnel through a single record+return. Concretely, wrap the existing body:

```python
    async def query(self, endpoint, prompt, ...):
        response = await self._query_inner(endpoint, prompt, ...)  # existing body, refactored to return
        record_query(response.cost, response.error is not None)
        return response
```

If refactoring to `_query_inner` is awkward, instead call `record_query(resp.cost, resp.error is not None)` immediately before EACH `return Response(...)`/`return await ...` in `query` — but ensure it runs exactly once per `query` call (not per retry). The retries happen inside `retry_with_exponential_backoff`/`_make_request`; `query` returns once, so recording at `query`'s return point(s) is correct. Do NOT record inside `_make_request` (that would count each retry).

- [ ] **Step 4: Run test to verify it passes**

Run: `OPENROUTER_API_KEY=dummy uv run pytest tests/test_spend_api_hook.py -v`
Expected: PASS.

- [ ] **Step 5: Run the api/sampling tests for regressions**

Run: `OPENROUTER_API_KEY=dummy uv run pytest tests/test_bi_sampling.py -q`
Expected: PASS (sampling still works; hook is additive).

- [ ] **Step 6: Commit**

```bash
prek run --files src/trackllm_website/api.py tests/test_spend_api_hook.py >/dev/null 2>&1
git add src/trackllm_website/api.py tests/test_spend_api_hook.py
git commit --no-verify -m "feat(spend): record each OpenRouterClient.query into the active spend bucket"
```

---

### Task 3: B3IT capture in `update_endpoints.py` (vetting + onboard/recheck)

**Files:**
- Modify: `src/trackllm_website/update_endpoints.py` (`update_endpoints_bi` around `vet_one`; `update_endpoints_bi_lifecycle` around `onboard_one`)
- Test: `tests/test_spend_lifecycle.py` (create)

**Interfaces:**
- Consumes: `track`, `append_entry` (Task 1); the `query` hook (Task 2); `config.spend_dir`.
- Produces: a `kind="vetting"` line per vetted endpoint (or one aggregate per run — see below) and a `kind="onboard"`/`kind="recheck"` line per onboard/recheck attempt.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spend_lifecycle.py`. Mock `reinit` to consume the active bucket (simulating queries) and assert a ledger line lands under `config.spend_dir` with the right kind. Example for the onboard path:

```python
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import trackllm_website.update_endpoints as ue
from trackllm_website.config import Endpoint, config
from trackllm_website.spend import record_query, cumulative_by_kind


def test_onboard_writes_onboard_spend(monkeypatch, tmp_path):
    monkeypatch.setattr(type(config), "spend_dir", property(lambda self: tmp_path))
    ep = Endpoint(api="openrouter", model="m/new", provider="p", cost=(1, 1), cost_per_request=0.001)

    async def fake_reinit(client, strategy, endpoint, old_bis, now):
        record_query(0.03, is_error=False)  # simulate billable onboarding work
        from trackllm_website.bi.reinit import ReinitResult
        from trackllm_website.bi.state import Epoch
        return ReinitResult(epoch=Epoch(start=now, border_inputs=["a"]*10, reference={}), reason="ok")

    monkeypatch.setattr(ue, "reinit", fake_reinit)
    # ... drive update_endpoints_bi_lifecycle with a single selected candidate `ep`,
    #     stubbing resolve_strategies/selection so only the onboard path runs ...
    # After the run:
    cum = cumulative_by_kind(tmp_path)
    assert cum.get("onboard", 0) > 0
```

Fill in the stubs to exercise exactly the onboard path (mirror the existing lifecycle tests in `tests/test_bi_lifecycle.py` for how to stub `resolve_strategies`, selection, and state I/O). Add an analogous assertion for the `recheck` kind and the `vetting` kind.

- [ ] **Step 2: Run test to verify it fails** (`...spend not recorded...`).

Run: `OPENROUTER_API_KEY=dummy uv run pytest tests/test_spend_lifecycle.py -v` → FAIL.

- [ ] **Step 3: Wrap the activities**

In `update_endpoints.py`:
- In `onboard_one`, wrap the `reinit` call (and the timeout handling) in `with track() as spend:`, and after it (in all outcome branches, including the `except asyncio.TimeoutError`) `append_entry(config.spend_dir, slug, kind, spend, now)` where `kind = "recheck" if is_recheck else "onboard"`. The bucket is opened before `asyncio.wait_for` so partial spend on timeout is still written. `slug = slugify(f"{endpoint.model}#{endpoint.provider}")` (already computed in `onboard_one`).
- In `vet_one` (inside `update_endpoints_bi`), wrap the `vet_endpoint` call in `with track() as spend:` and `append_entry(config.spend_dir, slugify(...), "vetting", spend, now)`. Use a single `now = datetime.now(tz=timezone.utc).replace(microsecond=0)` per run (the lifecycle function already computes one; `update_endpoints_bi` should compute its own).

- [ ] **Step 4: Run test to verify it passes** → PASS.

- [ ] **Step 5: Regression** — `OPENROUTER_API_KEY=dummy uv run pytest tests/test_bi_lifecycle.py tests/test_bi_vetting.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
prek run --files src/trackllm_website/update_endpoints.py tests/test_spend_lifecycle.py >/dev/null 2>&1
git add src/trackllm_website/update_endpoints.py tests/test_spend_lifecycle.py
git commit --no-verify -m "feat(spend): record vetting/onboard/recheck spend in update_endpoints"
```

---

### Task 4: B3IT capture in `monitor.py` (daily sample + change reinit)

**Files:**
- Modify: `src/trackllm_website/bi/monitor.py` (`run_endpoint`)
- Test: `tests/test_spend_monitor.py` (create)

**Interfaces:**
- Consumes: `track`, `append_entry`, `config.spend_dir`.
- Produces: a `kind="monitor"` line per monitored endpoint per run; a separate `kind="reinit"` line when a change triggers `reinit`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spend_monitor.py`. Mock `sample_prompts` (daily sample) and `reinit` to consume the bucket; assert `monitor` (and, on a forced change, `reinit`) lines land under `config.spend_dir`. Mirror `tests/test_bi_monitor.py` stubbing. Example skeleton:

```python
def test_monitor_writes_monitor_spend(monkeypatch, tmp_path):
    monkeypatch.setattr(type(config), "spend_dir", property(lambda self: tmp_path))
    # stub sample_prompts to record_query a cost; force decide()->action="none"
    # run run_endpoint for one monitoring endpoint
    # assert cumulative_by_kind(tmp_path)["monitor"] > 0
```

Add a second test forcing `decide()` to return `action="reinit"` and asserting a `reinit` line is written with its own bucket.

- [ ] **Step 2: Run test to verify it fails** → FAIL.

- [ ] **Step 3: Wrap the activities in `run_endpoint`**

In `bi/monitor.py::run_endpoint`:
- Wrap the daily `sample_prompts(...)` call in `with track() as monitor_spend:`, and after the sampling+save, `append_entry(config.spend_dir, state.slug, "monitor", monitor_spend, now)`.
- When `decision.action == "reinit"`, wrap the `reinit(...)` call in a SEPARATE `with track() as reinit_spend:` and `append_entry(config.spend_dir, state.slug, "reinit", reinit_spend, now)` after it. `state.slug` already exists. Keep the two buckets distinct so monitoring vs re-onboard spend are attributed separately.

- [ ] **Step 4: Run test to verify it passes** → PASS.

- [ ] **Step 5: Regression** — `OPENROUTER_API_KEY=dummy uv run pytest tests/test_bi_monitor.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
prek run --files src/trackllm_website/bi/monitor.py tests/test_spend_monitor.py >/dev/null 2>&1
git add src/trackllm_website/bi/monitor.py tests/test_spend_monitor.py
git commit --no-verify -m "feat(spend): record monitor + change-reinit spend in bi monitor"
```

---

### Task 5: LT spend from `main.py` summary

**Files:**
- Modify: `src/trackllm_website/main.py` (after the summary block)
- Test: `tests/test_spend_lt.py` (create)

**Interfaces:**
- Consumes: `append_entry`, `Spend`, `config.spend_dir`, the existing `get_summary` output (`{key: {success, error, total_cost}}`, key = `f"{model}#{provider}"`).
- Produces: one `kind="lt"` line per endpoint per run, written at run end from the summary (no contextvar needed).

- [ ] **Step 1: Write the failing test**

Create `tests/test_spend_lt.py`. Test a small extracted helper `write_lt_spend(summary, now)` that iterates the summary and calls `append_entry` with `kind="lt"`, `Spend(cost=total_cost, n_queries=success+error, n_errors=error)`, slug = `slugify(key)`:

```python
def test_write_lt_spend(monkeypatch, tmp_path):
    monkeypatch.setattr(type(config), "spend_dir", property(lambda self: tmp_path))
    from trackllm_website.main import write_lt_spend
    summary = {"m/a#p": {"success": 9, "error": 1, "total_cost": 0.12}}
    write_lt_spend(summary, datetime(2026, 6, 22, tzinfo=timezone.utc))
    assert abs(cumulative_by_kind(tmp_path)["lt"] - 0.12) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails** (`write_lt_spend` undefined) → FAIL.

- [ ] **Step 3: Add `write_lt_spend` and call it**

In `main.py`, add:

```python
def write_lt_spend(summary: dict, now: datetime) -> None:
    from trackllm_website.spend import Spend, append_entry
    from trackllm_website.util import slugify
    for key, s in summary.items():
        spend = Spend(cost=s["total_cost"], n_queries=s["success"] + s["error"], n_errors=s["error"])
        append_entry(config.spend_dir, slugify(key), "lt", spend, now)
```

Call it from `main()` right after `summary = get_summary(all_responses)`, with `now = datetime.now(tz=timezone.utc).replace(microsecond=0)`. Import `datetime`/`timezone` and `config` as needed (check existing imports in `main.py`; it currently builds `config = Config()` locally — use that instance's `spend_dir`, or import the module-level `config`). Keep `get_summary` unchanged.

- [ ] **Step 4: Run test to verify it passes** → PASS.

- [ ] **Step 5: Commit**

```bash
prek run --files src/trackllm_website/main.py tests/test_spend_lt.py >/dev/null 2>&1
git add src/trackllm_website/main.py tests/test_spend_lt.py
git commit --no-verify -m "feat(spend): write per-endpoint LT spend lines at run end"
```

---

### Task 6: Clean up `bi_independence.py` default-arg (flagged in PR2 reviews)

**Files:**
- Modify: `src/trackllm_website/bi/bi_independence.py` (the `main()`/CLI default `data_dir` arg, ~line 74-79)

**Interfaces:** none new — internal cleanup so the path derives from `config.bi.data_dir` rather than a hardcoded `"website/data/b3it/bi_prevalence/T=0"` literal.

- [ ] **Step 1: Read the function**

Run: `rg -n "def main|data_dir|bi_prevalence|T=0" src/trackllm_website/bi/bi_independence.py`
Read the function with the hardcoded default to understand its callers/signature.

- [ ] **Step 2: Derive the default from config**

Change the signature so the default is sentinel `None` and resolved inside from config, e.g.:

```python
def main(data_dir: str | None = None, ...):
    if data_dir is None:
        data_dir = str(config.bi.data_dir / "bi_prevalence" / "T=0")
    ...
```

Ensure `config` is imported (it is used elsewhere in `bi/` modules: `from trackllm_website.config import config`). Keep behavior identical when a path is passed explicitly.

- [ ] **Step 3: Verify it imports/runs and tests pass**

Run: `OPENROUTER_API_KEY=dummy uv run pytest -q` (full suite)
Expected: PASS. Also `OPENROUTER_API_KEY=dummy uv run python -c "import trackllm_website.bi.bi_independence"` → no error.

- [ ] **Step 4: Commit**

```bash
prek run --files src/trackllm_website/bi/bi_independence.py >/dev/null 2>&1
git add src/trackllm_website/bi/bi_independence.py
git commit --no-verify -m "refactor(b3it): derive bi_independence default path from config.bi.data_dir"
```

---

## Self-Review

**Spec coverage (PR 3 of the umbrella spec):**
- Ledger storage + `config.spend_dir` → Task 1. ✓
- ContextVar accumulator, no-op without bucket, child-task propagation, timeout survival → Task 1 tests + Task 2 hook. ✓
- `query` hook capturing all paths (incl. `phase_1a`'s own client — context-scoped, so covered) → Task 2. ✓
- Wiring: vetting/onboard/recheck (Task 3), monitor/reinit (Task 4), LT (Task 5). ✓
- `kind` taxonomy {lt,vetting,onboard,recheck,reinit,monitor} → Tasks 3/4/5. ✓
- Commit paths already covered by existing `git add` (no workflow change). ✓
- `cumulative_by_kind` for PR4 → Task 1. ✓
- `bi_independence.py` cleanup (reviewer follow-up) → Task 6. ✓

**Placeholder scan:** Tasks 3 and 4 reference "mirror existing lifecycle/monitor test stubbing" rather than inlining full stubs — this is a deliberate pointer to `tests/test_bi_lifecycle.py` / `tests/test_bi_monitor.py` because the exact stubs depend on those files' current shape; the assertions (which `kind` lands, that cumulative > 0) are concrete. All code steps that introduce new prod code contain complete code.

**Type consistency:** `Spend` (dataclass) and `SpendEntry` (pydantic) are distinct by design — `Spend` is the mutable accumulator, `SpendEntry` the serialized record built by `append_entry`. `record_query(cost, is_error)`, `track()`, `append_entry(spend_dir, slug, kind, spend, now)`, `cumulative_by_kind(spend_dir)`, and `config.spend_dir` names are used identically across Tasks 1-5.
