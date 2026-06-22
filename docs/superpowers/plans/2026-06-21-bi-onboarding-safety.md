# BI Onboarding Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound a single endpoint's BI onboarding so a dead/slow endpoint can't burn the full 15,000-query budget or stall a run for days; today the worst case is ~140h with no deadline.

**Architecture:** Onboarding (`reinit()` → phase-1 border search) and monitoring are **separate code paths**: onboarding uses `bi/common.py`'s `EndpointState`/`query_single` (via `Phase1EndpointState`) configured by `config.bi.phase_1`; monitoring uses its **own** `EndpointState` in `bi/phase_2.py` (config `bi.phase_2`, already has `abandon_this_run_after`). So the guards go entirely in the onboarding path — per-query knobs (`max_retries`, `abandon_after_timeouts`) live in **`[bi.phase_1]`**, the per-endpoint deadline in **`[bi.reinit]`** (sibling of `onboard_concurrency`). Monitoring is untouched, so Decision A holds by construction. There is no "init" vs "reinit": `reinit()` is the single onboarding primitive (fresh state, or `old_bis=[]` for rechecks); both run phase 1. Onboarding already resumes from on-disk partial results (`EndpointState.__post_init__` loads `_temp_results`); the deadline must preserve that, never restart from scratch.

**Tech Stack:** Python 3.13, asyncio, pydantic-settings, pytest, beartype, `uv`.

## Global Constraints
- Changes touch **only the onboarding path** — `bi/common.py` `query_single`/`run_queries` + `Phase1EndpointState` construction in `bi/phase_1.py`. Do **not** touch `bi/phase_2.py`/`bi/monitor.py` (monitoring is a separate `EndpointState`/loop with `config.bi.phase_2`, already 8-retry and with its own `abandon_this_run_after`). This is how Decision A is satisfied — by construction, not by per-state flags.
- New per-query tunables live in `config.toml` `[bi.phase_1]` (`Phase1Config`); the deadline lives in `[bi.reinit]` (`ReinitConfig`). Read values from config — single source of truth.
- `EndpointState` (common.py) is also constructed by analysis tools (`bi_prevalence.py`, `hardware_noise.py`, `logprob_stats.py`). New `EndpointState` fields must default to **current behavior** (retries = `config.api.max_retries`, `backoff_on_timeout=True`, timeout-abandon disabled) so only the phase-1 onboarding constructor opts into the safety values; those tools and tests stay unaffected.
- Abandon condition is **all-timeouts**: ≥ `abandon_after_timeouts` completed requests, every one a timeout, zero successes. (Decision B.)
- `onboard_concurrency`: 20 → **40**. (Decision C.)
- Resume, never restart: an endpoint short of its BI target must continue from saved partial results next run until enough BIs *or* the query budget. (Decision D.)
- TDD; run `prek run --all-files` before each commit (use `git commit --no-verify` — the repo's stale `pre-commit` hook shells to an uninstalled binary; `prek` is run manually). Tests run with `OPENROUTER_API_KEY=dummy uv run pytest`. New fakes that stand in for `OpenRouterClient` must subclass it (package-wide beartype enforces the type).

---

### Task 1: Config knobs

**Files:**
- Modify: `config.toml` (`[bi.phase_1]` for per-query knobs; `[bi.reinit]` for the deadline + concurrency)
- Modify: `src/trackllm_website/config.py` (`Phase1Config`, `ReinitConfig`)
- Test: `tests/test_bi_onboarding_safety.py` (new)

**Interfaces:**
- Produces: `config.bi.phase_1.max_retries: int` (=3 → 4 attempts), `config.bi.phase_1.abandon_after_timeouts: int` (=20); `config.bi.reinit.onboard_timeout_seconds: int` (=10800); `config.bi.reinit.onboard_concurrency` changed 20 → 40.

- [ ] **Step 1: Failing test** — `tests/test_bi_onboarding_safety.py`:
```python
from trackllm_website.config import config
def test_onboarding_safety_knobs_present():
    assert config.bi.phase_1.max_retries == 3            # 4 attempts (onboarding)
    assert config.bi.phase_1.abandon_after_timeouts == 20
    assert config.bi.reinit.onboard_timeout_seconds == 10800  # 3h
    assert config.bi.reinit.onboard_concurrency == 40
```
- [ ] **Step 2:** `OPENROUTER_API_KEY=dummy uv run pytest tests/test_bi_onboarding_safety.py::test_onboarding_safety_knobs_present -v` → FAIL.
- [ ] **Step 3:** In `config.py` add to `Phase1Config`: `max_retries: int`, `abandon_after_timeouts: int`; add to `ReinitConfig`: `onboard_timeout_seconds: int`. In `config.toml` `[bi.phase_1]` add `max_retries = 3`, `abandon_after_timeouts = 20`; in `[bi.reinit]` add `onboard_timeout_seconds = 10800` and change `onboard_concurrency = 20` → `40`.
- [ ] **Step 4:** rerun → PASS.
- [ ] **Step 5:** `prek run --all-files`; commit `feat(bi): onboarding-safety config knobs`.

---

### Task 2: Per-query retry override + skip backoff on timeout (BI sampling)

**Files:**
- Modify: `src/trackllm_website/api.py` (`OpenRouterClient.query`, `retry_with_exponential_backoff`)
- Modify: `src/trackllm_website/bi/common.py` (`EndpointState` fields; `query_single`)
- Test: `tests/test_bi_onboarding_safety.py`

**Interfaces:**
- `retry_with_exponential_backoff(..., backoff_on_timeout: bool = True)`: when `False`, a retryable `asyncio.TimeoutError` retries with **zero** sleep (it already cost `config.api.timeout`s).
- `OpenRouterClient.query(..., max_retries: int | None = None, backoff_on_timeout: bool = True)` threads both to the retry helper.
- `EndpointState` gains `max_retries: int | None = None` (None → `client.query` falls back to `config.api.max_retries`, preserving current behavior) and `backoff_on_timeout: bool = True`. Phase-1 overrides these in Task 4.
- `query_single` passes `max_retries=state.max_retries, backoff_on_timeout=state.backoff_on_timeout`.

- [ ] **Step 1: Failing test** — backoff helper honors the flag (no real network):
```python
import asyncio, time
from trackllm_website.api import retry_with_exponential_backoff
def test_no_backoff_on_timeout_when_disabled():
    calls = {"n": 0}
    async def always_timeout():
        calls["n"] += 1
        raise asyncio.TimeoutError()
    async def run():
        t0 = time.monotonic()
        try:
            await retry_with_exponential_backoff(
                always_timeout, max_retries=3, backoff_on_timeout=False)
        except asyncio.TimeoutError:
            pass
        return time.monotonic() - t0, calls["n"]
    elapsed, n = asyncio.run(run())
    assert n == 4                # 1 + 3 retries
    assert elapsed < 0.5         # zero backoff sleeps
```
- [ ] **Step 2:** run → FAIL (`backoff_on_timeout` unexpected kwarg).
- [ ] **Step 3:** In `retry_with_exponential_backoff` add `backoff_on_timeout: bool = True`; in the retry branch, if the exception is `asyncio.TimeoutError` and not `backoff_on_timeout`, set `wait_time = 0.0` (skip the `asyncio.sleep`). Thread `backoff_on_timeout` and the existing `max_retries` through `OpenRouterClient.query`. Add `max_retries: int` and `backoff_on_timeout: bool` to `EndpointState`; in `query_single`, pass `max_retries=state.max_retries, backoff_on_timeout=state.backoff_on_timeout` to `client.query`.
- [ ] **Step 4:** run → PASS; `OPENROUTER_API_KEY=dummy uv run pytest tests/ -q` green — the new fields default to current behavior, so existing construction sites (phase_2, analysis tools, tests) are unaffected.
- [ ] **Step 5:** `prek run --all-files`; commit `feat(bi): per-state retries + skip backoff on timeout`.

---

### Task 3: All-timeout abandon (mirror got_404)

**Files:**
- Modify: `src/trackllm_website/bi/common.py` (`EndpointState`; `query_single`; `query_all_for_token`; round loops in `run_queries` helpers)
- Test: `tests/test_bi_onboarding_safety.py`

**Interfaces:**
- `EndpointState` gains `abandon_after_timeouts: int | None = None` (None = disabled, preserving current behavior for monitoring/analysis-tool states), and runtime counters `timeout_count: int = 0`, `unresponsive: bool = False`.
- `query_single`: on a timeout error (`http_code == 0 and message.startswith("Timeout")`), `state.timeout_count += 1`; if `state.abandon_after_timeouts is not None and state.completed_queries >= state.abandon_after_timeouts and state.timeout_count == state.completed_queries` (≥N, all timeouts, no successes), set `state.unresponsive = True`, log `"{endpoint} unresponsive ({n} timeouts, 0 successes), abandoning for this run"`, return `False`.
- Every place that currently short-circuits on `state.got_404` also short-circuits on `state.unresponsive` (gate at top of `query_single`, and the `if state.got_404: return` checks in `query_all_for_token`).

- [ ] **Step 1: Failing test** — drive `query_single` with a `_Scripted(OpenRouterClient)` fake returning timeout errors; assert that after `abandon_after_timeouts` calls `state.unresponsive` is `True` and the next `query_single` returns `False` without querying. (Build the `EndpointState` with `abandon_after_timeouts=3` for a fast test; assert `client.calls == 3`.)
- [ ] **Step 2:** run → FAIL (`unresponsive`/`timeout_count` absent).
- [ ] **Step 3:** Implement per the Interfaces block. Mirror the existing `got_404` pattern at `bi/common.py` `query_single` (~line 800) and the `if state.got_404` gates in `query_all_for_token`.
- [ ] **Step 4:** run → PASS; full suite green.
- [ ] **Step 5:** `prek run --all-files`; commit `feat(bi): abandon endpoint after N all-timeout requests`.

---

### Task 4: Point the phase-1 onboarding constructor at `config.bi.phase_1`

**Files:**
- Modify: `src/trackllm_website/bi/phase_1.py` (the two `Phase1EndpointState(...)` constructions, ~lines 93 and 139)
- Test: `tests/test_bi_onboarding_safety.py`

**Interfaces:**
- Consumes: `EndpointState.max_retries`, `.backoff_on_timeout`, `.abandon_after_timeouts` (Tasks 2–3), which **default to current behavior** (retries = `config.api.max_retries`, `backoff_on_timeout=True`, timeout-abandon disabled).
- Both `Phase1EndpointState(...)` constructions add: `max_retries=config.bi.phase_1.max_retries`, `backoff_on_timeout=False`, `abandon_after_timeouts=config.bi.phase_1.abandon_after_timeouts`.
- Nothing else changes: `bi/phase_2.py` (monitoring) and the analysis tools (`bi_prevalence.py`, `hardware_noise.py`, `logprob_stats.py`) keep the behavior-preserving defaults.

- [ ] **Step 1: Failing test** — construct a `Phase1EndpointState` exactly as `phase_1.py` does (minimal real args + the new kwargs sourced from `config.bi.phase_1`) and assert `state.max_retries == 3`, `state.backoff_on_timeout is False`, `state.abandon_after_timeouts == 20`; and assert a bare `EndpointState(...)` (analysis-tool style, new kwargs omitted) keeps defaults `backoff_on_timeout is True` and timeout-abandon disabled. (No network.)
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** Add the three kwargs to both `Phase1EndpointState(...)` sites in `phase_1.py`.
- [ ] **Step 4:** run → PASS; full suite green (confirms phase_2 / analysis tools untouched).
- [ ] **Step 5:** `prek run --all-files`; commit `feat(bi): phase-1 onboarding uses the safety knobs`.

---

### Task 5: Per-endpoint 3h onboarding deadline (resume-preserving)

**Files:**
- Modify: `src/trackllm_website/update_endpoints.py` (`onboard_one` inside `update_endpoints_bi_lifecycle`)
- Test: `tests/test_bi_lifecycle.py` (extend) / `tests/test_bi_onboarding_safety.py`

**Interfaces:**
- Consumes: `config.bi.reinit.onboard_timeout_seconds`.
- `onboard_one` wraps the `await reinit(...)` call in `asyncio.wait_for(reinit(...), config.bi.reinit.onboard_timeout_seconds)`. On `asyncio.TimeoutError`: log `"{endpoint} onboarding exceeded {h}h, will resume next run"`; return (do NOT mark failed). Partial `_temp_results` are already persisted on disk during the run, so the next run's `EndpointState.__post_init__` resumes from them — verify no on-cancel state wipe (Decision D). Keep the existing broad `except Exception` logging below it.

- [ ] **Step 1: Failing test** — in `tests/test_bi_lifecycle.py`, monkeypatch `reinit` with an `async def` that sleeps longer than a tiny patched `onboard_timeout_seconds` (monkeypatch `config.bi.reinit.onboard_timeout_seconds = 0.05` or patch the constant the code reads) and asserts `update_endpoints_bi_lifecycle([ep])` returns without raising and the endpoint produced no crash/exception state — i.e. the deadline is caught, not propagated. Reuse `_patch_lifecycle_deps`.
- [ ] **Step 2:** run → FAIL (TimeoutError propagates / no wait_for).
- [ ] **Step 3:** Implement the `asyncio.wait_for` wrap + `except asyncio.TimeoutError` handler in `onboard_one`. Confirm by reading `reinit`/`run_queries` that partial results are written incrementally (SAVE_INTERVAL) so cancellation doesn't lose accumulated BIs; if a final flush is missing on cancel, add a `finally`-save in the reinit run loop.
- [ ] **Step 4:** run → PASS; full suite green.
- [ ] **Step 5:** `prek run --all-files`; commit `feat(bi): 3h per-endpoint onboarding deadline (resumable)`.

---

## Worst-case after this plan
Dead endpoint (all timeouts): abandons at ~20 queries (~1–2 min). Slow-but-responsive / never-enough-BIs: capped at the 3h deadline (and resumes next run). Per-endpoint worst case **~140h → ≤3h**; dead endpoints **15,000 → ~20 queries**.
