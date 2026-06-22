# Resumable BI Onboarding (Decision D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BI onboarding/recheck resume phase-1 border-input discovery across runs (accumulating toward enough BIs or the phase-1 budget) instead of restarting from scratch each time the 3h deadline cancels it.

**Architecture:** Phase-1 progress is already durably persisted during a run (`SAVE_INTERVAL=5` flushes + `run_queries`'s `finally: flush()`) and reloaded by `Phase1EndpointState.__post_init__`; the only thing defeating resume is that `discover_candidates` writes into a `tempfile.TemporaryDirectory()` deleted on exit/cancel. Replace that with a persistent **per-endpoint** dir `config.bi.data_dir / "onboarding_progress" / <slug>`, and delete it only when `reinit` returns normally (terminal: `ok`/`no_bis`/`bad_temperature`) — never on the deadline's cancellation, so the next run resumes.

**Tech Stack:** Python 3.13, asyncio, pydantic-settings, pytest, beartype, `uv`.

## Global Constraints

- Changes touch **only** `src/trackllm_website/bi/reinit.py` and its test. Do **not** touch `bi/common.py` / `bi/phase_1.py` persistence (already correct), `bi/phase_2.py` / `bi/monitor.py` (monitoring is a separate path), `bi/sampling.py` (references stay in-memory — deferred), or the `asyncio.wait_for` deadline in `update_endpoints.py` (already returns without wiping state on timeout).
- Persistent scratch dir: `config.bi.data_dir / "onboarding_progress" / <slug>` where `<slug> = slugify(f"{endpoint.model}#{endpoint.provider}")` — the same slug `get_output_path` uses.
- Cleanup runs **only on `reinit`'s normal return** (any of `ok` / `no_bis` / `bad_temperature`). `asyncio.CancelledError` (raised in the inner body when the deadline cancels) must propagate past the cleanup line so the scratch survives.
- **Reference-sampling persistence is out of scope** (Decision: defer). Do not modify `sample_prompts` or persist partial references.
- TDD; run `prek run --all-files` before each commit (commit with `git commit --no-verify` — the repo's stale `pre-commit` hook shells to an uninstalled binary; `prek` is run manually). Tests run with `OPENROUTER_API_KEY=dummy uv run pytest`. New fakes standing in for `OpenRouterClient` must subclass it (package-wide beartype enforces the type).

## File Structure

- `src/trackllm_website/bi/reinit.py` — add `onboarding_progress_dir()` + `_cleanup_onboarding_progress()` helpers; point `discover_candidates` at the persistent dir (drop `tempfile`); wrap `reinit`'s body so cleanup runs only on normal return.
- `tests/test_bi_reinit.py` — extend with resume / terminal-cleanup / cancel-preserves-scratch tests.

Existing relevant code (read before editing):
- `reinit.py:69-83` `discover_candidates` (the tempdir to replace)
- `reinit.py:86-148` `reinit` (the body to wrap)
- `reinit.py:40-66` `parse_phase_1_results` (globs `*.json` — why the dir must be per-endpoint)
- `phase_1.py:66-124` `phase_1a` (does `base_dir.mkdir(parents=True, exist_ok=True)`; constructs `Phase1EndpointState`)
- `common.py:355-424` `TemperatureResults` (`SAVE_INTERVAL=5`, `__post_init__` reload, `flush`)

---

### Task 1: Persistent per-endpoint scratch dir + cleanup helpers

Add the path helper and the best-effort cleanup helper, and switch `discover_candidates` from a tempdir to the persistent dir. After this task, partial phase-1 results survive across `discover_candidates` calls and resume is exercised by the test; cleanup is wired into `reinit` in Task 2.

**Files:**
- Modify: `src/trackllm_website/bi/reinit.py`
- Test: `tests/test_bi_reinit.py`

**Interfaces:**
- Produces: `onboarding_progress_dir(endpoint: Endpoint) -> Path` returning `config.bi.data_dir / "onboarding_progress" / slugify(f"{endpoint.model}#{endpoint.provider}")`.
- Produces: `_cleanup_onboarding_progress(endpoint: Endpoint) -> None` — `shutil.rmtree(onboarding_progress_dir(endpoint), ignore_errors=True)` (no-op if absent).
- Changes: `discover_candidates(endpoint, exclude)` now calls `phase_1a([endpoint], 0.0, onboarding_progress_dir(endpoint))` (was a `tempfile.TemporaryDirectory()`), and parses `config.bi.get_phase_1_dir(0.0, onboarding_progress_dir(endpoint))`. Return signature unchanged: `tuple[list[str], float]`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_bi_reinit.py`. A query-counting fake client (subclass of `OpenRouterClient`, per beartype) drives real `phase_1a`; the first run is interrupted partway, the second resumes and issues only the remaining queries. Keep it small via tiny config overrides.

```python
import shutil
from trackllm_website.api import OpenRouterClient, Response
from trackllm_website.bi import phase_1 as phase_1_mod


class _CountingClient(OpenRouterClient):
    """Counts queries and returns a fixed border-y token; raises after `fail_after`."""

    def __init__(self, fail_after: int | None = None):
        self.calls = 0
        self.fail_after = fail_after

    async def query(self, endpoint, prompt, **kwargs):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise asyncio.CancelledError()
        # Alternate tokens so prompts look like border inputs (>=2 distinct outputs).
        token = "a" if self.calls % 2 else "b"
        return Response(content=token, error=None)

    async def close(self):
        pass


def _shrink_phase1(monkeypatch, tmp_path, *, tokens, queries_per_token):
    """Point data_dir at tmp_path and shrink phase-1 so a run is a handful of queries."""
    monkeypatch.setattr(reinit_mod.config.bi, "data_dir", tmp_path)
    p1 = reinit_mod.config.bi.phase_1
    monkeypatch.setattr(p1, "tokens_per_endpoint", tokens)
    monkeypatch.setattr(p1, "queries_per_token", queries_per_token)
    monkeypatch.setattr(p1, "queries_per_candidate", queries_per_token)
    # Avoid early-stop so the budget (tokens * queries_per_token) is the only ceiling.
    monkeypatch.setattr(p1, "target_border_inputs", 9999)
    monkeypatch.setattr(p1, "border_input_candidate_ratio", 1.0)


def test_discover_candidates_resumes_partial_progress(monkeypatch, tmp_path):
    _shrink_phase1(monkeypatch, tmp_path, tokens=4, queries_per_token=2)
    ep = ENDPOINT

    # phase_1a builds its own client internally via run_queries; patch run_queries'
    # OpenRouterClient to our counting client so we can count + interrupt.
    first = _CountingClient(fail_after=3)
    monkeypatch.setattr("trackllm_website.bi.common.OpenRouterClient", lambda *a, **k: first)
    # Skip strategy resolution network call.
    async def fake_resolve(client, endpoints, **k):
        return ({str(e): None for e in endpoints}, [])
    monkeypatch.setattr(phase_1_mod, "resolve_strategies", fake_resolve)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(reinit_mod.discover_candidates(ep, exclude=[]))
    assert first.calls == 4  # 3 ok + the 4th raises

    scratch = reinit_mod.onboarding_progress_dir(ep)
    assert scratch.exists()  # partial results persisted, NOT a tempdir

    second = _CountingClient()
    monkeypatch.setattr("trackllm_website.bi.common.OpenRouterClient", lambda *a, **k: second)
    asyncio.run(reinit_mod.discover_candidates(ep, exclude=[]))
    # Budget is 4 tokens * 2 = 8 queries total; 3 already done -> <= 5 remain.
    assert second.calls <= 5
    assert second.calls < 8  # proves resume, not restart
```

Add `import asyncio`, `import pytest` at the top if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `OPENROUTER_API_KEY=dummy uv run pytest tests/test_bi_reinit.py::test_discover_candidates_resumes_partial_progress -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'onboarding_progress_dir'`.

- [ ] **Step 3: Write minimal implementation** — in `reinit.py`: drop `import tempfile`, add `import shutil`, add `from trackllm_website.util import slugify`. Add the helpers and rewrite `discover_candidates`:

```python
def onboarding_progress_dir(endpoint: Endpoint) -> Path:
    """Persistent per-endpoint scratch dir for resumable phase-1 onboarding.

    Per-endpoint (not shared) because parse_phase_1_results globs *.json and
    onboarding runs at concurrency 40; a shared dir would mix endpoints' files.
    """
    slug = slugify(f"{endpoint.model}#{endpoint.provider}")
    return config.bi.data_dir / "onboarding_progress" / slug


def _cleanup_onboarding_progress(endpoint: Endpoint) -> None:
    shutil.rmtree(onboarding_progress_dir(endpoint), ignore_errors=True)


async def discover_candidates(
    endpoint: Endpoint, exclude: list[str]
) -> tuple[list[str], float]:
    """Run phase 1a discovery for one endpoint, resuming from persisted progress.

    Returns (candidates, prevalence) where prevalence = n_border / n_prompts_sampled,
    so the caller can decide whether to run the temperature gate.
    """
    base_dir = onboarding_progress_dir(endpoint)
    await phase_1a([endpoint], 0.0, base_dir)
    results_dir = config.bi.get_phase_1_dir(0.0, base_dir)
    candidates, n_sampled = parse_phase_1_results(results_dir, exclude)
    prevalence = len(candidates) / n_sampled if n_sampled else 0.0
    return candidates, prevalence
```

- [ ] **Step 4: Run test to verify it passes**

Run: `OPENROUTER_API_KEY=dummy uv run pytest tests/test_bi_reinit.py::test_discover_candidates_resumes_partial_progress -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
prek run --all-files
git add src/trackllm_website/bi/reinit.py tests/test_bi_reinit.py
git commit --no-verify -m "feat(bi): persistent per-endpoint onboarding scratch dir (resumable phase 1)"
```

---

### Task 2: Cleanup on terminal return, preserved on cancel

Wire `_cleanup_onboarding_progress` so the scratch dir is removed whenever `reinit` returns normally (terminal outcome), but survives when the deadline cancels mid-run.

**Files:**
- Modify: `src/trackllm_website/bi/reinit.py`
- Test: `tests/test_bi_reinit.py`

**Interfaces:**
- Consumes: `onboarding_progress_dir`, `_cleanup_onboarding_progress` (Task 1).
- Changes: rename the current `reinit` body to `async def _reinit(client, strategy, endpoint, old_bis, now) -> ReinitResult` (identical body, same returns). New `reinit` with the same signature awaits `_reinit`, then cleans, then returns; cancellation propagates past the cleanup.

- [ ] **Step 1: Write the failing test** — append to `tests/test_bi_reinit.py`. Two tests: terminal return cleans; cancellation preserves.

```python
def test_reinit_cleans_scratch_on_terminal_return(monkeypatch, tmp_path):
    monkeypatch.setattr(reinit_mod.config.bi, "data_dir", tmp_path)
    scratch = reinit_mod.onboarding_progress_dir(ENDPOINT)
    scratch.mkdir(parents=True)
    (scratch / "marker.txt").write_text("partial")

    # no_bis terminal path: discovery returns nothing.
    async def fake_discover(endpoint, exclude):
        return [], 0.0

    monkeypatch.setattr(reinit_mod, "discover_candidates", fake_discover)
    result = asyncio.run(reinit_mod.reinit(None, None, ENDPOINT, [], NOW))
    assert result.reason == "no_bis"
    assert not scratch.exists()  # terminal -> cleaned


def test_reinit_preserves_scratch_on_cancel(monkeypatch, tmp_path):
    monkeypatch.setattr(reinit_mod.config.bi, "data_dir", tmp_path)
    scratch = reinit_mod.onboarding_progress_dir(ENDPOINT)
    scratch.mkdir(parents=True)
    (scratch / "marker.txt").write_text("partial")

    async def cancelling_discover(endpoint, exclude):
        raise asyncio.CancelledError()

    monkeypatch.setattr(reinit_mod, "discover_candidates", cancelling_discover)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(reinit_mod.reinit(None, None, ENDPOINT, [], NOW))
    assert scratch.exists()  # cancel -> scratch kept for resume
```

- [ ] **Step 2: Run test to verify it fails**

Run: `OPENROUTER_API_KEY=dummy uv run pytest tests/test_bi_reinit.py::test_reinit_cleans_scratch_on_terminal_return tests/test_bi_reinit.py::test_reinit_preserves_scratch_on_cancel -v`
Expected: `test_reinit_cleans_scratch_on_terminal_return` FAILs (scratch still exists — no cleanup yet); the cancel test may already pass (no cleanup) but must still pass after.

- [ ] **Step 3: Write minimal implementation** — in `reinit.py`, rename `async def reinit(...)` (the current full body at lines ~86-148) to `async def _reinit(...)` (unchanged body), then add the wrapper:

```python
async def reinit(
    client,
    strategy: QueryStrategy | None,
    endpoint: Endpoint,
    old_bis: list[str],
    now: datetime,
) -> ReinitResult:
    """Re-init / onboard one endpoint, resuming persisted phase-1 progress.

    Onboarding scratch is removed only on a normal (terminal) return; if the
    caller's deadline cancels us mid-run, CancelledError propagates past the
    cleanup so the next run resumes from disk.
    """
    result = await _reinit(client, strategy, endpoint, old_bis, now)
    _cleanup_onboarding_progress(endpoint)
    return result
```

Keep `_reinit`'s docstring/body exactly as the current `reinit` (the onboarding/temperature-gate logic).

- [ ] **Step 4: Run test to verify it passes**

Run: `OPENROUTER_API_KEY=dummy uv run pytest tests/test_bi_reinit.py -v`
Expected: PASS (new tests + all existing reinit tests — their `discover_candidates`/`sample_prompts` monkeypatches still apply, and cleanup of a non-existent dir is a no-op).

- [ ] **Step 5: Commit**

```bash
prek run --all-files
git add src/trackllm_website/bi/reinit.py tests/test_bi_reinit.py
git commit --no-verify -m "feat(bi): clean onboarding scratch on terminal return, keep it on deadline cancel"
```

---

### Task 3: Full-suite regression gate

No new code — confirm the change is isolated and nothing else regressed.

**Files:** none.

- [ ] **Step 1: Run the full suite**

Run: `OPENROUTER_API_KEY=dummy uv run pytest -q`
Expected: PASS (153+), no new failures. In particular `test_bi_lifecycle.py` (deadline caught), `test_bi_reinit.py`, and any analysis-tool/phase tests are green.

- [ ] **Step 2: Lint/format gate**

Run: `prek run --all-files`
Expected: clean.

- [ ] **Step 3: Commit (only if Step 1/2 required a fixup)**

```bash
git add -A
git commit --no-verify -m "test(bi): regression gate for resumable onboarding"
```

---

## Self-review notes

- **Spec coverage:** persistent per-endpoint dir (Task 1) ✓; resume = automatic via reload (Task 1 test) ✓; cleanup-on-terminal incl. `no_bis`/`bad_temperature` (Task 2) ✓; cancel preserves scratch (Task 2) ✓; reference persistence deferred (no task — by design) ✓; monitoring/sampling untouched (Global Constraints) ✓.
- **Naming:** `onboarding_progress_dir` / `_cleanup_onboarding_progress` / `_reinit` used consistently across tasks.
- **Watch:** if `phase_1a`'s internal client construction is not `trackllm_website.bi.common.OpenRouterClient`, adjust the Task-1 monkeypatch target to wherever `run_queries` instantiates the client (`common.py` `run_queries` builds `OpenRouterClient()` — confirm the patch path during implementation).
