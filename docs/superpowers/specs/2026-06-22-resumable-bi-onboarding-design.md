# Resumable BI Onboarding (Decision D) — Design

**Date:** 2026-06-22
**Builds on:** PR #9 (`438363400f` — 3h per-endpoint onboarding deadline, all-timeout abandon, fewer onboarding retries) and its plan `docs/superpowers/plans/2026-06-21-bi-onboarding-safety.md`.

## Goal

An endpoint short of its BI target must **accumulate border inputs across runs** until it has enough BIs *or* exhausts the phase-1 query budget — never restart from scratch. The merged 3h deadline (`asyncio.wait_for`) cancels `reinit()` and, today, discards all partial work because phase-1 discovery runs in a temp dir.

## Root cause

`reinit()` → `discover_candidates` (`src/trackllm_website/bi/reinit.py`) runs the phase-1a border search inside a `tempfile.TemporaryDirectory()`, which is deleted on normal exit **and** on the deadline's cancel. So nothing lands in a location the next run can reload.

Everything else needed for resume already exists:
- `TemperatureResults` flushes to disk every `SAVE_INTERVAL = 5` queries, and `run_queries` has a `finally: state.flush()` (`bi/common.py`) — progress is durably persisted *during* a run, including a best-effort flush on cancellation.
- `Phase1EndpointState.__post_init__` → `TemperatureResults.__post_init__` reloads existing results; `get_unfinished_prompts` + `stop_early_phase1` (`bi/phase_1.py`) only query what is still missing.

So once phase-1a writes to a **persistent** location, resume happens for free.

## Design

### Core change: one persistent per-endpoint scratch dir

Replace the `tempfile.TemporaryDirectory()` in `discover_candidates` with a persistent, per-endpoint directory:

```
config.bi.data_dir / "onboarding_progress" / <endpoint-slug>
```

where `<endpoint-slug> = slugify(f"{endpoint.model}#{endpoint.provider}")` (the same slug `get_output_path` uses).

- **Resume is automatic.** Re-running `reinit` with `old_bis=[]` calls `discover_candidates` → `phase_1a([endpoint], 0.0, scratch_dir)`. `Phase1EndpointState.__post_init__` reloads `_temp_results` from disk; `get_unfinished_prompts`/`stop_early_phase1` skip completed work. The run continues from where it left off and issues only the remaining queries.
- **Per-endpoint subdir (not a shared dir)** is required: `parse_phase_1_results` globs `*.json` in the results dir, and onboarding runs at concurrency 40. A shared dir would mix every concurrent endpoint's file into every endpoint's parse. Per-endpoint isolation also makes cleanup a single `rmtree`.
- **Distinct from the manual `data_dir / "phase_1"` pipeline** (standalone `phase_1a`/`phase_1b`), so onboarding scratch never collides with that workflow.

A small helper centralizes the path:

```python
def onboarding_progress_dir(endpoint: Endpoint) -> Path:
    slug = slugify(f"{endpoint.model}#{endpoint.provider}")
    return config.bi.data_dir / "onboarding_progress" / slug
```

`discover_candidates` uses it as `base_dir` (creating it via the existing `phase_1a` `mkdir(parents=True, exist_ok=True)`), and the cleanup step removes it.

### Cleanup lifecycle — the resume-vs-restart rule

| Situation | Scratch dir | Effect |
|---|---|---|
| In-progress onboard/recheck re-run | **kept** | phase-1a resumes from disk |
| `reinit` returns `ok` / `no_bis` / `bad_temperature` (terminal) | **deleted** | budget spent or BIs found; a future genuine recheck starts fresh |
| Deadline fires (`asyncio.wait_for` cancels `reinit`) | **kept** | next run resumes |

- **Terminal cleanup on every normal return**, not just `ok`. `no_bis` is reached only after phase-1a hits `reached_target` or exhausts the budget (it cannot return `no_bis` with work still pending), so cleaning there is correct: the phase-1 budget was spent and retirement is terminal. `bad_temperature` is likewise terminal (the endpoint is cached/excluded). Deleting on terminal is exactly what lets a later recheck (`old_bis=[]`, "rediscover from scratch") see no scratch and start clean.
- **Cancellation must skip cleanup.** `asyncio.wait_for` on timeout cancels the inner `reinit`, which observes `asyncio.CancelledError` (a `BaseException` in Python 3.13). Structure `reinit` so cleanup runs only on the normal-return path:

  ```python
  async def reinit(client, strategy, endpoint, old_bis, now) -> ReinitResult:
      result = await _reinit(client, strategy, endpoint, old_bis, now)  # current body
      _cleanup_onboarding_progress(endpoint)
      return result
  ```

  If `_reinit` is cancelled mid-flight, `CancelledError` propagates out of `reinit` (skipping the cleanup line) and `wait_for` surfaces it to `onboard_one` as `asyncio.TimeoutError` — which already returns without touching state. The scratch dir is left intact for the next run.

  `_cleanup_onboarding_progress` does a best-effort `shutil.rmtree(dir, ignore_errors=True)` guarded by existence, so an already-absent dir (e.g. discovery never ran because survivors sufficed) is a no-op.

### The "phase-1 query budget"

The budget is implicit and unchanged: `tokens_per_endpoint × queries_per_token`. `get_unfinished_prompts` returns fewer items as queries accrue; when it empties, phase-1a does nothing further and `discover_candidates` returns whatever candidates exist. Resume accumulates toward this same ceiling across runs — "enough BIs OR the phase-1 budget" falls out of the existing early-stop/exhaustion logic.

## Out of scope

**Reference-sampling (phase-1b) persistence.** After discovery, `reinit` collects `reference_samples` per candidate via `sample_prompts` in memory; partial references are not persisted (only the final `ok` epoch is, via `_persist_reference`). If the deadline cancels *during* reference collection, those samples are lost; the next run re-discovers phase-1 instantly (cached) and redoes reference sampling from scratch. References are bounded (`≤ target_border_inputs × reference_samples` queries) and small versus the ~15k-query phase-1 budget, and the resume run's full 3h is available for them. Persisting them would add surface to the shared `sampling.py` (also used by re-probe and the temperature gate) for marginal benefit. **Decision: defer.** Revisit only if a slow endpoint is observed stalling specifically in the reference phase.

## What is NOT touched

- `bi/common.py` / `bi/phase_1.py` persistence and resume logic — already correct.
- `bi/phase_2.py` / `bi/monitor.py` — monitoring is a separate `EndpointState`/loop (Decision A holds by construction).
- `bi/sampling.py` — unchanged (references stay in-memory per the deferral above).
- The `asyncio.wait_for` deadline and `onboard_one` handler in `update_endpoints.py` — already return without wiping state on timeout; we only ensure scratch survives the cancel (achieved in `reinit`).

## Testing

- **Resume (the headline test):** point `config.bi.data_dir` at a `tmp_path`; drive phase-1a (via `discover_candidates`/`reinit`) with a query-counting fake `OpenRouterClient` (subclass, per the beartype constraint). Interrupt after K queries (deadline or a client that cancels), assert the scratch file under `onboarding_progress/<slug>/` holds K results. Re-run; assert the client is asked only for the *remaining* queries (total − K), not a full restart, and that accumulated results persist.
- **Terminal cleanup:** after a normal `reinit` return (`ok`, `no_bis`, `bad_temperature`), assert the endpoint's `onboarding_progress/<slug>/` dir is gone.
- **Cancel preserves scratch:** when `reinit` is cancelled (simulating the deadline), assert the scratch dir remains.
- Existing `tests/test_bi_reinit.py` continues to pass (its `discover_candidates`/`sample_prompts` monkeypatches bypass real I/O).

## Done when

A test interrupts onboarding mid-search, re-runs, and asserts it continues (issues only remaining queries, BIs accrue) rather than restarts; terminal-cleanup and cancel-preserves-scratch tests pass; full suite (153+) and `prek run --all-files` green.
