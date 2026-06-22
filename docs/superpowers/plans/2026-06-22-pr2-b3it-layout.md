# PR2 — B3IT Layout Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move B3IT data from `data_bi/` to `website/data/b3it/` so LT (`website/data/lt/`) and B3IT are symmetric peers under one root.

**Architecture:** All B3IT paths derive from `config.bi.data_dir`, so changing that one config value relocates the whole subtree; the committed data is `git mv`-ed to match. A latent conflation is fixed: `costs.py` writes `bi_costs.json` via `config.data_dir` (the LT root) — repointed to `config.bi.data_dir`. Zero live-site impact (nothing serves B3IT data yet); the static-site generator reads `website/data/lt` (from PR1), so `b3it/` is a sibling outside its enumeration root.

**Tech Stack:** Python 3.13, pydantic-settings, pytest, uv; GitHub Actions.

## Global Constraints

- `config.data_dir` value in `config.toml` stays `"website/data"` (the shared root; do NOT change it).
- `config.bi.data_dir` value in `config.toml` becomes `"website/data/b3it"`.
- LT data (`website/data/lt/`) and the generator are NOT touched.
- No edits to `run-main.yml`, `update-endpoints.yml`, `deploy-pages.yml`.
- `data_bi/` is ~274 MB of git-tracked blobs; move it wholesale with `git mv`.
- Use `prek run --files <changed>` before committing (the `pre-commit` binary is absent; `prek` runs the ruff hooks); commit with `--no-verify` (the git hook shells out to the missing `pre-commit`). The data-move commit has no Python files to lint — `prek` is only needed when a commit touches `.py` files.
- Tools: `rg`/`fd`, `uv run pytest`. `config` is a module-level `Config()` instantiated at import, which requires `OPENROUTER_API_KEY` in the env; tests set it via `monkeypatch` or rely on the project `.env`.

---

### Task 1: Relocate B3IT data to `website/data/b3it/`

This is the atomic migration: the config value and the `git mv` must land together or every B3IT path would point at an empty/missing dir.

**Files:**
- Modify: `config.toml` (the `[bi]` section, `data_dir` on line 22)
- Modify: `tests/test_config_paths.py` (add a B3IT assertion)
- Modify: `.github/workflows/bi-monitor.yml:41` (commit path)
- Move: `data_bi/` → `website/data/b3it/`

**Interfaces:**
- Consumes: existing `config.bi.data_dir` (a `Path`) and its derived properties (`state_dir`, `phase_2_dir`, `tokenizers_dir`, `get_phase_1_dir`).
- Produces: B3IT data under `website/data/b3it/`; `config.bi.data_dir == Path("website/data/b3it")`.

- [ ] **Step 1: Write the failing test**

In `tests/test_config_paths.py`, add (keep the existing `from pathlib import Path` / `from trackllm_website.config import Config` imports):

```python
def test_bi_data_dir_under_website_data(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    cfg = Config()
    assert cfg.bi.data_dir == Path("website/data/b3it")
    assert cfg.bi.state_dir == cfg.bi.data_dir / "state"
    assert cfg.bi.phase_2_dir == cfg.bi.data_dir / "phase_2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_paths.py::test_bi_data_dir_under_website_data -v`
Expected: FAIL — `cfg.bi.data_dir` is still `Path("data_bi")`, so the first assert fails.

- [ ] **Step 3: Change the config value**

In `config.toml`, under `[bi]` (line 22), change:

```toml
data_dir = "website/data/b3it"
```

(Leave the top-level `data_dir = "website/data"` on line 2 unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_paths.py::test_bi_data_dir_under_website_data -v`
Expected: PASS

- [ ] **Step 5: Move the committed B3IT data**

```bash
git mv data_bi website/data/b3it
# Sanity: data_bi is gone; b3it is populated alongside lt
fd -d 1 . website/data
ls website/data/b3it | head
```

Expected: `fd` prints `website/data/b3it` and `website/data/lt` (and nothing else); `ls` shows the B3IT subdirs (`state`, `phase_2`, `phase_1`, `tokenizers`, `bi_prevalence`, `logprob_stats`).

- [ ] **Step 6: Update the bi-monitor commit path**

In `.github/workflows/bi-monitor.yml`, line 41, change:

```yaml
        git add website/data
```

(Was `git add data_bi`. The broader path also covers the `website/data/spend/` ledger bi-monitor writes in PR3; it never modifies LT data, so nothing extra is staged.)

- [ ] **Step 7: Confirm no stale `data_bi` references remain**

Run: `rg -n "data_bi" --glob '!uv.lock' --glob '!docs/**' .`
Expected: no matches in code or workflows (design docs under `docs/` may mention it historically — those are fine). If any code/workflow match appears, repoint it to `website/data/b3it` (or `config.bi.data_dir`) and note it in the report.

- [ ] **Step 8: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS (all BI tests; paths derive from `config.bi.data_dir`, and any test reading the real committed state now finds it under `website/data/b3it`).

- [ ] **Step 9: Commit**

```bash
prek run --files config.toml tests/test_config_paths.py >/dev/null 2>&1
git add config.toml tests/test_config_paths.py .github/workflows/bi-monitor.yml website/data data_bi
git commit --no-verify -m "refactor(b3it): move B3IT data under website/data/b3it"
```

(Staging both `website/data` and `data_bi` records the rename; `git add -A` would also work.)

---

### Task 2: Fix the `bi_costs.json` conflation in `costs.py`

`costs.py` writes `bi_costs.json` (a B3IT artifact) under `config.data_dir` — the LT root. After PR1 that root is `website/data` and after later work it is conceptually LT's neighbor; either way the file belongs under B3IT. Repoint it to `config.bi.data_dir`. No file move is needed (`bi_costs.json` has never been generated/committed) and nothing reads it (only `costs.py` writes it).

**Files:**
- Modify: `src/trackllm_website/bi/costs.py` (extract a `costs_path()` helper; use it in `write_cost_summary`)
- Test: `tests/test_bi_costs.py` (add a path assertion)

**Interfaces:**
- Consumes: `config.bi.data_dir` (Task 1), `COSTS_FILENAME` (existing constant = `"bi_costs.json"`).
- Produces: `costs_path() -> Path` returning `config.bi.data_dir / COSTS_FILENAME`.

- [ ] **Step 1: Write the failing test**

In `tests/test_bi_costs.py`, add (note the existing imports at the top of that file; add `from pathlib import Path` and import `costs_path` and `config`):

```python
def test_costs_path_under_b3it():
    from pathlib import Path

    from trackllm_website.bi.costs import costs_path
    from trackllm_website.config import config

    assert costs_path() == config.bi.data_dir / "bi_costs.json"
    assert costs_path() == Path("website/data/b3it/bi_costs.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bi_costs.py::test_costs_path_under_b3it -v`
Expected: FAIL with `ImportError: cannot import name 'costs_path'`.

- [ ] **Step 3: Add the helper and use it**

In `src/trackllm_website/bi/costs.py`, add a helper near the top (just after the `COSTS_FILENAME = "bi_costs.json"` line / the imports):

```python
def costs_path() -> Path:
    return config.bi.data_dir / COSTS_FILENAME
```

Then in `write_cost_summary`, replace the line:

```python
    path = Path(config.data_dir) / COSTS_FILENAME
```

with:

```python
    path = costs_path()
```

(`Path` and `config` are already imported in `costs.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bi_costs.py::test_costs_path_under_b3it -v`
Expected: PASS

- [ ] **Step 5: Run the cost tests**

Run: `uv run pytest tests/test_bi_costs.py -q`
Expected: PASS (existing `build_cost_summary` / `ensure_costs` / `format_preview` tests unaffected).

- [ ] **Step 6: Commit**

```bash
prek run --files src/trackllm_website/bi/costs.py tests/test_bi_costs.py >/dev/null 2>&1
git add src/trackllm_website/bi/costs.py tests/test_bi_costs.py
git commit --no-verify -m "fix(b3it): write bi_costs.json under config.bi.data_dir, not the LT root"
```

---

## Self-Review

**Spec coverage (PR 2 section of the umbrella spec):**
- `[bi] data_dir` → `website/data/b3it` → Task 1 Step 3. ✓
- `git mv data_bi website/data/b3it` → Task 1 Step 5. ✓
- Fix `bi_costs.json` conflation (repoint `costs.py` to `config.bi.data_dir`) → Task 2. ✓
- `bi-monitor.yml` `git add data_bi` → `git add website/data` → Task 1 Step 6. ✓
- Generator unaffected (reads `website/data/lt`) → asserted in Global Constraints; no generator change in either task. ✓
- `#11` `reinit.py` uses `config.bi.data_dir`-derived paths → relocated automatically; no special handling (no task needed). ✓
- Tests: `config.bi` paths resolve under `website/data/b3it` (Task 1 Step 1); `costs.py` writes under `config.bi.data_dir` (Task 2 Step 1). ✓
- No-stale-reference guard → Task 1 Step 7. ✓

**Placeholder scan:** none — every code/command step has concrete content.

**Type consistency:** `config.bi.data_dir` is a `Path`; `costs_path()` returns `config.bi.data_dir / COSTS_FILENAME` (a `Path`), consumed verbatim in Task 2's test and in `write_cost_summary`. `COSTS_FILENAME` is the existing constant, unchanged.
