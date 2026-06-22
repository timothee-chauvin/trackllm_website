# PR1 — LT Layout Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move LT data from `website/data/{slug}` to `website/data/lt/{slug}` so LT and B3IT can later be symmetric peers under one root.

**Architecture:** `config.data_dir` stays the `website/data` root; a new `config.lt_dir` property (`data_dir / "lt"`) becomes LT's dir. All LT readers/writers switch to `lt_dir`, the committed data is `git mv`-ed under `lt/`, and the frontend's fetch base gains the `lt/` segment. The static-site generator iterating `website/data/lt` means future siblings (`b3it/`, `spend/`) fall outside its enumeration root, so no reserved-name skip is ever needed.

**Tech Stack:** Python 3.13, pydantic-settings, pytest, uv; TypeScript + Bun (frontend); GitHub Actions.

## Global Constraints

- `config.data_dir` value in `config.toml` stays `"website/data"` (the shared root; do NOT change it).
- LT's dir is `config.lt_dir` = `config.data_dir / "lt"`.
- Leave `bi/costs.py` on `config.data_dir` — `bi_costs.json` is a B3IT artifact, repointed in PR2, not here.
- No reserved-name skip in the generator (the symmetric layout makes it unnecessary).
- No `deploy-pages.yml`, `run-main.yml`, or `update-endpoints.yml` edits (deploy uploads `website/`; the run workflows commit via `git add .`).
- Use `prek run --files <changed>` before committing (the `pre-commit` binary is absent; `prek` runs the ruff hooks); commit with `--no-verify` since the git hook shells out to the missing `pre-commit`.
- Tools: `rg`/`fd`, `uv run pytest`.

---

### Task 1: Add `config.lt_dir` property

**Files:**
- Modify: `src/trackllm_website/config.py` (the `Config(BaseSettings)` class, after the `data_dir: Path` field near line 219)
- Test: `tests/test_config_paths.py` (create)

**Interfaces:**
- Consumes: existing `Config.data_dir: Path`.
- Produces: `Config.lt_dir -> Path` (= `data_dir / "lt"`), used by Tasks 2–3.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_paths.py`:

```python
from pathlib import Path

from trackllm_website.config import Config


def test_lt_dir_is_lt_subdir_of_data_dir(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    cfg = Config()
    assert cfg.data_dir == Path("website/data")
    assert cfg.lt_dir == cfg.data_dir / "lt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_paths.py -v`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'lt_dir'`

- [ ] **Step 3: Add the property**

In `src/trackllm_website/config.py`, inside `class Config(BaseSettings)`, add after the `plotting: PlottingConfig` field (i.e. after the "read from config.toml" block):

```python
    @property
    def lt_dir(self) -> Path:
        return self.data_dir / "lt"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_paths.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
prek run --files src/trackllm_website/config.py tests/test_config_paths.py >/dev/null 2>&1
git add src/trackllm_website/config.py tests/test_config_paths.py
git commit --no-verify -m "feat(config): add lt_dir property (data_dir/lt)"
```

---

### Task 2: Repoint LT Python readers/writers to `lt_dir`, move the data, fix the CLI test

This is the atomic migration: switching the code and moving the data must land together or the generator/scorer would read an empty `lt/`. The frontend fetch path (Task 3) is split out because a reviewer can judge the TS change independently.

**Files:**
- Modify: `src/trackllm_website/main.py:42`
- Modify: `src/trackllm_website/update_endpoints.py:250`
- Modify: `src/trackllm_website/lt_scores.py:223` and `:255`
- Modify: `src/trackllm_website/generate_site.py:12`
- Modify: `tests/test_lt_cli.py` (seed under `lt/`)
- Move: every child of `website/data/` → `website/data/lt/`

**Interfaces:**
- Consumes: `config.lt_dir` (Task 1).
- Produces: LT per-slug data and `lt_changes.json` under `website/data/lt/`; readers/writers all keyed on `config.lt_dir`.

- [ ] **Step 1: Update the CLI test to seed under `lt/` (failing test first)**

In `tests/test_lt_cli.py`, the test seeds via `ResultsStorage(data_dir)` at `tmp_path` and runs the CLI with `DATA_DIR=tmp_path`. After the migration the CLI reads `config.lt_dir` (`tmp_path/lt`). Change the seed target to the `lt` subdir. Find the seeding call (line ~16, `storage = ResultsStorage(data_dir)`) and its caller; make the seeded storage write under `Path(data_dir) / "lt"`:

```python
def _seed_endpoint(data_dir, n=50):
    storage = ResultsStorage(Path(data_dir) / "lt")
    # ... rest unchanged ...
```

Ensure `from pathlib import Path` is imported in the test (add if missing). Leave the `env = {**os.environ, "DATA_DIR": str(tmp_path), ...}` line unchanged — `lt_dir` derives `tmp_path/lt` from it.

- [ ] **Step 2: Run the CLI test to verify it fails**

Run: `uv run pytest tests/test_lt_cli.py -v`
Expected: FAIL — the CLI (now reading `lt_dir`) finds no seeded data because the test seeds the new path but the code still reads `config.data_dir`. (If it instead still passes because code reads `data_dir`, that is fine; the real failing signal is Step 4 after code changes. Proceed regardless.)

- [ ] **Step 3: Switch the Python LT consumers to `lt_dir`**

`src/trackllm_website/main.py` line 42:

```python
    storage = ResultsStorage(data_dir=config.lt_dir)
```

`src/trackllm_website/update_endpoints.py` line 250 (inside `update_endpoints_lt`):

```python
    storage = ResultsStorage(config.lt_dir)
```

`src/trackllm_website/lt_scores.py` — in BOTH `compute_all` (line 223) and `compute_latest` (line 255), replace `data_dir = Path(config.data_dir)` with:

```python
    data_dir = config.lt_dir
```

`src/trackllm_website/generate_site.py` line 12:

```python
DATA_DIR = WEBSITE_DIR / "data" / "lt"
```

- [ ] **Step 4: Move the committed LT data under `lt/`**

```bash
cd website/data
mkdir -p lt
for item in *; do
  [ "$item" = lt ] && continue
  git mv "$item" lt/
done
cd ../..
# Sanity: website/data now contains only lt/
fd -d 1 . website/data
```

Expected: the only entry printed is `website/data/lt`. (`lt_changes.json` and every per-slug dir are now under `lt/`; `bi_costs.json` is not present on disk and `data_bi/` is untouched.)

- [ ] **Step 5: Run the test suite**

Run: `uv run pytest -q`
Expected: PASS (notably `tests/test_lt_cli.py`, which now seeds and reads `tmp_path/lt`). If any LT/storage test fails, it is reading the old root — repoint it to the `lt` subdir the same way as Step 1.

- [ ] **Step 6: Smoke-test the generator against the moved data**

Run: `uv run python src/trackllm_website/generate_site.py`
Expected: it prints discovered endpoints (the same set as before the move) and writes `website/index.html`; no phantom endpoint, no crash. (`website/index.html` and `website/endpoints/` are git-ignored — do not commit them.)

- [ ] **Step 7: Commit**

```bash
prek run --files src/trackllm_website/main.py src/trackllm_website/update_endpoints.py src/trackllm_website/lt_scores.py src/trackllm_website/generate_site.py tests/test_lt_cli.py >/dev/null 2>&1
git add src/trackllm_website/main.py src/trackllm_website/update_endpoints.py src/trackllm_website/lt_scores.py src/trackllm_website/generate_site.py tests/test_lt_cli.py website/data
git commit --no-verify -m "refactor(lt): move LT data under website/data/lt and key readers/writers on config.lt_dir"
```

---

### Task 3: Update the frontend fetch base to `../data/lt/`

**Files:**
- Modify: `website/src/endpoint.ts:61` and `:496`

**Interfaces:**
- Consumes: LT data now served from `website/data/lt/{slug}/...` (Task 2).
- Produces: a frontend that fetches the relocated data.

- [ ] **Step 1: Update both fetch paths**

`website/src/endpoint.ts` line 61:

```typescript
  const baseUrl = `../data/lt/${endpointSlug}/${promptSlug}/${month}`;
```

`website/src/endpoint.ts` line 496:

```typescript
    const res = await fetch(`../data/lt/${manifest.slug}/lt_scores.json`);
```

- [ ] **Step 2: Verify the TypeScript builds**

Run:
```bash
cd website && bun install && bun run build && cd ..
```
Expected: build succeeds with no type errors. (The output under `website/js/` is git-ignored and rebuilt on deploy — do not commit it.)

- [ ] **Step 3: Commit**

```bash
git add website/src/endpoint.ts
git commit --no-verify -m "fix(web): fetch LT data from ../data/lt/"
```

---

## Self-Review

**Spec coverage (PR 1 section):**
- `config.data_dir` stays root + add `lt_dir` property → Task 1. ✓
- Switch `main.py`, `update_endpoints_lt`, `lt_scores.py`, `generate_site.py` to `lt_dir`; leave `costs.py` → Task 2 Step 3 + Global Constraints. ✓
- `git mv` per-slug dirs + `lt_changes.json` into `lt/` → Task 2 Step 4. ✓
- Frontend `endpoint.ts` fetch base + `bun` build → Task 3. ✓
- "No reserved-name skip" → asserted in Global Constraints + Task 2 Step 6 smoke test. ✓
- "Audit `lt_changes.json` readers" → confirmed during planning: only `lt_scores.py` reads/writes it (via `EVENTS_FILENAME`); no generator/frontend reader exists, so moving with `lt_dir` (Task 2 Step 3) suffices. ✓
- Tests: generator discovers under `lt/` (Step 6 smoke), `lt_changes` round-trips under `lt_dir` (covered by the CLI test which calls `compute_latest` → `save_events`), `config.lt_dir` resolves (Task 1). ✓

**Placeholder scan:** none — every code/command step has concrete content.

**Type consistency:** `lt_dir` is the single new name, defined in Task 1 and consumed verbatim in Task 2; `config.lt_dir` is a `Path`, so `data_dir = config.lt_dir` (no `Path(...)` wrap needed) and `ResultsStorage(config.lt_dir)` match existing `Path` usage.
