# Frontend redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the TrackLLM site (Overview, Endpoint, Model pages + a restyled Spend) on a new design system with a unified drift-from-reference signal, in one PR.

**Architecture:** Keep the static stack — Python/Jinja generates HTML, TypeScript (Bun) runs the interactivity, generated data is fetched as JSON at runtime. Three **validated mock files** in `docs/superpowers/mocks/{overview,endpoint,model}.html` are the source of truth for markup, CSS, and interaction code; each task ports one out of its inlined form into the build (CSS → `style.css`, `<body>` → a Jinja template, the `<script>` → a TS module that `fetch`es generated JSON instead of reading an inlined blob). New Python in `generate_site/` emits that JSON. Design decisions and data semantics: `docs/superpowers/specs/2026-07-27-frontend-redesign-design.md`.

**Tech Stack:** Python 3.13 (Jinja2, pydantic, numpy, orjson), TypeScript + Bun, Plotly/inline SVG, pytest.

## Global Constraints

- All Python via `uv` (`uv run pytest`, `uv run python -m trackllm_website.generate_site`); frontend via `bun` (`cd website && bun run build`). Full build: `make build`.
- After editing code run `prek run --all-files`; commit with `git commit --no-verify` (stale `pre-commit` shim). Revert `uv.lock` churn from `uv run` (`git checkout uv.lock`).
- Single source of truth for every parameter (top-level constant or config); no duplicated default args.
- Comments ~10% of the urge; match existing style; prefer small focused modules.
- **Copy is fixed** — use the blog/paper language verbatim, per spec §1: headline "LLM API outputs are unstable over time."; "LLM APIs are opaque black boxes, even for open-weight models"; "undisclosed changes"; "logprob tracking (LT)"; "black-box border input tracking (B3IT)"; "Small changes are fine, but they should be disclosed." Never claim a *cause*.
- **Drift is display-only**; never change detection (`compute_statistics`, `detect_changes`) or its parameters.
- LT precedes B3IT (badge order, corroboration wording).
- Both themes (light/dark) styled at the token level; the mocks already do this — preserve it.

## Porting method (applies to every UI task)

Each mock file is one self-contained HTML page: `<style>` block, `<body>` markup, and a `<script id="site-data">` data blob + a `<script>` render module. To port one:

1. **CSS** → append/merge its `:root` tokens + component rules into `website/style.css` (Task 3 does this once for shared pieces; page-specific rules go with their page).
2. **Markup** → the static shell (`<nav>`, section headers, empty mount points) becomes a Jinja template; the render module fills the dynamic parts client-side.
3. **Data** → the inlined `site-data` JSON is replaced by a `fetch()` of a file that `generate_site` writes (Task 2). The **data contract** (the JSON shape each page reads) is specified per task and must match what the generator emits.
4. **Logic** → the mock's render functions (`render()`, `strip()`, `lane()`, `volGrid()`, etc.) move verbatim into a `.ts` module, typed, reading the fetched data.

Verification for UI tasks is visual, via the run/verify skill: `make build` then load the page and confirm it matches the mock.

---

## File structure

- **Create:** `src/trackllm_website/lt_drift.py`, `tests/test_lt_drift.py` (Task 1)
- **Create:** `src/trackllm_website/generate_site/overview.py`, `model.py`; `tests/test_generate_site_overview.py`, `tests/test_generate_site_model.py` (Task 2)
- **Modify:** `src/trackllm_website/lt_scores.py` (drift fields, Task 1)
- **Modify:** `src/trackllm_website/generate_site/render.py` (emit overview.json, render model pages, Tasks 2/7)
- **Rewrite:** `website/style.css` (Task 3); `website/templates/base.html.j2`, `index.html.j2`, `endpoint.html.j2`, `spend.html.j2` (Tasks 4–6, 8); **create** `website/templates/model.html.j2` (Task 7)
- **Create/rewrite:** `website/src/overview.ts`, `endpoint.ts`, `model.ts`, `spend.ts`; **modify** `website/package.json` build entrypoints (Tasks 5–8)
- **Reference (read-only):** `docs/superpowers/mocks/{overview,endpoint,model}.html`

---

### Task 0: Branch and land docs + mocks

- [ ] **Step 1:** Isolated worktree off `origin/main` (via superpowers:using-git-worktrees), `.env` symlinked, branch `feat/frontend-redesign`.
- [ ] **Step 2:** Commit the design doc, this plan, and the mock references.

```bash
git add docs/superpowers/specs/2026-07-27-frontend-redesign-design.md \
        docs/superpowers/plans/2026-07-27-frontend-redesign.md \
        docs/superpowers/mocks/
git commit --no-verify -m "docs: frontend redesign design, plan, and validated mocks"
```

---

### Task 1: LT drift-from-reference in the pipeline

Adds the unified signal's LT half. Full detail below; this is one task of the PR.

**Files:** Create `src/trackllm_website/lt_drift.py`, `tests/test_lt_drift.py`; Modify `src/trackllm_website/lt_scores.py`.

**Interfaces — Produces:** `compute_drift_series(observations: list[tuple[datetime, dict[str,float]]]) -> list[tuple[datetime, float]]`; `LTScores.drift_dates: list[datetime]`, `LTScores.drift: list[float]` (default empty).

- [ ] **Step 1: Failing unit tests** — `tests/test_lt_drift.py`:

```python
from datetime import datetime, timedelta, timezone
from trackllm_website.lt_drift import compute_drift_series

def _obs(day, dist): return datetime(2026,1,1,12,tzinfo=timezone.utc)+timedelta(days=day), dist

def test_too_few_days_returns_empty():
    assert compute_drift_series([_obs(0,{"A":-0.1}), _obs(1,{"A":-0.1})]) == []

def test_stable_series_stays_near_zero():
    s = compute_drift_series([_obs(d,{"A":-0.02,"B":-4.0}) for d in range(30)])
    assert len(s) == 30 and max(v for _,v in s) < 0.05

def test_sustained_shift_raises_drift():
    stable = [_obs(d,{"A":-0.02,"B":-4.0}) for d in range(15)]
    shifted = [_obs(d,{"A":-4.0,"B":-0.02}) for d in range(15,30)]
    s = compute_drift_series(stable+shifted)
    assert max(v for dt,v in s if dt.day<=10 and dt.month==1) < 0.3
    assert min(v for dt,v in s if dt>=datetime(2026,1,25,tzinfo=timezone.utc)) > 1.0

def test_unsorted_input_is_handled():
    s = compute_drift_series([_obs(d,{"A":-0.02}) for d in reversed(range(5))])
    assert [dt.day for dt,_ in s] == [1,2,3,4,5]
```

- [ ] **Step 2:** `uv run pytest tests/test_lt_drift.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `src/trackllm_website/lt_drift.py`:**

```python
"""Drift-from-reference for LT: distance of daily behaviour from a baseline period.

Display-only companion to the change-detection statistic in lt_scores.py: the LT
analogue of B3IT total variation. 0 while the endpoint matches its reference
period; rises and stays elevated after a real change.
"""
import statistics
from collections import defaultdict
from datetime import datetime, timezone

REFERENCE_DAYS = 14
LOGPROB_FLOOR = -30.0
SMOOTH_WINDOW = 5


def _mean_vector(dicts, extra_tokens):
    floor = min(min(d.values()) for d in dicts)
    tokens = {t for d in dicts for t in d} | extra_tokens
    return {t: statistics.mean([d.get(t, floor) for d in dicts]) for t in tokens}, floor


def compute_drift_series(observations):
    obs = sorted((dt, {t: max(LOGPROB_FLOOR, v) for t, v in d.items()}) for dt, d in observations if d)
    if len({dt.date() for dt, _ in obs}) < 3:
        return []
    start = obs[0][0]
    ref_dicts = [d for dt, d in obs if (dt - start).days < REFERENCE_DAYS]
    ref_tokens = {t for d in ref_dicts for t in d}
    ref_mean, ref_floor = _mean_vector(ref_dicts, ref_tokens)
    by_day = defaultdict(list)
    for dt, d in obs:
        by_day[dt.date()].append(d)
    raw = []
    for day in sorted(by_day):
        day_mean, day_floor = _mean_vector(by_day[day], ref_tokens)
        floor = min(day_floor, ref_floor)
        tokens = set(day_mean) | set(ref_mean)
        drift = statistics.mean(abs(day_mean.get(t, floor) - ref_mean.get(t, floor)) for t in tokens)
        raw.append((datetime(day.year, day.month, day.day, tzinfo=timezone.utc), drift))
    vals = [v for _, v in raw]
    half = SMOOTH_WINDOW // 2
    return [(dt, round(statistics.median(vals[max(0, i - half):i + half + 1]), 4)) for i, (dt, _) in enumerate(raw)]
```

- [ ] **Step 4:** `uv run pytest tests/test_lt_drift.py -v` → PASS.

- [ ] **Step 5: Wire into `lt_scores.py`.** Add `from pydantic import BaseModel, Field` and `from trackllm_website.lt_drift import compute_drift_series`. Add to `LTScores`:

```python
    drift_dates: list[datetime] = Field(default_factory=list)
    drift: list[float] = Field(default_factory=list)
```

In `compute_endpoint_scores`, keep each prompt's full observations and compute drift on the longest prompt:

```python
    per_prompt_data: list[list[tuple[datetime, dict]]] = []
    # ... inside the loop, after per_prompt_dates.append(...):
        per_prompt_data.append(data)
    # ... after `changes, sigmas = detect_changes(avg_scores)`:
    drift_series = compute_drift_series(per_prompt_data[longest])
    return LTScores(
        n_per_test=N_PER_TEST, dates=ref_dates, scores=avg_scores.tolist(),
        sigmas=[normalize_sigma(v) for v in sigmas.tolist()], changes=changes,
        drift_dates=[dt for dt, _ in drift_series], drift=[v for _, v in drift_series],
    )
```

- [ ] **Step 6: Integration test** in `tests/test_lt_scores.py` (create if absent):

```python
from datetime import datetime, timedelta, timezone
import trackllm_website.lt_scores as lt_scores
from trackllm_website.lt_scores import compute_endpoint_scores

def test_compute_endpoint_scores_populates_drift(tmp_path, monkeypatch):
    ep = tmp_path/"endpoint"; prompt = ep/"prompt1"; prompt.mkdir(parents=True)
    (prompt/"info.json").write_text("{}")
    base = datetime(2026,1,1,12,tzinfo=timezone.utc)
    data = [(base+timedelta(days=day,hours=k), {"A":-0.02,"B":-4.0} if day<15 else {"A":-4.0,"B":-0.02})
            for day in range(30) for k in range(4)]
    monkeypatch.setattr(lt_scores, "load_prompt_logprobs", lambda _dir: data)
    s = compute_endpoint_scores(ep)
    assert s is not None and len(s.drift) == len(s.drift_dates) > 0
    assert s.drift[0] < 0.3 and max(s.drift) > 1.0
```

- [ ] **Step 7:** `uv run pytest tests/test_lt_scores.py tests/test_lt_drift.py -q` → PASS; `prek run --all-files`; `git checkout uv.lock`; commit `feat: LT drift-from-reference in lt_scores.json`.

---

### Task 2: Data layer — generate the JSON the frontend reads

Emit two artifacts from data already on disk (`lt_scores.json` now carries drift, `b3it.json` carries tv_series/changes, `changes.json`, `spend.json`): a site-wide **`overview.json`** and per-model **`models/<slug>.json`**. These read the *existing* generated files — no raw logprob re-reads.

**Files:** Create `src/trackllm_website/generate_site/overview.py`, `model.py` + tests; Modify `render.py` to call them.

**Interfaces — Produces (data contracts consumed by Tasks 5 & 7):**

```
overview.json = {
  stats: {endpoints, active, providers, models, orgs, changes_total, changes_lt,
          changes_b3it, changed_endpoints, queries, since, spend_cumulative,
          lt_endpoints, b3it_endpoints, b3it_monitoring, b3it_since},
  feed: [{date, iso, daysAgo, model, provider, method:"lt"|"b3it",
          primary, secondary, sevKey:"alert"|"changed"|"stable", desc, trace:[float], changeFrac}],
  providers: [{name, n_endpoints, endpoint_years, months, n_changes, rate, conf}],
  endpoints: [{slug, model, org, provider, methods:[str], status:"stable"|"changed"|"retired",
               stableDays:int|null, nChanges:int, trace:[float]}]  # trace = downsampled lt drift
}
models/<modelSlug>.json = {
  model, org, date_min, date_max, n_providers, n_changed,
  endpoints: [{slug, provider, methods, first, last, n_changes,
               lt:{drift:[[date,val]], changes:[{date,sigma,drift}]}|null,
               b3it:{tv:[[date,val]], changes:[{date,peakTV}]}|null}]
}
```

The mocks' inlined `site-data` blobs are worked examples of these exact shapes — match them field-for-field so the ported render code works unchanged.

- [ ] **Step 1: Failing test** `tests/test_generate_site_overview.py` — build a fixture with one LT `lt_scores.json` (with a `drift` array and a change) and assert `build_overview(...)` returns the contract:

```python
from datetime import datetime, timezone
from trackllm_website.generate_site.overview import build_overview, downsample_trace

def test_downsample_trace_caps_length():
    assert len(downsample_trace(list(range(200)), 28)) == 28

def test_build_overview_shape(fake_site):  # fixture writes minimal lt/b3it/changes/spend data
    ov = build_overview(fake_site)
    assert set(ov) == {"stats", "feed", "providers", "endpoints"}
    ep = ov["endpoints"][0]
    assert set(ep) >= {"slug","model","provider","methods","status","nChanges","trace"}
    assert ov["stats"]["changes_total"] == ov["stats"]["changes_lt"] + ov["stats"]["changes_b3it"]
```

(Provide the `fake_site` fixture in the test: a `tmp_path/website/data` tree with one `lt/<slug>/lt_scores.json` containing `dates`, `scores`, `changes`, `drift`, `drift_dates`; one `b3it/<slug>/b3it.json`; a `changes.json`; a `spend.json`. Use the mock's field names.)

- [ ] **Step 2:** `uv run pytest tests/test_generate_site_overview.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `generate_site/overview.py`.** Port the aggregation logic from the reference extractor (validated): read each endpoint's `lt_scores.json` (`drift`/`drift_dates`/`changes`/`dates`) and B3IT view; derive per-endpoint `status` (retired if last obs >14d before the newest observation across the fleet; changed if a change within 60d; else stable), `stableDays`, `nChanges` (LT + B3IT), and `trace = downsample_trace(drift, 28)`; the change **feed** (6 latest LT with drift-level magnitude + σ, 4 latest B3IT with peak TV, merged by date, each pre-formatted into `primary`/`secondary`/`sevKey`/`desc`); the **provider drift rate** (changes per endpoint-year, `conf = 0.3 + 0.7*min(1, ey/4)`, include zero-change providers); and `stats`. Constants (`REFERENCE window not needed here`, `RECENT_CHANGE_DAYS=60`, `RETIRED_GAP_DAYS=14`, feed sizes) top-level. Full field list per the contract above; the reference implementation is `docs/superpowers/mocks/overview.html`'s data blob + the semantics in spec §6.1.

- [ ] **Step 4:** `uv run pytest tests/test_generate_site_overview.py -v` → PASS.

- [ ] **Step 5: Implement `generate_site/model.py`** (`build_model_views(site) -> dict[modelSlug, dict]`), grouping endpoints by `model`, and for each provider-endpoint reading its `lt_scores.json` drift/changes and `b3it.json` tv/changes into the `models/<slug>.json` contract; smooth already applied upstream for LT drift, B3IT tv from the view. Test `tests/test_generate_site_model.py`: one model with two providers → one model view with 2 endpoints, `n_changed` correct, shared `date_min/date_max`.

- [ ] **Step 6:** `uv run pytest tests/test_generate_site_model.py -v` → PASS.

- [ ] **Step 7: Wire into `render.py`.** After the existing derivations, write `overview.json` and per-model files:

```python
from trackllm_website.generate_site import overview as overview_mod
from trackllm_website.generate_site import model as model_mod
# ...
(website_dir/"data"/"overview.json").write_text(json.dumps(overview_mod.build_overview(website_dir)))
models_dir = website_dir/"data"/"models"; models_dir.mkdir(parents=True, exist_ok=True)
model_views = model_mod.build_model_views(website_dir)
for mslug, view in model_views.items():
    (models_dir/f"{mslug}.json").write_text(json.dumps(view))
```

- [ ] **Step 8:** `prek run --all-files`; `git checkout uv.lock`; commit `feat: generate overview.json and per-model data`.

---

### Task 3: Design system (`style.css`)

**Files:** Rewrite `website/style.css`.

- [ ] **Step 1:** Replace `style.css` with the shared token system + components from the mocks: the `:root` custom properties and both-theme overrides (`@media (prefers-color-scheme: dark)` + `:root[data-theme=…]`), plus the shared components used across pages — `.nav`, `.brand`, `.badge`(.lt/.b3it), `.pill`(.stable/.changed/.retired), `.crumb`, `.wrap`, `.footnote`, `footer.site`. Copy these blocks verbatim from `docs/superpowers/mocks/overview.html` and `endpoint.html` (they are identical across mocks). Page-specific rules are added by their page tasks.
- [ ] **Step 2: Verify** `make build` succeeds and any existing page renders with the new tokens (dark + light via the OS/theme toggle). Commit `feat: design-system tokens and shared components`.

---

### Task 4: Base template + nav

**Files:** Modify `website/templates/base.html.j2`.

- [ ] **Step 1:** Put the sticky `<nav>` (brand glyph + links `Overview · Spend` + external Methodology→blog / GitHub / Paper, per spec §5) and the `data-theme` toggle hook into `base.html.j2`; keep the goatcounter snippet. Blocks: `title`, `content`, `scripts`.
- [ ] **Step 2: Verify** `make build`; nav appears, links resolve. Commit `feat: base template + nav`.

---

### Task 5: Overview page

**Files:** Rewrite `website/templates/index.html.j2`, create `website/src/overview.ts`, add `overview.ts` to `website/package.json` build entrypoints. Reference: `docs/superpowers/mocks/overview.html`.

**Interfaces — Consumes:** `data/overview.json` (Task 2 contract).

- [ ] **Step 1:** `index.html.j2` extends base; body = the mock's static shell: hero (headline + lede + telemetry mount + cap), "Latest detected changes" feed mount, "Provider drift rate" panel mount, "Endpoints" toolbar (search + chips) + table shell with `<tbody id="dirBody">`. Copy the overview-specific CSS blocks (`.hero`, `.telemetry`, `.feed`/`.event`, `.rate-*`, `.vol`/grid, `.toolbar`/`.chip`, `table.dir`, sparkline) into `style.css`.
- [ ] **Step 2:** `overview.ts` = the mock's `<script>` module, typed, changed only to `const DATA = await (await fetch("data/overview.json")).json()` instead of reading the inlined `#site-data`. Keep `sparkPath` (directory sparkline fixed domain `[0,1.5]`), telemetry, feed, provider `volGrid`, and the directory `render()` (search + chips + sortable columns).
- [ ] **Step 3: Verify (run/verify skill):** `make build`, open `/index.html`; check headline reads "LLM API outputs are unstable over time.", the directory search/filter/sort work, drift sparklines are flat-low for stable rows, feed shows LT+B3IT with drift/TV. Compare against the mock.
- [ ] **Step 4:** `prek run --all-files`; commit `feat: overview page`.

---

### Task 6: Endpoint page

**Files:** Rewrite `website/templates/endpoint.html.j2`, rewrite `website/src/endpoint.ts`. Reference: `docs/superpowers/mocks/endpoint.html`.

**Interfaces — Consumes:** the endpoint's `lt_scores.json` (now with `drift`/`drift_dates`/`changes`) and `b3it.json` (tv_series/changes); model+provider from the manifest.

- [ ] **Step 1:** Template shell: breadcrumb, header + status card mount, compare-providers banner (link to `models/<modelSlug>.html`), the stacked-lane raw-signal panel mount, detected-changes table mount, methodology cards (static). Copy endpoint-specific CSS.
- [ ] **Step 2:** `endpoint.ts` = the mock's module, typed, fetching `lt_scores.json` + `b3it.json` for this slug (build the `focus`-shaped object the mock renders: `lt.drift` from `drift_dates`+`drift`, `lt.changes` with drift level + σ, `b3it.tv`/`changes`). Keep the `lane()` stacked-axis chart, status card, changes table.
- [ ] **Step 3: Verify:** `make build`, open `endpoints/deepseek2fdeepseek-chat-v3-032423fireworks.html`; LT-drift + B3IT-TV lanes share one time axis, changepoints land on visible rises, banner links to the model page. Compare to mock.
- [ ] **Step 4:** `prek run --all-files`; commit `feat: endpoint page with unified drift panel`.

---

### Task 7: Model page

**Files:** Create `website/templates/model.html.j2`, `website/src/model.ts`; modify `render.py` to render one page per model; add `model.ts` entrypoint. Reference: `docs/superpowers/mocks/model.html`.

**Interfaces — Consumes:** `data/models/<modelSlug>.json` (Task 2 contract).

- [ ] **Step 1:** In `render.py`, after building `model_views` (Task 2), render a `models/<modelSlug>.html` per model from `model.html.j2` (passing `model_slug`, `model`, `org`); create the `models/` output dir and clear stale files like the endpoints loop does.
- [ ] **Step 2:** `model.html.j2` shell: breadcrumb, header + summary mount, "Drift by provider" section with the `#cmp` mount + legend. Copy model-specific CSS (`.cmp`, `.summary`).
- [ ] **Step 3:** `model.ts` = the mock's module, typed, `fetch`ing `../data/models/<slug>.json` (derive slug from a `<script>`-injected value or the manifest). Keep the shared-timeline `strip()` with the shared per-method y-scale (`LT_MAX`) and change dots.
- [ ] **Step 4: Verify:** `make build`, open `models/deepseek2fdeepseek-chat-v3-0324.html`; 11 providers on one shared timeline, shared per-method scale, dots on changes, rows link to endpoint pages.
- [ ] **Step 5:** `prek run --all-files`; commit `feat: model page (cross-provider comparison)`.

---

### Task 8: Spend page restyle

**Files:** Rewrite `website/templates/spend.html.j2`, adjust `website/src/spend.ts`.

- [ ] **Step 1:** Re-skin the existing spend page into the design system (tokens, nav, tables/cards); no data changes. Keep the Plotly daily-spend chart, restyled to the palette.
- [ ] **Step 2: Verify** `make build`, open `/spend.html`. Commit `feat: restyle spend page`.

---

### Task 9: Full build, verify, PR

- [ ] **Step 1:** `make build` clean from scratch (`make clean && make build`); no errors; all four page types generated.
- [ ] **Step 2:** `uv run pytest -q` all pass; `git checkout uv.lock`.
- [ ] **Step 3: Verify skill** end-to-end: click Overview → model → endpoint → back; theme toggle works on every page; search/filter/sort on Overview; both themes legible.
- [ ] **Step 4:** Push `feat/frontend-redesign`; open PR referencing the spec. Note in the body that per-endpoint drift appears after the next LT pipeline recompute / a `compute_all` backfill (this PR is code only).

---

## Self-review

- **Spec coverage:** design system §3 → T3; unified drift §4 → T1 (LT) + T2 (surfaced) + T6 (rendered); IA/nav §5 → T4; Overview §6.1 → T2/T5; Endpoint §6.2 → T6; Model §6.3 → T2/T7; Spend §6.4 → T8; MVP scope §10 (Provider/Changes/Rankings excluded; Methodology = external link) → honored. Copy rules → Global Constraints + per-task.
- **Placeholder scan:** the UI tasks intentionally reference the committed mock files as the implementation source (a real artifact in the repo, not a "TBD"); all novel logic (drift, data generation) has complete code + TDD. Data contracts are specified field-for-field.
- **Type consistency:** `overview.json` / `models/<slug>.json` field names in Task 2's contract match the mock blobs the render modules (T5/T7) consume; `compute_drift_series` signature and `LTScores.drift`/`drift_dates` match across T1's tests, model, and Task 2's readers.
- **Ordering:** T1 (drift in data) precedes T2 (reads drift) precedes T5/T6/T7 (render it); T3 (tokens) precedes all UI; T4 (base) precedes pages.
