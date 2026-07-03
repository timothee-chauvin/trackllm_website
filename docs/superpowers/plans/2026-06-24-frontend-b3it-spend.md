# Frontend for B3IT, change feed, and spend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the B3IT change-detection state, a merged LT+B3IT change feed, and the actual-spend ledger on the static website.

**Architecture:** Refactor the monolithic `generate_site.py` into a `generate_site/` package. All display values are **derived at build time** in Python (TV series via `bi.detection`, spend aggregates via `spend.py`), emitted as small per-page JSON the TypeScript fetches and renders with Plotly. Backend data formats are untouched; the frontend only consumes what PRs 1–17 already emit.

**Tech Stack:** Python 3 (Jinja2, pydantic), TypeScript + Plotly (bundled with `bun`), `uv` for all Python commands, `pytest`.

## Global Constraints

- All Python commands run via `uv` (e.g. `uv run pytest`, `uv run python -m trackllm_website.generate_site`).
- Tests first (write failing test → implement → green), per project practice.
- Use `slugify` from `util.py` for any new on-disk name; B3IT state and LT data dirs already share the `slugify(f"{model}#{provider}")` scheme, so the union is keyed by slug.
- Plots use Plotly only.
- "lt" = logprob tracking; "bi"/"B3IT" = border inputs.
- Derived display values are recomputed at build time with the current algorithm; never persist them in state. Empty/sparse data (reference-less migrated epochs, ledger with only `vetting` rows) is a **normal** state — render gracefully, never error.
- After editing code, run `prek run --all-files`. The git pre-commit hook is misconfigured (`pre-commit` not installed; project uses `prek`); commit code with `prek` run manually and `git commit --no-verify` when the hook blocks.
- Status vocabulary normalized in the UI: a method shows **monitoring** when actively querying the endpoint (LT: last query within `INACTIVE_THRESHOLD_DAYS=3`; B3IT: `status == "monitoring"`), else **retired** (B3IT carries the `retired.reason`); `—` when the method does not cover the endpoint.

**Kind→group mapping (used in PR 4):** `onboard|recheck|reinit → "onboarding"`, `monitor → "monitoring"`, `lt → "lt"`, `vetting → "vetting"`.

---

## PR 1 — Build refactor: `generate_site.py` → `generate_site/` package (no visible change)

Pure refactor. LT output (index.html + endpoint pages) must be byte-identical. This PR adds the first generator tests, which lock current behavior.

### File structure after PR 1

- Create `src/trackllm_website/generate_site/__init__.py`
- Create `src/trackllm_website/generate_site/lt.py` — LT discovery (moved from `generate_site.py`)
- Create `src/trackllm_website/generate_site/render.py` — Jinja rendering + file emission
- Create `src/trackllm_website/generate_site/__main__.py` — orchestration entrypoint
- Delete `src/trackllm_website/generate_site.py`
- Modify `Makefile` (build target)
- Create `tests/test_generate_site_lt.py`
- Create `tests/test_generate_site_render.py`

### Task 1: LT discovery module

**Files:**
- Create: `src/trackllm_website/generate_site/__init__.py` (empty)
- Create: `src/trackllm_website/generate_site/lt.py`
- Test: `tests/test_generate_site_lt.py`

**Interfaces:**
- Produces:
  - `INACTIVE_THRESHOLD_DAYS: int = 3`
  - `@dataclass PromptInfo(slug: str, prompt: str, months: list[str])`
  - `@dataclass EndpointInfo(model, provider, slug, prompts, last_query_date)` with `.is_active: bool` and `.last_query_str: str` properties (moved verbatim from `generate_site.py:20-50`)
  - `get_last_query_date(endpoint_dir: Path) -> datetime | None` (verbatim from `generate_site.py:53-118`)
  - `get_endpoint_info(endpoint_dir: Path) -> EndpointInfo | None` (verbatim from `generate_site.py:121-167`)
  - `discover_lt_endpoints(lt_dir: Path) -> list[EndpointInfo]` — iterate `sorted(lt_dir.iterdir())`, call `get_endpoint_info`, collect non-None (extracted from `main()` loop at `generate_site.py:187-193`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate_site_lt.py
import json
from pathlib import Path

from trackllm_website.generate_site.lt import discover_lt_endpoints


def _make_lt_endpoint(root: Path, slug: str, model: str, provider: str):
    d = root / slug / "default"
    d.mkdir(parents=True)
    (d / "info.json").write_text(
        json.dumps({"prompt": "hi", "endpoint": {"model": model, "provider": provider}})
    )
    md = d / "2026-06"
    md.mkdir()
    (md / "queries.json").write_text(json.dumps([["24 10:00:00", 0]]))


def test_discover_lt_endpoints(tmp_path):
    _make_lt_endpoint(tmp_path, "m2fa23p", "m/a", "p")
    eps = discover_lt_endpoints(tmp_path)
    assert len(eps) == 1
    assert eps[0].model == "m/a"
    assert eps[0].provider == "p"
    assert eps[0].prompts[0].months == ["2026-06"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate_site_lt.py -v`
Expected: FAIL with `ModuleNotFoundError: trackllm_website.generate_site.lt`

- [ ] **Step 3: Create the module**

Create `src/trackllm_website/generate_site/__init__.py` (empty). Create `src/trackllm_website/generate_site/lt.py`: move `PromptInfo`, `EndpointInfo`, `get_last_query_date`, `get_endpoint_info`, and `INACTIVE_THRESHOLD_DAYS` **verbatim** from `generate_site.py` (lines 16–167), keeping the `import json`, `dataclass`, and `datetime/timedelta/timezone` imports. Append:

```python
def discover_lt_endpoints(lt_dir: Path) -> list[EndpointInfo]:
    endpoints: list[EndpointInfo] = []
    for endpoint_dir in sorted(lt_dir.iterdir()):
        if not endpoint_dir.is_dir():
            continue
        info = get_endpoint_info(endpoint_dir)
        if info:
            endpoints.append(info)
    return endpoints
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate_site_lt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uv run prek run --all-files || true
git add src/trackllm_website/generate_site/__init__.py src/trackllm_website/generate_site/lt.py tests/test_generate_site_lt.py
git commit --no-verify -m "refactor: extract LT discovery into generate_site.lt"
```

### Task 2: Render + emit module and orchestration entrypoint

**Files:**
- Create: `src/trackllm_website/generate_site/render.py`
- Create: `src/trackllm_website/generate_site/__main__.py`
- Delete: `src/trackllm_website/generate_site.py`
- Modify: `Makefile` (build target)
- Test: `tests/test_generate_site_render.py`

**Interfaces:**
- Consumes: `discover_lt_endpoints`, `EndpointInfo` (Task 1)
- Produces:
  - `render_site(website_dir: Path) -> None` — full build: discover LT endpoints, split active/inactive (sorted by `model.lower()`), render `index.html` and per-endpoint pages, exactly as today's `main()` (`generate_site.py:170-242`). Reads templates from `website_dir / "templates"`, LT data from `website_dir / "data" / "lt"`, writes `index.html` and `endpoints/<slug>.html`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate_site_render.py
import json
import shutil
from pathlib import Path

from trackllm_website.generate_site.render import render_site


def _scaffold(website: Path):
    # copy real templates + style so rendering matches production
    src = Path("website")
    (website / "templates").mkdir(parents=True)
    for t in (src / "templates").glob("*.j2"):
        shutil.copy(t, website / "templates" / t.name)
    (website / "style.css").write_text((src / "style.css").read_text())
    ep = website / "data" / "lt" / "m2fa23p" / "default"
    ep.mkdir(parents=True)
    ep_info = {"prompt": "hi", "endpoint": {"model": "m/a", "provider": "p"}}
    (ep / "info.json").write_text(json.dumps(ep_info))
    md = ep / "2026-06"
    md.mkdir()
    (md / "queries.json").write_text(json.dumps([["24 10:00:00", 0]]))


def test_render_site_produces_index_and_endpoint(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path)
    index = (tmp_path / "index.html").read_text()
    assert "m/a" in index
    assert (tmp_path / "endpoints" / "m2fa23p.html").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate_site_render.py -v`
Expected: FAIL with `ModuleNotFoundError: trackllm_website.generate_site.render`

- [ ] **Step 3: Create render.py and __main__.py, delete the old module, update Makefile**

`render.py`: move the body of `main()` (`generate_site.py:170-242`) into `render_site(website_dir: Path)`, replacing the module-level `WEBSITE_DIR`/`DATA_DIR`/`ENDPOINTS_DIR`/`TEMPLATES_DIR` constants with locals derived from `website_dir` (`data_dir = website_dir / "data" / "lt"`, `endpoints_dir = website_dir / "endpoints"`, `templates_dir = website_dir / "templates"`). Import `discover_lt_endpoints` from `.lt`. Keep all rendering, manifest JSON, active/inactive split, and cleanup of old `*.html` exactly as-is.

`__main__.py`:
```python
from pathlib import Path

from trackllm_website.generate_site.render import render_site


def main() -> None:
    render_site(Path("website"))


if __name__ == "__main__":
    main()
```

Delete `src/trackllm_website/generate_site.py`. In `Makefile`, change the build line `uv run python src/trackllm_website/generate_site.py` → `uv run python -m trackllm_website.generate_site`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate_site_render.py tests/test_generate_site_lt.py -v`
Expected: PASS

- [ ] **Step 5: Verify full build still works**

Run: `make build`
Expected: completes; `website/index.html` and `website/endpoints/*.html` regenerated. Confirm `git diff --stat website/index.html` shows no content change vs the prior build (LT output identical).

- [ ] **Step 6: Commit**

```bash
uv run prek run --all-files || true
git add -A src/trackllm_website/generate_site tests/test_generate_site_render.py Makefile
git rm src/trackllm_website/generate_site.py
git commit --no-verify -m "refactor: split generate_site into a package (LT output unchanged)"
```

---

## PR 2 — B3IT endpoint section

Derive per-endpoint B3IT display data and render a TV-over-time chart on the endpoint page.

### File structure

- Create `src/trackllm_website/generate_site/b3it.py`
- Modify `src/trackllm_website/generate_site/render.py` (emit `b3it.json`, render endpoint pages for the union)
- Modify `website/src/endpoint.ts` (fetch + render B3IT section)
- Modify `website/templates/endpoint.html.j2` (B3IT section container)
- Create `tests/test_generate_site_b3it.py`

### Task 3: B3IT derivation

**Files:**
- Create: `src/trackllm_website/generate_site/b3it.py`
- Test: `tests/test_generate_site_b3it.py`

**Interfaces:**
- Consumes: `EndpointBIState`, `RetiredInfo` (`bi.state`), `load_all_states` (`bi.state`), `load_phase2_results` (`bi.analyze`), `epoch_tv_series`, `adaptive_transitions`, `is_unstable` (`bi.detection`).
- Produces:
  - `@dataclass B3ITView`:
    - `slug: str`, `model: str`, `provider: str`
    - `status: str` (`"monitoring"` | `"retired"`)
    - `retired_reason: str | None`
    - `n_bis: int`
    - `unstable: bool`
    - `epochs: list[dict]` — each `{"start": iso, "end": iso|None, "end_reason": str|None, "change_date": iso|None}`
    - `tv_series: dict` — `{"dates": list[str], "values": list[float]}` (current epoch; empty when no reference/results)
    - `changes: list[dict]` — `[{"date": iso, "kind": "onset"}]`
  - `current_epoch_results(state: EndpointBIState, results: dict) -> tuple[Epoch | None, dict]` — mirrors `monitor.decide` (monitor.py:54-79): the last epoch with no `end` is current; filter `results` to its `border_inputs` and timestamps `>= epoch.start`.
  - `derive_b3it(state: EndpointBIState, results: dict) -> B3ITView`
  - `to_json(view: B3ITView) -> dict` — the on-disk `b3it.json` payload (status, retired_reason, n_bis, unstable, epochs, tv_series, changes)
  - `discover_b3it_views(state_dir: Path, phase_2_dir: Path) -> dict[str, B3ITView]` — `load_all_states(state_dir)`; for each, `load_phase2_results(phase_2_dir / state.slug)`; map `state.slug -> derive_b3it(...)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generate_site_b3it.py
from datetime import datetime, timezone

from trackllm_website.bi.state import EndpointBIState, RetiredInfo
from trackllm_website.config import Endpoint
from trackllm_website.bi.state import Epoch
from trackllm_website.generate_site.b3it import derive_b3it


def _ep():
    return Endpoint(api="openrouter", model="m/a", provider="p", cost=[0.1, 0.2], max_logprobs=None)


def test_retired_no_reference_yields_empty_tv_but_full_timeline():
    state = EndpointBIState(
        endpoint=_ep(),
        status="retired",
        retired=RetiredInfo(reason="no_bis", since=datetime(2026, 2, 5, tzinfo=timezone.utc),
                            last_recheck=datetime(2026, 2, 5, tzinfo=timezone.utc)),
        epochs=[Epoch(start=datetime(2026, 1, 14, tzinfo=timezone.utc),
                      border_inputs=[], reference={},
                      end=datetime(2026, 2, 5, tzinfo=timezone.utc), end_reason="gap")],
    )
    view = derive_b3it(state, {})
    assert view.status == "retired"
    assert view.retired_reason == "no_bis"
    assert view.tv_series == {"dates": [], "values": []}
    assert len(view.epochs) == 1
    assert view.epochs[0]["end_reason"] == "gap"
    assert view.n_bis == 0


def test_monitoring_with_reference_yields_tv_series(monkeypatch):
    # Two border inputs, a reference batch and one later batch differing from it.
    ref = {"p1": [("2026-06-01T00:00:00Z", "A")] * 10}
    results = {
        "p1": {
            "2026-06-01T00:00:00Z": [("2026-06-01T00:00:00Z", "A")] * 10,
            "2026-06-02T00:00:00Z": [("2026-06-02T00:00:00Z", "B")] * 10,
        }
    }
    state = EndpointBIState(
        endpoint=_ep(), status="monitoring", retired=None,
        epochs=[Epoch(start=datetime(2026, 6, 1, tzinfo=timezone.utc),
                      border_inputs=["p1"], reference=ref)],
    )
    view = derive_b3it(state, results)
    assert view.status == "monitoring"
    assert view.n_bis == 1
    assert view.tv_series["values"]  # non-empty
    assert view.tv_series["values"][0] > 0  # B vs A => TV 1.0
```

(Adjust `Epoch`/`Endpoint`/`ReferenceSamples` construction to the real pydantic field names — read `bi/state.py` and `config.py` first; `reference` is a `ReferenceSamples` type.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_generate_site_b3it.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement b3it.py**

```python
"""Build-time derivation of per-endpoint B3IT display data."""

from dataclasses import dataclass
from pathlib import Path

from trackllm_website.bi.analyze import load_phase2_results
from trackllm_website.bi.detection import (
    adaptive_transitions,
    epoch_tv_series,
    is_unstable,
)
from trackllm_website.bi.state import Epoch, EndpointBIState, load_all_states


@dataclass
class B3ITView:
    slug: str
    model: str
    provider: str
    status: str
    retired_reason: str | None
    n_bis: int
    unstable: bool
    epochs: list[dict]
    tv_series: dict
    changes: list[dict]


def current_epoch_results(state: EndpointBIState, results: dict) -> tuple[Epoch | None, dict]:
    open_epochs = [e for e in state.epochs if e.end is None]
    if not open_epochs:
        return None, {}
    epoch = open_epochs[-1]
    start = epoch.start.isoformat().replace("+00:00", "Z")
    epoch_results = {
        p: {ts: s for ts, s in results.get(p, {}).items() if ts >= start}
        for p in epoch.border_inputs
    }
    return epoch, epoch_results


def _iso(dt) -> str | None:
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def derive_b3it(state: EndpointBIState, results: dict) -> B3ITView:
    epoch, epoch_results = current_epoch_results(state, results)
    tv = epoch_tv_series(epoch.reference, epoch_results) if epoch else []
    changes = adaptive_transitions(tv) if tv else []
    return B3ITView(
        slug=state.slug,
        model=state.endpoint.model,
        provider=state.endpoint.provider,
        status=state.status,
        retired_reason=state.retired.reason if state.retired else None,
        n_bis=len(epoch.border_inputs) if epoch else 0,
        unstable=is_unstable(tv),
        epochs=[
            {"start": _iso(e.start), "end": _iso(e.end),
             "end_reason": e.end_reason, "change_date": _iso(e.change_date)}
            for e in state.epochs
        ],
        tv_series={"dates": [ts for ts, _ in tv], "values": [v for _, v in tv]},
        changes=[{"date": ts, "kind": "onset"} for ts in changes],
    )


def to_json(view: B3ITView) -> dict:
    return {
        "status": view.status,
        "retired_reason": view.retired_reason,
        "n_bis": view.n_bis,
        "unstable": view.unstable,
        "epochs": view.epochs,
        "tv_series": view.tv_series,
        "changes": view.changes,
    }


def discover_b3it_views(state_dir: Path, phase_2_dir: Path) -> dict[str, B3ITView]:
    views: dict[str, B3ITView] = {}
    if not state_dir.exists():
        return views
    for state in load_all_states(state_dir):
        results = load_phase2_results(phase_2_dir / state.slug)
        views[state.slug] = derive_b3it(state, results)
    return views
```

Verify `state.slug` exists (it does — `monitor.py:120` uses `state.slug`) and `epoch_tv_series`'s timestamp comparison matches the `ts >= start` filtering used in `monitor.decide`; align with monitor.py if its filter differs.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_generate_site_b3it.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uv run prek run --all-files || true
git add src/trackllm_website/generate_site/b3it.py tests/test_generate_site_b3it.py
git commit --no-verify -m "feat: derive per-endpoint B3IT display data at build time"
```

### Task 4: Emit b3it.json and render endpoint pages for the union

**Files:**
- Modify: `src/trackllm_website/generate_site/render.py`
- Test: `tests/test_generate_site_render.py` (extend)

**Interfaces:**
- Consumes: `discover_b3it_views`, `to_json`, `B3ITView` (Task 3)
- Produces: in `render_site`, after LT discovery: `b3it_views = discover_b3it_views(website_dir/"data"/"b3it"/"state", website_dir/"data"/"b3it"/"phase_2")`. For each view, write `website_dir/"data"/"b3it"/<slug>/"b3it.json"` = `json.dumps(to_json(view))`. Generate an endpoint page for **every slug in the union** of LT endpoints and B3IT views (B3IT-only endpoints get a manifest with empty `prompts` and the model/provider from the B3IT view).

- [ ] **Step 1: Write the failing test**

```python
def test_render_emits_b3it_json_and_b3it_only_page(tmp_path):
    _scaffold(tmp_path)
    # add a B3IT-only monitoring state with no LT data
    import json
    sd = tmp_path / "data" / "b3it" / "state"
    sd.mkdir(parents=True)
    state = {
        "endpoint": {"api": "openrouter", "model": "b/x", "provider": "q",
                     "cost": [0.1, 0.2], "max_logprobs": None},
        "status": "monitoring", "retired": None,
        "epochs": [{"start": "2026-06-01T00:00:00Z", "border_inputs": [],
                    "reference": {}, "end": None}],
    }
    (sd / "b2fx23q.json").write_text(json.dumps(state))
    from trackllm_website.generate_site.render import render_site
    render_site(tmp_path)
    assert (tmp_path / "data" / "b3it" / "b2fx23q" / "b3it.json").exists()
    assert (tmp_path / "endpoints" / "b2fx23q.html").exists()
```

(Match the real `EndpointBIState` JSON schema — dump a constructed model with `.model_dump_json()` in a scratch REPL if unsure.)

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_generate_site_render.py::test_render_emits_b3it_json_and_b3it_only_page -v`
Expected: FAIL (no `b3it.json` / no page)

- [ ] **Step 3: Wire b3it into render_site**

In `render.py`, after collecting LT `endpoints`: build `lt_by_slug = {e.slug: e for e in endpoints}`, call `discover_b3it_views`, write each `b3it.json` (create parent dirs). Replace the per-LT-endpoint page loop with a union loop over `sorted(set(lt_by_slug) | set(b3it_views))`: for each slug build the manifest from the LT `EndpointInfo` if present (existing manifest code), else `{"model": view.model, "provider": view.provider, "slug": slug, "prompts": []}`. Render `endpoint.html.j2` as today. (The index still renders LT-only here — unified index is PR 3.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_generate_site_render.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uv run prek run --all-files || true
git add src/trackllm_website/generate_site/render.py tests/test_generate_site_render.py
git commit --no-verify -m "feat: emit b3it.json and render endpoint pages for LT∪B3IT union"
```

### Task 5: Endpoint-page B3IT chart (TypeScript + template)

**Files:**
- Modify: `website/templates/endpoint.html.j2`
- Modify: `website/src/endpoint.ts`

**Interfaces:**
- Consumes: `../data/b3it/<slug>/b3it.json` (Task 4 schema).
- Produces: a `#b3it-section` rendered before the LT charts when `b3it.json` is present.

- [ ] **Step 1: Add the container to the template**

In `endpoint.html.j2`, add `<section id="b3it-section"></section>` above the existing `#charts-container` (read the template first to place it correctly).

- [ ] **Step 2: Add the B3IT renderer in endpoint.ts**

Add an interface + fetch + render. The TV chart reuses `makeChangeShapes` for epoch/change markers (visual parity with the LT anomaly chart):

```typescript
interface B3ITData {
  status: string;
  retired_reason: string | null;
  n_bis: number;
  unstable: boolean;
  epochs: { start: string; end: string | null; end_reason: string | null; change_date: string | null }[];
  tv_series: { dates: string[]; values: number[] };
  changes: { date: string; kind: string }[];
}

async function renderB3IT(slug: string): Promise<void> {
  const section = document.getElementById("b3it-section");
  if (!section) return;
  let data: B3ITData;
  try {
    const res = await fetch(`../data/b3it/${slug}/b3it.json`);
    if (!res.ok) return;
    data = await res.json();
  } catch {
    return;
  }

  const statusLabel = data.status === "monitoring" ? "monitoring"
    : `retired (${data.retired_reason ?? "?"})`;
  const badges = [
    `<span class="badge">B3IT: ${statusLabel}</span>`,
    `<span class="badge">${data.n_bis} border inputs</span>`,
    data.unstable ? `<span class="badge warn">⚠ unstable</span>` : "",
  ].join(" ");

  const header = document.createElement("div");
  header.className = "b3it-header";
  header.innerHTML = `<h2>Border-input drift</h2>${badges}`;
  section.innerHTML = "";
  section.appendChild(header);

  if (data.tv_series.values.length === 0) {
    const note = document.createElement("div");
    note.className = "no-data";
    note.textContent = "No TV data for the current epoch.";
    section.appendChild(note);
    return;
  }

  const dates = data.tv_series.dates.map((d) => new Date(d));
  const changeDts = data.changes.map((c) => new Date(c.date));
  const plotDiv = document.createElement("div");
  plotDiv.className = "chart";
  section.appendChild(plotDiv);
  Plotly.newPlot(
    plotDiv,
    [{
      x: dates, y: data.tv_series.values, type: "scatter", mode: "lines",
      name: "Mean TV vs reference", line: { width: 1.5, color: "#8250df" },
      hovertemplate: "%{x}<br>TV: %{y:.4f}<extra></extra>",
    }],
    {
      title: { text: "Border-input TV distance over time", font: { color: "#1f2328", size: 14 } },
      xaxis: { title: { text: "Date" }, gridcolor: "#d0d7de" },
      yaxis: { title: { text: "Mean TV distance" }, gridcolor: "#d0d7de", rangemode: "tozero" },
      paper_bgcolor: "#f6f8fa", plot_bgcolor: "#ffffff", font: { color: "#1f2328" },
      height: 360, margin: { t: 40, r: 20, b: 50, l: 60 },
      shapes: makeChangeShapes(changeDts),
    },
    { responsive: true, displayModeBar: false }
  );
}
```

Call `renderB3IT(manifest.slug)` from `init()` (alongside `renderCharts(manifest)`). Add minimal `.badge`/`.badge.warn`/`.b3it-header` styles to `website/style.css` consistent with existing classes.

- [ ] **Step 3: Build and verify**

Run: `make build`
Expected: build succeeds; open a known-monitoring endpoint page (or any page — B3IT-less pages simply omit the section). Currently all B3IT states are migrated→retired with empty references, so the section shows status + "No TV data for the current epoch." — that is the expected empty state.

- [ ] **Step 4: Commit**

```bash
git add website/src/endpoint.ts website/templates/endpoint.html.j2 website/style.css
git commit --no-verify -m "feat: render B3IT TV-over-time section on endpoint pages"
```

---

## PR 3 — Unified index + change feed

### File structure

- Create `src/trackllm_website/generate_site/changes.py`
- Modify `src/trackllm_website/generate_site/render.py` (emit `changes.json`, build unified index rows)
- Modify `website/templates/index.html.j2` (changes feed + unified table)
- Create `tests/test_generate_site_changes.py`
- Modify `tests/test_generate_site_render.py`

### Task 6: Merged change feed derivation

**Files:**
- Create: `src/trackllm_website/generate_site/changes.py`
- Test: `tests/test_generate_site_changes.py`

**Interfaces:**
- Consumes: `EndpointInfo` (lt), `B3ITView` (b3it), raw `lt/lt_changes.json` (dict slug→list of `{endpoint, index, date, sigma, first_detected}`).
- Produces:
  - `@dataclass ChangeEvent(date: str, slug: str, model: str, provider: str, method: str, magnitude: float | None)`
  - `merge_changes(lt_changes: dict, lt_by_slug: dict[str, EndpointInfo], b3it_views: dict[str, B3ITView]) -> list[ChangeEvent]` — LT events (method `"LT"`, `magnitude = sigma`); B3IT events from epochs with `end_reason == "change_detected"` and a `change_date` (method `"B3IT"`, `magnitude=None`). Sorted by `date` descending. Model/provider resolved from `lt_by_slug` (LT) / the view (B3IT); for an LT slug absent from `lt_by_slug`, fall back to the slug as model.
  - `to_json(events: list[ChangeEvent]) -> list[dict]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate_site_changes.py
from trackllm_website.generate_site.changes import merge_changes


def test_merge_sorts_newest_first_across_methods():
    lt_changes = {"s1": [{"endpoint": "s1", "index": 5, "date": "2026-03-01T00:00:00Z",
                          "sigma": 12.0, "first_detected": "2026-06-01T00:00:00Z"}]}
    # minimal stand-ins; real test passes EndpointInfo / B3ITView instances
    class V:
        model = "m/b"; provider = "q"
        epochs = [{"start": "2026-01-01T00:00:00Z", "end": "2026-05-01T00:00:00Z",
                   "end_reason": "change_detected", "change_date": "2026-04-15T00:00:00Z"}]
    class LT:
        model = "m/a"; provider = "p"
    events = merge_changes(lt_changes, {"s1": LT()}, {"s2": V()})
    assert [e.method for e in events] == ["B3IT", "LT"]  # 2026-04-15 > 2026-03-01
    assert events[0].slug == "s2"
    assert events[1].magnitude == 12.0


def test_empty_inputs_yield_empty_feed():
    assert merge_changes({}, {}, {}) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_generate_site_changes.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement changes.py**

```python
from dataclasses import asdict, dataclass


@dataclass
class ChangeEvent:
    date: str
    slug: str
    model: str
    provider: str
    method: str
    magnitude: float | None


def merge_changes(lt_changes, lt_by_slug, b3it_views) -> list[ChangeEvent]:
    events: list[ChangeEvent] = []
    for slug, evs in lt_changes.items():
        ep = lt_by_slug.get(slug)
        model = ep.model if ep else slug
        provider = ep.provider if ep else ""
        for ev in evs:
            events.append(ChangeEvent(ev["date"], slug, model, provider, "LT", ev.get("sigma")))
    for slug, view in b3it_views.items():
        for epoch in view.epochs:
            if epoch.get("end_reason") == "change_detected" and epoch.get("change_date"):
                events.append(ChangeEvent(epoch["change_date"], slug, view.model,
                                          view.provider, "B3IT", None))
    events.sort(key=lambda e: e.date, reverse=True)
    return events


def to_json(events) -> list[dict]:
    return [asdict(e) for e in events]
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_generate_site_changes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uv run prek run --all-files || true
git add src/trackllm_website/generate_site/changes.py tests/test_generate_site_changes.py
git commit --no-verify -m "feat: merge LT and B3IT change events into one feed"
```

### Task 7: Emit changes.json + build unified index rows

**Files:**
- Modify: `src/trackllm_website/generate_site/render.py`
- Test: `tests/test_generate_site_render.py` (extend)

**Interfaces:**
- Consumes: `merge_changes`, `changes.to_json` (Task 6), `discover_b3it_views` (Task 3), `lt/lt_changes.json`.
- Produces:
  - `@dataclass IndexRow(slug, model, provider, lt_status: str | None, b3it_status: str | None, b3it_reason: str | None, recent_change: bool)` in `render.py`.
  - `build_index_rows(lt_endpoints, b3it_views, recent_slugs: set[str]) -> list[IndexRow]` — union by slug, alphabetical by `model.lower()`. `lt_status`: `"monitoring"` if `is_active` else `"retired"`, `None` if no LT. `b3it_status`: `"monitoring"`/`"retired"`/`None`.
  - In `render_site`: write `website_dir/"data"/"changes.json"`; compute `recent_slugs` = slugs with a change in the last `RECENT_CHANGE_DAYS = 14` days; render index with `rows` + the recent `changes` list.

- [ ] **Step 1: Write the failing test**

```python
def test_render_emits_changes_and_unified_index(tmp_path):
    _scaffold(tmp_path)
    import json
    (tmp_path / "data" / "lt" / "lt_changes.json").write_text(json.dumps(
        {"m2fa23p": [{"endpoint": "m2fa23p", "index": 3, "date": "2026-06-20T00:00:00Z",
                      "sigma": 9.0, "first_detected": "2026-06-21T00:00:00Z"}]}))
    from trackllm_website.generate_site.render import render_site
    render_site(tmp_path)
    assert (tmp_path / "data" / "changes.json").exists()
    index = (tmp_path / "index.html").read_text()
    assert "m/a" in index            # endpoint row
    assert "2026-06-20" in index     # change feed entry
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_generate_site_render.py::test_render_emits_changes_and_unified_index -v`
Expected: FAIL

- [ ] **Step 3: Wire changes + index rows into render_site**

Read `lt/lt_changes.json` (default `{}` if absent), call `merge_changes`, write `changes.json`. Compute `recent_slugs`. Build `IndexRow`s via `build_index_rows`. Render `index.html.j2` with `rows=...`, `changes=<recent slice>`, `body_class="index"`. (Template updated in Task 8.) Drop the old `active_endpoints`/`inactive_endpoints` context.

- [ ] **Step 4: Update index.html.j2 minimally to consume `rows` + `changes` (full styling in Task 8)** — enough for the assertions to pass.

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_generate_site_render.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
uv run prek run --all-files || true
git add src/trackllm_website/generate_site/render.py website/templates/index.html.j2 tests/test_generate_site_render.py
git commit --no-verify -m "feat: emit changes.json and build unified index rows"
```

### Task 8: Index template — feed + unified table + recent badge

**Files:**
- Modify: `website/templates/index.html.j2`
- Modify: `website/style.css`

**Interfaces:**
- Consumes: `rows: list[IndexRow]`, `changes: list[dict]` (recent slice).

- [ ] **Step 1: Render the changes feed + unified table**

Replace the body with: an h1/subtitle; a "Recent changes" section listing `changes` (date, `<a href="endpoints/{{c.slug}}.html">{{c.model}}</a>`, method tag, magnitude `{{ '%.0fσ'|format(c.magnitude) if c.magnitude else '' }}`) with an empty-state line when none; then a unified table — columns Model | Provider | LT | B3IT — one row per `IndexRow`. Show `—` for a `None` status; for B3IT retired show `retired ({{row.b3it_reason}})`. Add `class="recent"` to rows whose `recent_change` is true.

```html
{% extends "base.html.j2" %}
{% block title %}TrackLLM{% endblock %}
{% block content %}
        <h1>TrackLLM</h1>
        <p class="subtitle">Tracking LLM API drift via logprobs (LT) and border inputs (B3IT)</p>

        <section class="changes-section">
          <h2>Recent changes</h2>
          {% if changes %}
          <table><tbody>
          {% for c in changes %}
            <tr><td class="date">{{ c.date[:10] }}</td>
                <td><a href="endpoints/{{ c.slug }}.html">{{ c.model }}</a></td>
                <td>{{ c.provider }}</td>
                <td><span class="method {{ c.method|lower }}">{{ c.method }}</span></td>
                <td>{% if c.magnitude %}{{ '%.0f'|format(c.magnitude) }}σ{% endif %}</td></tr>
          {% endfor %}
          </tbody></table>
          {% else %}<p class="no-data">No changes detected yet.</p>{% endif %}
        </section>

        <section class="endpoints-section">
          <h2>Endpoints <span class="count">({{ rows | length }})</span></h2>
          <table>
            <thead><tr><th>Model</th><th>Provider</th><th>LT</th><th>B3IT</th></tr></thead>
            <tbody>
          {% for row in rows %}
            <tr class="{{ 'recent' if row.recent_change else '' }}">
              <td><a href="endpoints/{{ row.slug }}.html">{{ row.model }}</a></td>
              <td>{{ row.provider }}</td>
              <td>{{ row.lt_status or '—' }}</td>
              <td>{% if row.b3it_status == 'retired' %}retired ({{ row.b3it_reason }}){% else %}{{ row.b3it_status or '—' }}{% endif %}</td>
            </tr>
          {% endfor %}
            </tbody>
          </table>
        </section>
{% endblock %}
```

- [ ] **Step 2: Add styles** for `.changes-section`, `.method.lt`/`.method.b3it`, `tr.recent` to `style.css`, consistent with existing table styles.

- [ ] **Step 3: Build and verify**

Run: `make build`
Expected: index shows the unified table; the LT change events present in `website/data/lt/lt_changes.json` appear in the feed (B3IT feed empty until live change events exist).

- [ ] **Step 4: Commit**

```bash
git add website/templates/index.html.j2 website/style.css
git commit --no-verify -m "feat: unified index with change feed and per-method status"
```

---

## PR 4 — Spend dashboard

### File structure

- Create `src/trackllm_website/generate_site/spend.py`
- Create `website/templates/spend.html.j2`
- Create `website/src/spend.ts`
- Modify `src/trackllm_website/generate_site/render.py` (emit `spend.json`, render `spend.html`, index strip)
- Modify `website/templates/index.html.j2` (spend strip)
- Modify `website/package.json` / build config so `spend.ts` is bundled
- Create `tests/test_generate_site_spend.py`

### Task 9: Spend aggregation

**Files:**
- Create: `src/trackllm_website/generate_site/spend.py`
- Test: `tests/test_generate_site_spend.py`

**Interfaces:**
- Consumes: ledger files `website/data/spend/<slug>/<YYYY-MM>.jsonl` (`{timestamp, kind, cost, n_queries, n_errors}`).
- Produces:
  - `GROUPS: dict[str, str]` = `{"onboard": "onboarding", "recheck": "onboarding", "reinit": "onboarding", "monitor": "monitoring", "lt": "lt", "vetting": "vetting"}`
  - `group_for_kind(kind: str) -> str` (unknown → `"other"`)
  - `aggregate_spend(spend_dir: Path, today: str) -> dict` returning:
    ```
    {"cumulative": {group: $}, "last_30d": {group: $},
     "daily": [{"date": "YYYY-MM-DD", "groups": {group: $}}],   # ascending
     "by_endpoint": [{"slug": s, "groups": {group: $}, "total": $}]}  # desc by total
    ```
    `today` is the reference `YYYY-MM-DD` for the 30-day window (passed in — no `datetime.now()` in the pure aggregator, for testability).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate_site_spend.py
from trackllm_website.generate_site.spend import aggregate_spend, group_for_kind


def test_group_for_kind():
    assert group_for_kind("reinit") == "onboarding"
    assert group_for_kind("monitor") == "monitoring"
    assert group_for_kind("lt") == "lt"
    assert group_for_kind("zzz") == "other"


def _line(ts, kind, cost):
    import json
    return json.dumps({"timestamp": ts, "kind": kind, "cost": cost, "n_queries": 1, "n_errors": 0})


def test_aggregate(tmp_path):
    d = tmp_path / "s1"; d.mkdir(parents=True)
    (d / "2026-06.jsonl").write_text(
        _line("2026-06-24T00:00:00Z", "onboard", 0.10) + "\n" +
        _line("2026-06-24T00:00:00Z", "monitor", 0.02) + "\n" +
        _line("2026-05-01T00:00:00Z", "vetting", 0.01) + "\n")
    out = aggregate_spend(tmp_path, "2026-06-24")
    assert round(out["cumulative"]["onboarding"], 2) == 0.10
    assert round(out["cumulative"]["monitoring"], 2) == 0.02
    assert round(out["last_30d"].get("vetting", 0), 2) == 0.0  # May 1 is >30d before Jun 24
    assert out["by_endpoint"][0]["slug"] == "s1"
    assert any(day["date"] == "2026-06-24" for day in out["daily"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_generate_site_spend.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement spend.py**

```python
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

GROUPS = {"onboard": "onboarding", "recheck": "onboarding", "reinit": "onboarding",
          "monitor": "monitoring", "lt": "lt", "vetting": "vetting"}


def group_for_kind(kind: str) -> str:
    return GROUPS.get(kind, "other")


def aggregate_spend(spend_dir: Path, today: str) -> dict:
    cumulative: dict[str, float] = defaultdict(float)
    last_30d: dict[str, float] = defaultdict(float)
    daily: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_endpoint: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    cutoff = date.fromisoformat(today) - timedelta(days=30)

    if spend_dir.exists():
        for f in sorted(spend_dir.glob("*/*.jsonl")):
            slug = f.parent.name
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                g = group_for_kind(rec["kind"])
                cost = rec["cost"]
                day = str(rec["timestamp"])[:10]
                cumulative[g] += cost
                by_endpoint[slug][g] += cost
                daily[day][g] += cost
                if date.fromisoformat(day) > cutoff:
                    last_30d[g] += cost

    by_ep = [{"slug": s, "groups": dict(g), "total": sum(g.values())}
             for s, g in by_endpoint.items()]
    by_ep.sort(key=lambda r: r["total"], reverse=True)
    return {
        "cumulative": dict(cumulative),
        "last_30d": dict(last_30d),
        "daily": [{"date": d, "groups": dict(g)} for d, g in sorted(daily.items())],
        "by_endpoint": by_ep,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_generate_site_spend.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uv run prek run --all-files || true
git add src/trackllm_website/generate_site/spend.py tests/test_generate_site_spend.py
git commit --no-verify -m "feat: build-time spend aggregation for the site"
```

### Task 10: Emit spend.json + render /spend page + index strip

**Files:**
- Modify: `src/trackllm_website/generate_site/render.py`
- Create: `website/templates/spend.html.j2`
- Modify: `website/templates/index.html.j2`
- Test: `tests/test_generate_site_render.py` (extend)

**Interfaces:**
- Consumes: `aggregate_spend` (Task 9).
- Produces: `render_site` writes `website_dir/"data"/"spend.json"` = `aggregate_spend(spend_dir, today)` (today via `datetime.now(timezone.utc).strftime("%Y-%m-%d")` in `render_site`); renders `spend.html` from `spend.html.j2`; passes `spend=<aggregate>` to the index for the strip.

- [ ] **Step 1: Write the failing test**

```python
def test_render_emits_spend(tmp_path):
    _scaffold(tmp_path)
    import json
    sp = tmp_path / "data" / "spend" / "m2fa23p"; sp.mkdir(parents=True)
    (sp / "2026-06.jsonl").write_text(json.dumps(
        {"timestamp": "2026-06-24T00:00:00Z", "kind": "lt", "cost": 0.05,
         "n_queries": 1, "n_errors": 0}) + "\n")
    from trackllm_website.generate_site.render import render_site
    render_site(tmp_path)
    assert (tmp_path / "data" / "spend.json").exists()
    assert (tmp_path / "spend.html").exists()
    assert "spend" in (tmp_path / "index.html").read_text().lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_generate_site_render.py::test_render_emits_spend -v`
Expected: FAIL

- [ ] **Step 3: Implement**

In `render_site`: compute `today`, `spend = aggregate_spend(website_dir/"data"/"spend", today)`, write `spend.json`. Create `spend.html.j2` (extends `base.html.j2`, `css_path="style.css"`): cumulative-by-group table, a `<div id="spend-chart"></div>` placeholder (filled by `spend.ts`), and a per-endpoint table from `spend.by_endpoint`. Add a spend strip to `index.html.j2` (cumulative + last-30d totals, link to `spend.html`). Render `spend.html` in `render_site`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_generate_site_render.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uv run prek run --all-files || true
git add src/trackllm_website/generate_site/render.py website/templates/spend.html.j2 website/templates/index.html.j2 tests/test_generate_site_render.py
git commit --no-verify -m "feat: emit spend.json, render /spend page and index spend strip"
```

### Task 11: Spend page chart (TypeScript + bundling)

**Files:**
- Create: `website/src/spend.ts`
- Modify: `website/package.json` (and `tsconfig.json` if entry points are listed)

**Interfaces:**
- Consumes: `data/spend.json` (`{cumulative, last_30d, daily, by_endpoint}` from Task 9).

- [ ] **Step 1: Inspect the build**

Read `website/package.json` build/watch scripts and how `endpoint.ts`/`index` entry points are declared (esbuild/bun bundle invocation). Add `spend.ts` as an entry point the same way `endpoint.ts` is built into `website/js/`.

- [ ] **Step 2: Implement spend.ts**

```typescript
import Plotly from "plotly.js-dist-min";

interface SpendData {
  cumulative: Record<string, number>;
  last_30d: Record<string, number>;
  daily: { date: string; groups: Record<string, number> }[];
  by_endpoint: { slug: string; groups: Record<string, number>; total: number }[];
}

const GROUP_ORDER = ["onboarding", "monitoring", "lt", "vetting", "other"];
const GROUP_COLOR: Record<string, string> = {
  onboarding: "#8250df", monitoring: "#0969da", lt: "#1a7f37",
  vetting: "#9a6700", other: "#57606a",
};

async function init(): Promise<void> {
  const el = document.getElementById("spend-chart");
  if (!el) return;
  const res = await fetch("data/spend.json");
  if (!res.ok) return;
  const data: SpendData = await res.json();
  const dates = data.daily.map((d) => new Date(d.date));
  const traces = GROUP_ORDER.filter((g) => data.daily.some((d) => d.groups[g]))
    .map((g) => ({
      x: dates, y: data.daily.map((d) => d.groups[g] ?? 0),
      type: "bar" as const, name: g, marker: { color: GROUP_COLOR[g] },
    }));
  Plotly.newPlot(el, traces, {
    barmode: "stack",
    title: { text: "Daily spend by category", font: { color: "#1f2328", size: 14 } },
    xaxis: { title: { text: "Date" }, gridcolor: "#d0d7de" },
    yaxis: { title: { text: "USD" }, gridcolor: "#d0d7de", rangemode: "tozero" },
    paper_bgcolor: "#f6f8fa", plot_bgcolor: "#ffffff", font: { color: "#1f2328" },
    height: 400, margin: { t: 40, r: 20, b: 50, l: 60 },
  }, { responsive: true, displayModeBar: false });
}

init();
```

Add the `<script>` tag for the built `spend.js` to `spend.html.j2` (mirror how `endpoint.html.j2` references its built JS).

- [ ] **Step 3: Build and verify**

Run: `make build`
Expected: `website/js/spend.js` produced; `spend.html` renders the stacked daily-spend chart, cumulative table, and per-endpoint table. With only `vetting` rows present today, the chart shows vetting spend — expected.

- [ ] **Step 4: Commit**

```bash
git add website/src/spend.ts website/package.json website/tsconfig.json website/templates/spend.html.j2
git commit --no-verify -m "feat: daily spend chart on the /spend page"
```

---

## Final verification

- [ ] Run the full suite: `uv run pytest -q` — all green.
- [ ] `make build` — completes; `index.html`, `endpoints/*.html`, `spend.html`, and `data/{changes.json,spend.json,b3it/*/b3it.json}` present.
- [ ] `uv run prek run --all-files` — clean.
- [ ] Spot-check in a browser (`make serve`): index unified table + change feed + spend strip; an endpoint page B3IT section (empty-state today); `/spend` chart and tables.

## Self-review notes (spec coverage)

- Build-time derivation, package split → PR 1, Tasks 3/6/9.
- `b3it.json` / `changes.json` / `spend.json` artifacts → Tasks 4, 7, 10.
- Unified index + normalized status + recent-change badge → Tasks 7, 8.
- Change feed on homepage → Task 8.
- Spend page **and** index strip → Tasks 10, 11.
- Endpoint B3IT section (TV chart, BI count, status, instability badge) → Task 5.
- Empty/partial-data handling → Tasks 3 (empty `tv_series`), 5 (no-data note), 8 (empty feed), 9 (zeroed aggregate).
- TDD with real-data fixtures → every task is test-first.
