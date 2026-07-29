# Aggregate Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the deferred aggregate pages — a per-provider page and a global Changes page — replace the Overview's per-variant "Provider drift rate" panel with a base-provider ranking that carries uncertainty, and cross-link every page to its model, provider and endpoint.

**Architecture:** Everything is derived at build time from already-generated outputs (`data/lt/<slug>/lt_scores.json`, B3IT views, `data/changes.json`) — no raw logprobs, no detection changes. Three new pure-ish Python modules (`rates.py`, `feed.py`, `provider.py`, `changes_page.py`) feed new JSON blobs that static Jinja shells hydrate with TypeScript, exactly like the existing Overview/Model pages.

**Tech Stack:** Python 3 + `uv`, Jinja2 templates, TypeScript bundled by Bun, pytest, `prek` for format/lint.

**Spec:** `docs/superpowers/specs/2026-07-28-aggregate-pages-design.md`

## Global Constraints

- All commands run through `uv`. Use `uv run --frozen pytest tests -q` for the suite: `--frozen` stops `uv sync` from churning `uv.lock`. (The `tests` scope used to also dodge a collection error under a local `reference/` checkout; that directory was deleted on 2026-07-28, so a bare `uv run --frozen pytest` now collects cleanly too.)
- **beartype is active package-wide** (`beartype_this_package()` in `src/trackllm_website/__init__.py`). Type annotations are enforced at runtime and the implicit numeric tower is off: passing an `int` where a parameter is annotated `float` raises. Annotate honestly and pass `0.0`, not `0`.
- Tests alone do not prove the build works. Before committing any task that touches the generator, run it against the real repo data: `uv run --frozen python -m trackllm_website.generate_site` must exit 0.
- Filenames for generated output always go through `slugify` from `src/trackllm_website/util.py`.
- Plots use plotly; the sparklines/strips here are hand-rolled inline SVG, matching the existing `overview.ts` / `model.ts` — do not introduce a chart library for them.
- Comment sparingly (~10% of what feels natural); never delete an existing comment.
- No default argument values in new functions — parameters come from constants or the caller.
- Never silence an error to get a green test. If a fixture doesn't produce the data a test needs, fix the fixture.
- After editing code, run `prek run --all-files`. `git commit` fails with "pre-commit not found" in this repo — run `prek` first, then commit with `--no-verify`.
- TypeScript must typecheck clean: `cd website && bunx tsc --noEmit`.
- Terminology, verbatim: "LT" = logprob tracking, "B3IT" = black-box border input tracking, "border inputs", "drift from baseline". Never claim a *cause* for a change — we detect that outputs moved, not why.
- Every rate on every surface is `changes / endpoint-year`; below `MIN_ENDPOINT_YEARS = 0.5` no rate is published anywhere.

---

## File Structure

**Create:**
- `src/trackllm_website/generate_site/rates.py` — normalised rates + Poisson intervals (pure, no I/O).
- `src/trackllm_website/generate_site/feed.py` — change-feed enrichment (magnitude, sparkline window, severity, link slugs), extracted from `overview.py`.
- `src/trackllm_website/generate_site/provider.py` — base-provider aggregation → `data/providers/<slug>.json` + the Overview's provider rows.
- `src/trackllm_website/generate_site/changes_page.py` → `data/changes_page.json`.
- `website/templates/provider.html.j2`, `website/templates/changes.html.j2`.
- `website/src/components.ts` — shared render helpers (sparkline, rate bar, volume grid, badges, pills).
- `website/src/provider.ts`, `website/src/changes.ts`.
- `tests/conftest.py` — the fixture writers currently duplicated across test files.
- `tests/test_generate_site_rates.py`, `tests/test_generate_site_feed.py`, `tests/test_generate_site_provider.py`, `tests/test_generate_site_changes_page.py`.

**Test layout note:** `tests/` has no `__init__.py`, so pytest puts that directory
on `sys.path` and helpers are imported as `from conftest import ...` — never
`from tests.test_generate_site_overview import ...`, which does not resolve here.

**Modify:**
- `src/trackllm_website/generate_site/overview.py` — feed logic moves out; provider-stats block deleted; `providers` key removed from its return.
- `src/trackllm_website/generate_site/render.py` — orchestrate provider + changes-page builds, render the new pages, pass provider slug to endpoint pages.
- `src/trackllm_website/generate_site/model.py` — emit `base` per endpoint and a flattened model-wide change list.
- `website/src/overview.ts` — consume the new provider rows, link model/provider names.
- `website/src/model.ts` — five model-page edits.
- `website/templates/base.html.j2` — nav gains Changes.
- `website/templates/endpoint.html.j2` — provider link.
- `website/package.json` — new bundler entrypoints.
- `tests/test_generate_site_overview.py`, `tests/test_generate_site_model.py`, `tests/test_generate_site_render.py`.

---

## Task 1: `rates.py` — normalised rate + Poisson interval

**Files:**
- Create: `src/trackllm_website/generate_site/rates.py`
- Test: `tests/test_generate_site_rates.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `MIN_ENDPOINT_YEARS: float = 0.5`, `poisson_interval(k: int, exposure: float) -> tuple[float, float] | None`, `drift_rate(k: int, exposure: float) -> float | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate_site_rates.py`:

```python
import pytest

from trackllm_website.generate_site.rates import (
    MIN_ENDPOINT_YEARS,
    drift_rate,
    poisson_interval,
)


def test_zero_changes_uses_rule_of_three():
    # no events over T endpoint-years bounds the rate at 3/T, it does not prove 0
    assert poisson_interval(0, 3.0) == (0.0, 1.0)


def test_interval_is_symmetric_around_the_point_estimate():
    lo, hi = poisson_interval(4, 2.0)
    assert lo == pytest.approx((4 - 1.96 * 2) / 2.0)
    assert hi == pytest.approx((4 + 1.96 * 2) / 2.0)


def test_lower_bound_clamped_at_zero():
    lo, hi = poisson_interval(1, 1.0)
    assert lo == 0.0
    assert hi == pytest.approx(1 + 1.96)


def test_no_interval_without_exposure():
    assert poisson_interval(0, 0.0) is None
    assert poisson_interval(3, 0.0) is None


def test_rate_withheld_below_threshold():
    assert drift_rate(1, MIN_ENDPOINT_YEARS - 0.01) is None
    assert drift_rate(0, 0.0) is None


def test_rate_published_at_exactly_the_threshold():
    assert drift_rate(1, MIN_ENDPOINT_YEARS) == pytest.approx(2.0)


def test_rate_is_changes_per_endpoint_year():
    assert drift_rate(49, 15.04) == pytest.approx(49 / 15.04)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_generate_site_rates.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'trackllm_website.generate_site.rates'`

- [ ] **Step 3: Write the implementation**

Create `src/trackllm_website/generate_site/rates.py`:

```python
"""Normalised drift rates and their uncertainty. Pure: no I/O, no build state."""

import math

# Below this much monitoring a rate is not a measurement: one change in three
# weeks computes to ~17/year. Every surface withholds the rate instead.
MIN_ENDPOINT_YEARS = 0.5

_Z = 1.96


def poisson_interval(k: int, exposure: float) -> tuple[float, float] | None:
    """95% interval for k changes observed over `exposure` endpoint-years.

    k == 0 uses the rule of three: zero events is evidence of a rate below
    3/exposure, not evidence of a rate of zero.
    """
    if exposure <= 0:
        return None
    if k == 0:
        return (0.0, 3.0 / exposure)
    half = _Z * math.sqrt(k)
    return (max(0.0, (k - half) / exposure), (k + half) / exposure)


def drift_rate(k: int, exposure: float) -> float | None:
    """Detected changes per endpoint-year, or None when exposure is too thin."""
    if exposure < MIN_ENDPOINT_YEARS:
        return None
    return k / exposure
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_generate_site_rates.py -q`
Expected: 7 passed

- [ ] **Step 5: Lint and commit**

```bash
prek run --all-files
git add src/trackllm_website/generate_site/rates.py tests/test_generate_site_rates.py
git commit --no-verify -m "feat(site): normalised drift rate with Poisson interval"
```

---

## Task 2: `tests/conftest.py` — one copy of the fixture writers

`_write_lt_endpoint` is byte-for-byte duplicated in
`tests/test_generate_site_overview.py` and `tests/test_generate_site_model.py`;
three more test files are about to need it. Hoist it (and `_write_b3it_state`)
into `conftest.py` before writing them. The `_write_b3it_with_transition`
variants genuinely differ between the two files — leave those alone.

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_generate_site_overview.py`, `tests/test_generate_site_model.py`

**Interfaces:**
- Produces: `write_lt_endpoint(root: Path, slug: str, model: str, provider: str, *, dates: list[str], changes: list[dict], drift: list[float]) -> None` and `write_b3it_state(root: Path, slug: str, model: str, provider: str, *, status: str) -> None`, imported by tests as `from conftest import write_lt_endpoint`.

- [ ] **Step 1: Create `tests/conftest.py`**

Move the bodies verbatim from `tests/test_generate_site_overview.py` (its
`_write_lt_endpoint`, lines 33–65, and `_write_b3it_state`, lines 68–91),
dropping the leading underscore and making `status` an explicit keyword
parameter with no default (per the no-default-arguments rule — every caller
passes it):

```python
import json
from pathlib import Path


def write_lt_endpoint(
    root: Path, slug: str, model: str, provider: str, *, dates, changes, drift
):
    d = root / "data" / "lt" / slug
    prompt_dir = d / "default"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "info.json").write_text(
        json.dumps({"prompt": "hi", "endpoint": {"model": model, "provider": provider}})
    )
    month = dates[-1][:7]
    day = dates[-1][8:10]
    month_dir = prompt_dir / month
    month_dir.mkdir()
    (month_dir / "queries.json").write_text(json.dumps([[f"{day} 00:00:00", 0]]))
    (d / "lt_scores.json").write_text(
        json.dumps(
            {
                "n_per_test": 24,
                "dates": dates,
                "scores": [0.5] * len(dates),
                "sigmas": [None] * len(dates),
                "changes": changes,
                "drift_dates": dates,
                "drift": drift,
            }
        )
    )


def write_b3it_state(root: Path, slug: str, model: str, provider: str, *, status):
    state = {
        "endpoint": {
            "api": "openrouter",
            "model": model,
            "provider": provider,
            "cost": [0.1, 0.2],
            "max_logprobs": None,
        },
        "status": status,
        "retired": None,
        "epochs": [
            {
                "start": "2026-01-01T00:00:00Z",
                "border_inputs": [],
                "reference": {},
                "end": None,
            }
        ],
    }
    sd = root / "data" / "b3it" / "state"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / f"{slug}.json").write_text(json.dumps(state))
```

- [ ] **Step 2: Point the two existing test files at it**

In `tests/test_generate_site_overview.py`: delete the local `_write_lt_endpoint`
and `_write_b3it_state`, add `from conftest import write_b3it_state, write_lt_endpoint`,
and rename every call site (the `_write_b3it_state(...)` calls must now pass
`status="monitoring"` explicitly).

In `tests/test_generate_site_model.py`: delete the local `_write_lt_endpoint`,
add `from conftest import write_lt_endpoint`, rename every call site.

- [ ] **Step 3: Run the suite to prove the move changed nothing**

Run: `uv run pytest tests/test_generate_site_overview.py tests/test_generate_site_model.py -q`
Expected: same pass count as before the move (12 + the model file's tests), zero failures.

- [ ] **Step 4: Lint and commit**

```bash
prek run --all-files
git add tests
git commit --no-verify -m "test: hoist duplicated site fixture writers into conftest"
```

---

## Task 3: `feed.py` — extract change-feed enrichment

The Changes page needs the same magnitude + sparkline enrichment the Overview
applies to its latest 10 changes. Move it out of `overview.py` so both use one
implementation, and add the link slugs every consumer needs.

**One deliberate behaviour change:** the B3IT feed items are now built from
`data/changes.json` (the canonical merged list, which includes live epoch
closures) instead of from `B3ITView.changes` (TV onsets only). This makes the
Overview feed agree with the Changes log. Assert it with a test rather than
hiding it.

**Files:**
- Create: `src/trackllm_website/generate_site/feed.py`
- Modify: `src/trackllm_website/generate_site/overview.py` (delete lines 16–31 constants that move, 38–50 `downsample_trace`, 113–228 feed helpers; import from `feed` instead)
- Test: `tests/test_generate_site_feed.py`
- Modify: `tests/test_generate_site_overview.py` (import `downsample_trace` from `feed`)

**Interfaces:**
- Consumes: `trackllm_website.generate_site.b3it.B3ITView`.
- Produces:
  - `TRACE_LEN: int = 28`, `downsample_trace(vals: list[float | int], n: int) -> list[float]`
  - `build_feed_items(changes: list[dict], drift_by_slug: dict[str, list[tuple[datetime, float]]], b3it_by_slug: dict[str, B3ITView], now: datetime) -> list[dict]`
  - Each item: `{date, iso, daysAgo, slug, model, org, modelSlug, provider, providerSlug, method, desc, primary, secondary, sevKey, trace, changeFrac, magnitude}` where `method` is `"lt"`/`"b3it"`, `magnitude` is drift in nats (LT) or peak TV (B3IT) or `None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate_site_feed.py`:

```python
from datetime import datetime, timezone

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.feed import build_feed_items, downsample_trace

NOW = datetime(2026, 6, 30, tzinfo=timezone.utc)


def _drift(n: int, jump_at: int):
    return [
        (datetime(2026, 6, 1, tzinfo=timezone.utc).replace(day=1 + i), 0.1 if i < jump_at else 1.2)
        for i in range(n)
    ]


def _b3it_view(slug: str) -> B3ITView:
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 21)]
    values = [0.05] * 12 + [0.8] * 8
    return B3ITView(
        slug=slug,
        model="m/a",
        provider="p/fp8",
        status="monitoring",
        retired_reason=None,
        n_bis=3,
        unstable=False,
        epochs=[],
        tv_series={"dates": dates, "values": values},
        changes=[{"date": dates[12], "kind": "onset"}],
    )


def test_downsample_trace_caps_length():
    assert len(downsample_trace(list(range(200)), 28)) == 28


def test_downsample_trace_short_input_untouched():
    assert downsample_trace([1.0, 2.0], 28) == [1.0, 2.0]


def test_lt_item_carries_drift_magnitude_and_link_slugs():
    changes = [
        {
            "date": "2026-06-15T00:00:00Z",
            "slug": "m2fa23p",
            "model": "org/model-x",
            "provider": "chutes/fp8",
            "method": "LT",
            "magnitude": 40.0,
            "magnitude_display": "40σ",
        }
    ]
    items = build_feed_items(changes, {"m2fa23p": _drift(20, 14)}, {}, NOW)
    (item,) = items
    assert item["method"] == "lt"
    assert item["magnitude"] == 1.2
    assert item["model"] == "model-x"
    assert item["org"] == "org"
    assert item["providerSlug"] == "chutes"
    assert item["modelSlug"] == "org2fmodel-x"
    assert item["slug"] == "m2fa23p"
    assert item["secondary"] == "40σ conf"
    assert item["trace"]


def test_b3it_item_uses_peak_tv_from_the_view():
    changes = [
        {
            "date": "2026-06-13T00:00:00Z",
            "slug": "s1",
            "model": "org/model-x",
            "provider": "p/fp8",
            "method": "B3IT",
            "magnitude": None,
            "magnitude_display": "",
        }
    ]
    items = build_feed_items(changes, {}, {"s1": _b3it_view("s1")}, NOW)
    (item,) = items
    assert item["method"] == "b3it"
    assert item["magnitude"] == 0.8
    assert item["sevKey"] == "alert"


def test_items_sorted_newest_first():
    changes = [
        {
            "date": f"2026-06-{d:02d}T00:00:00Z",
            "slug": "m2fa23p",
            "model": "org/model-x",
            "provider": "p",
            "method": "LT",
            "magnitude": 10.0,
            "magnitude_display": "10σ",
        }
        for d in (3, 20, 11)
    ]
    items = build_feed_items(changes, {"m2fa23p": _drift(20, 14)}, {}, NOW)
    assert [i["date"] for i in items] == ["2026-06-20", "2026-06-11", "2026-06-03"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_generate_site_feed.py -q`
Expected: FAIL — no module named `feed`

- [ ] **Step 3: Create `feed.py` by moving the logic out of `overview.py`**

Create `src/trackllm_website/generate_site/feed.py` with the constants and
helpers currently in `overview.py` lines 16–31, 38–50 and 113–228, changed only
where noted:

```python
"""Change-feed enrichment: magnitude, sparkline window and link slugs per change.

Shared by the Overview (its latest slice) and the Changes page (all of them), so
the two never disagree about what a change looked like.
"""

from datetime import datetime

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.util import slugify

TRACE_LEN = 28

FEED_TRACE_LEN = 40
FEED_WINDOW_BEFORE = 60
FEED_WINDOW_AFTER = 20
FEED_MIN_WINDOW = 6
FEED_LT_PEAK_WINDOW = 20
FEED_B3IT_PEAK_WINDOW = 8
FEED_DEFAULT_CHANGE_FRAC = 0.5
LT_ALERT_THRESHOLD = 0.8
B3IT_ALERT_THRESHOLD = 0.6
CHANGED_THRESHOLD = 0.3


def downsample_trace(vals: list[float | int], n: int) -> list[float]:
    """Bucket-mean downsample to at most n points, rounded to 3 decimals."""
    if not vals:
        return []
    if len(vals) <= n:
        return [round(float(v), 3) for v in vals]
    out = []
    for b in range(n):
        lo = b * len(vals) // n
        hi = (b + 1) * len(vals) // n
        chunk = vals[lo:hi] or [vals[min(lo, len(vals) - 1)]]
        out.append(round(sum(chunk) / len(chunk), 3))
    return out


def _nearest_index(pairs: list[tuple[datetime, float]], target: datetime) -> int:
    return min(
        range(len(pairs)), key=lambda i: abs((pairs[i][0] - target).total_seconds())
    )


def _window(pairs: list[tuple[datetime, float]], k: int) -> tuple[list[float], float]:
    lo = max(0, k - FEED_WINDOW_BEFORE)
    hi = min(len(pairs), k + FEED_WINDOW_AFTER)
    window = [v for _, v in pairs[lo:hi]]
    if len(window) < FEED_MIN_WINDOW:
        return [], FEED_DEFAULT_CHANGE_FRAC
    return downsample_trace(window, FEED_TRACE_LEN), round((k - lo) / (hi - lo), 3)


def _severity(value: float, alert: float) -> str:
    if value >= alert:
        return "alert"
    return "changed" if value >= CHANGED_THRESHOLD else "stable"


def _links(change: dict) -> dict:
    model = change["model"]
    provider = change["provider"] or ""
    return {
        "slug": change["slug"],
        "model": model.split("/")[-1],
        "org": model.split("/")[0],
        "modelSlug": slugify(model),
        "provider": provider,
        "providerSlug": provider.split("/")[0],
    }


def _lt_item(
    change: dict, drift: list[tuple[datetime, float]], now: datetime
) -> dict:
    cd = datetime.fromisoformat(change["date"])
    magnitude = None
    trace: list[float] = []
    frac = FEED_DEFAULT_CHANGE_FRAC
    if drift:
        k = _nearest_index(drift, cd)
        peak_hi = min(len(drift), k + FEED_LT_PEAK_WINDOW)
        magnitude = round(max(v for _, v in drift[k:peak_hi]), 2)
        trace, frac = _window(drift, k)
    display = magnitude if magnitude is not None else "—"
    return {
        "date": change["date"][:10],
        "iso": change["date"],
        "daysAgo": (now - cd).days,
        "method": "lt",
        "magnitude": magnitude,
        "desc": f"Logprob averages moved {display} nats from the reference period.",
        "primary": f"drift {display}",
        "secondary": f"{change['magnitude_display']} conf",
        "sevKey": _severity(magnitude or 0.0, LT_ALERT_THRESHOLD),
        "trace": trace,
        "changeFrac": frac,
        **_links(change),
    }


def _b3it_item(change: dict, view: B3ITView | None, now: datetime) -> dict:
    cd = datetime.fromisoformat(change["date"])
    peak = 0.0
    trace: list[float] = []
    frac = FEED_DEFAULT_CHANGE_FRAC
    pairs = (
        list(
            zip(
                (datetime.fromisoformat(s) for s in view.tv_series["dates"]),
                view.tv_series["values"],
            )
        )
        if view
        else []
    )
    if pairs:
        k = _nearest_index(pairs, cd)
        peak_hi = min(len(pairs), k + FEED_B3IT_PEAK_WINDOW)
        peak = round(max(v for _, v in pairs[k:peak_hi]), 3)
        trace, frac = _window(pairs, k)
    return {
        "date": change["date"][:10],
        "iso": change["date"],
        "daysAgo": (now - cd).days,
        "method": "b3it",
        "magnitude": peak,
        "desc": f"Border-input output distribution moved (TV {peak:.2f}) from the reference.",
        "primary": f"TV {peak:.2f}",
        "secondary": "border-input shift",
        "sevKey": _severity(peak, B3IT_ALERT_THRESHOLD),
        "trace": trace,
        "changeFrac": frac,
        **_links(change),
    }


def build_feed_items(
    changes: list[dict],
    drift_by_slug: dict[str, list[tuple[datetime, float]]],
    b3it_by_slug: dict[str, B3ITView],
    now: datetime,
) -> list[dict]:
    """Enrich merged change events (changes.json shape) for display, newest first."""
    items = []
    for change in changes:
        if change["method"] == "LT":
            items.append(_lt_item(change, drift_by_slug.get(change["slug"], []), now))
        else:
            items.append(_b3it_item(change, b3it_by_slug.get(change["slug"]), now))
    items.sort(key=lambda i: i["iso"], reverse=True)
    return items
```

- [ ] **Step 4: Rewire `overview.py` onto `feed.py`**

In `src/trackllm_website/generate_site/overview.py`:

1. Delete `TRACE_LEN`, `FEED_*`, `LT_ALERT_THRESHOLD`, `B3IT_ALERT_THRESHOLD`, `CHANGED_THRESHOLD`, `downsample_trace`, `_nearest_index`, `_feed_window`, `_build_lt_feed_item`, `_build_b3it_feed_item`, `_build_feed`.
2. Add at the top: `from trackllm_website.generate_site.feed import TRACE_LEN, build_feed_items, downsample_trace`.
3. Keep `FEED_LT_SIZE = 6` and `FEED_B3IT_SIZE = 4` in `overview.py` — they are the Overview's slice policy, not feed logic.
4. Replace the `feed = _build_feed(...)` call inside `build_overview` with:

```python
    drift_by_slug = {slug: d.drift for slug, d in lt_data.items()}
    all_items = (
        build_feed_items(changes, drift_by_slug, b3it_views, now) if now else []
    )
    lt_items = [i for i in all_items if i["method"] == "lt"][:FEED_LT_SIZE]
    b3it_items = [i for i in all_items if i["method"] == "b3it"][:FEED_B3IT_SIZE]
    feed = sorted(lt_items + b3it_items, key=lambda i: i["iso"], reverse=True)
```

- [ ] **Step 5: Update the moved-symbol import in the existing overview test**

In `tests/test_generate_site_overview.py`, change

```python
from trackllm_website.generate_site.overview import build_overview, downsample_trace
```

to

```python
from trackllm_website.generate_site.feed import downsample_trace
from trackllm_website.generate_site.overview import build_overview
```

- [ ] **Step 6: Add the agreement test to `tests/test_generate_site_feed.py`**

```python
def test_overview_feed_entries_come_from_changes_json(fake_site_feed_agreement):
    # the Overview's slice must be a subset of the canonical merged change list
    ov, changes = fake_site_feed_agreement
    canonical = {(c["date"][:10], c["slug"], c["method"].lower()) for c in changes}
    for item in ov["feed"]:
        assert (item["date"], item["slug"], item["method"]) in canonical
```

with the fixture (reuse the shape from `tests/test_generate_site_overview.py`):

```python
import json

import pytest
from conftest import write_b3it_state, write_lt_endpoint

from trackllm_website.generate_site.b3it import discover_b3it_views
from trackllm_website.generate_site.lt import discover_lt_endpoints
from trackllm_website.generate_site.overview import build_overview


@pytest.fixture
def fake_site_feed_agreement(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(
        root,
        "m2fa23p",
        "m/a",
        "p",
        dates=dates,
        changes=[{"index": 24, "sigma": 40.0}],
        drift=[0.1] * 24 + [1.5] * 6,
    )
    write_b3it_state(root, "m2fa23p", "m/a", "p", status="monitoring")
    changes = [
        {
            "date": dates[24],
            "slug": "m2fa23p",
            "model": "m/a",
            "provider": "p",
            "method": "LT",
            "magnitude": 40.0,
            "magnitude_display": "40σ",
        }
    ]
    (root / "data" / "changes.json").write_text(json.dumps(changes))
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {"lt": 1.0}}))
    lt_endpoints = list(discover_lt_endpoints(root / "data" / "lt"))
    views = discover_b3it_views(
        root / "data" / "b3it" / "state", root / "data" / "b3it" / "phase_2"
    )
    return build_overview(root, lt_endpoints, views), changes
```

- [ ] **Step 7: Run the full generate_site suite**

Run: `uv run pytest tests/test_generate_site_feed.py tests/test_generate_site_overview.py tests/test_generate_site_render.py tests/test_generate_site_model.py -q`
Expected: all pass. If an overview feed assertion fails on B3IT ordering, that is the deliberate behaviour change above — update that assertion, do not restore the old code path.

- [ ] **Step 8: Lint and commit**

```bash
prek run --all-files
git add -A src/trackllm_website/generate_site tests
git commit --no-verify -m "refactor(site): extract change-feed enrichment into feed.py"
```

---

## Task 4: `provider.py` — base-provider aggregation

**Files:**
- Create: `src/trackllm_website/generate_site/provider.py`
- Test: `tests/test_generate_site_provider.py`

**Interfaces:**
- Consumes: `rates.MIN_ENDPOINT_YEARS`, `rates.drift_rate`, `rates.poisson_interval`, `feed.build_feed_items`, `lt.EndpointInfo`, `lt.load_lt_scores`, `b3it.B3ITView`, the Overview endpoint rows (`{slug, model, org, provider, methods, status, stableDays, nChanges, trace}`).
- Produces:
  - `base_provider(provider: str) -> str`, `variant_name(provider: str) -> str` (`""` for the default variant)
  - `endpoint_years(first: str, last: str) -> float`
  - `build_provider_views(website_dir: Path, lt_endpoints: list[EndpointInfo], b3it_views: dict[str, B3ITView], endpoint_rows: list[dict]) -> dict[str, dict]` keyed by provider slug
  - `overview_rows(views: dict[str, dict]) -> list[dict]`
  - View JSON: `{name, slug, n_endpoints, n_models, n_variants, first, last, months, lt, b3it, variants, changes, endpoints}` where `lt`/`b3it` are `{endpoints, years, changes, rate, ci}` and each variant is `{name, n_endpoints, lt, b3it, monitoring}`.
  - Overview row: `{name, slug, n_endpoints, n_models, n_variants, lt_years, lt_changes, lt_rate, lt_ci, b3it_endpoints, b3it_years, last_change}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate_site_provider.py`:

```python
import json

import pytest

from conftest import write_b3it_state, write_lt_endpoint
from trackllm_website.generate_site.b3it import discover_b3it_views
from trackllm_website.generate_site.lt import discover_lt_endpoints
from trackllm_website.generate_site.overview import build_overview
from trackllm_website.generate_site.provider import (
    base_provider,
    build_provider_views,
    overview_rows,
    variant_name,
)


def test_base_provider_and_variant_split():
    assert base_provider("chutes/fp8") == "chutes"
    assert variant_name("chutes/fp8") == "fp8"
    assert base_provider("chutes") == "chutes"
    assert variant_name("chutes") == ""


@pytest.fixture
def fake_site(tmp_path):
    """One provider `p` serving two models under two variants, one of which changed."""
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(
        root,
        "a23p",
        "org/a",
        "p",
        dates=dates,
        changes=[{"index": 24, "sigma": 40.0}],
        drift=[0.1] * 24 + [1.5] * 6,
    )
    write_lt_endpoint(
        root, "b23p2ffp8", "org/b", "p/fp8", dates=dates, changes=[], drift=[0.1] * 30
    )
    write_b3it_state(root, "a23p", "org/a", "p", status="monitoring")
    changes = [
        {
            "date": dates[24],
            "slug": "a23p",
            "model": "org/a",
            "provider": "p",
            "method": "LT",
            "magnitude": 40.0,
            "magnitude_display": "40σ",
        }
    ]
    (root / "data" / "changes.json").write_text(json.dumps(changes))
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))
    return root


def _views(root):
    lt_endpoints = list(discover_lt_endpoints(root / "data" / "lt"))
    b3it = discover_b3it_views(
        root / "data" / "b3it" / "state", root / "data" / "b3it" / "phase_2"
    )
    rows = build_overview(root, lt_endpoints, b3it)["endpoints"]
    return build_provider_views(root, lt_endpoints, b3it, rows)


def test_variants_group_under_one_provider(fake_site):
    views = _views(fake_site)
    assert list(views) == ["p"]
    view = views["p"]
    assert view["name"] == "p"
    assert view["n_endpoints"] == 2
    assert view["n_models"] == 2
    assert view["n_variants"] == 2
    assert {v["name"] for v in view["variants"]} == {"", "fp8"}


def test_exposure_summed_across_endpoints_and_rate_withheld_when_thin(fake_site):
    view = _views(fake_site)["p"]
    # two endpoints x 29 days each, well under MIN_ENDPOINT_YEARS
    assert view["lt"]["endpoints"] == 2
    assert view["lt"]["changes"] == 1
    assert view["lt"]["years"] < 0.5
    assert view["lt"]["rate"] is None
    assert view["lt"]["ci"] is None


def test_methods_kept_separate(fake_site):
    view = _views(fake_site)["p"]
    assert view["b3it"]["endpoints"] == 1
    assert view["b3it"]["changes"] == 0
    assert view["lt"]["changes"] == 1


def test_rate_published_once_exposure_clears_threshold(tmp_path):
    root = tmp_path / "website"
    dates = [f"2025-{m:02d}-01T00:00:00Z" for m in range(1, 13)] + [
        "2026-01-01T00:00:00Z"
    ]
    write_lt_endpoint(
        root,
        "long23q",
        "org/a",
        "q",
        dates=dates,
        changes=[{"index": 6, "sigma": 30.0}],
        drift=[0.1] * 6 + [1.0] * 7,
    )
    (root / "data" / "changes.json").write_text(
        json.dumps(
            [
                {
                    "date": dates[6],
                    "slug": "long23q",
                    "model": "org/a",
                    "provider": "q",
                    "method": "LT",
                    "magnitude": 30.0,
                    "magnitude_display": "30σ",
                }
            ]
        )
    )
    (root / "data" / "spend.json").write_text(json.dumps({"cumulative": {}}))
    view = _views(root)["q"]
    assert view["lt"]["years"] == pytest.approx(1.0, abs=0.05)
    assert view["lt"]["rate"] == pytest.approx(1.0, abs=0.06)
    assert view["lt"]["ci"][0] < view["lt"]["rate"] < view["lt"]["ci"][1]


def test_monthly_monitoring_counts_match_endpoint_spans(fake_site):
    view = _views(fake_site)["p"]
    assert view["months"] == ["2026-06"]
    for variant in view["variants"]:
        assert variant["monitoring"] == [1]


def test_provider_carries_its_changes_and_endpoint_rows(fake_site):
    view = _views(fake_site)["p"]
    assert [c["date"] for c in view["changes"]] == ["2026-06-25"]
    assert {e["slug"] for e in view["endpoints"]} == {"a23p", "b23p2ffp8"}


def test_overview_rows_are_one_per_provider_with_last_change(fake_site):
    rows = overview_rows(_views(fake_site))
    (row,) = rows
    assert row["name"] == "p"
    assert row["slug"] == "p"
    assert row["n_variants"] == 2
    assert row["lt_rate"] is None
    assert row["last_change"] == "2026-06-25"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_generate_site_provider.py -q`
Expected: FAIL — no module named `provider`

- [ ] **Step 3: Write the implementation**

Create `src/trackllm_website/generate_site/provider.py`:

```python
"""Per-provider aggregation: providers/<slug>.json plus the Overview's provider rows.

A provider is the company -- the part of the OpenRouter provider string before
the "/". Its serving variants (fp8, fp4, ...) are separate serving stacks and
drift separately, so they stay visible as rows inside the provider.

Reads only already-generated data (lt_scores.json's drift/drift_dates, B3IT
build-time views, changes.json) -- never raw logprobs.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.feed import build_feed_items
from trackllm_website.generate_site.lt import EndpointInfo, load_lt_scores
from trackllm_website.generate_site.rates import drift_rate, poisson_interval
from trackllm_website.util import slugify

DAYS_PER_YEAR = 365.25


def base_provider(provider: str) -> str:
    return provider.split("/")[0]


def variant_name(provider: str) -> str:
    """The serving variant, or "" for a provider's default serving stack."""
    return provider.split("/", 1)[1] if "/" in provider else ""


def endpoint_years(first: str, last: str) -> float:
    """Monitoring exposure in endpoint-years; a single observation counts as a day."""
    span = (datetime.fromisoformat(last) - datetime.fromisoformat(first)).days
    return max(1, span) / DAYS_PER_YEAR


def _months(first: str, last: str) -> list[str]:
    out, y, m = [], int(first[:4]), int(first[5:7])
    while f"{y:04d}-{m:02d}" <= last[:7]:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _method_block(endpoints: int, years: float, changes: int) -> dict:
    rate = drift_rate(changes, years)
    ci = poisson_interval(changes, years) if rate is not None else None
    return {
        "endpoints": endpoints,
        "years": round(years, 2),
        "changes": changes,
        "rate": round(rate, 2) if rate is not None else None,
        "ci": [round(ci[0], 2), round(ci[1], 2)] if ci else None,
    }


class _Span:
    """Accumulator for one provider variant."""

    def __init__(self):
        self.slugs: list[str] = []
        self.lt_endpoints = 0
        self.b3it_endpoints = 0
        self.lt_years = 0.0
        self.b3it_years = 0.0
        self.lt_changes = 0
        self.b3it_changes = 0
        self.spans: list[tuple[str, str]] = []


def build_provider_views(
    website_dir: Path,
    lt_endpoints: list[EndpointInfo],
    b3it_views: dict[str, B3ITView],
    endpoint_rows: list[dict],
) -> dict[str, dict]:
    data_dir = website_dir / "data"
    lt_dir = data_dir / "lt"
    lt_by_slug = {e.slug: e for e in lt_endpoints}
    rows_by_slug = {r["slug"]: r for r in endpoint_rows}

    changes_path = data_dir / "changes.json"
    changes = json.loads(changes_path.read_text()) if changes_path.exists() else []

    drift_by_slug: dict[str, list[tuple[datetime, float]]] = {}
    lt_span: dict[str, tuple[str, str]] = {}
    lt_change_count: dict[str, int] = {}
    for slug in lt_by_slug:
        d = load_lt_scores(lt_dir, slug)
        if d is None:
            continue
        drift_by_slug[slug] = [
            (datetime.fromisoformat(s), v)
            for s, v in zip(d.get("drift_dates", []), d.get("drift", []))
        ]
        lt_span[slug] = (d["dates"][0][:10], d["dates"][-1][:10])
        lt_change_count[slug] = len(d["changes"])

    b3it_span: dict[str, tuple[str, str]] = {}
    for slug, view in b3it_views.items():
        dates = view.tv_series["dates"]
        if dates:
            b3it_span[slug] = (dates[0][:10], dates[-1][:10])

    now = max((datetime.fromisoformat(s[1]) for s in lt_span.values()), default=None)
    items = build_feed_items(changes, drift_by_slug, b3it_views, now) if now else []

    by_provider: dict[str, dict[str, _Span]] = defaultdict(
        lambda: defaultdict(_Span)
    )
    models: dict[str, set[str]] = defaultdict(set)
    for slug in sorted(set(lt_by_slug) | set(b3it_views)):
        ep = lt_by_slug.get(slug)
        view = b3it_views.get(slug)
        provider = ep.provider if ep else view.provider
        model = ep.model if ep else view.model
        base, variant = base_provider(provider), variant_name(provider)
        acc = by_provider[base][variant]
        acc.slugs.append(slug)
        models[base].add(model)
        if slug in lt_span:
            acc.lt_endpoints += 1
            acc.lt_years += endpoint_years(*lt_span[slug])
            acc.lt_changes += lt_change_count[slug]
            acc.spans.append(lt_span[slug])
        if slug in b3it_span:
            acc.b3it_endpoints += 1
            acc.b3it_years += endpoint_years(*b3it_span[slug])
            acc.b3it_changes += len(view.changes)
            acc.spans.append(b3it_span[slug])

    views: dict[str, dict] = {}
    for base, variants in by_provider.items():
        spans = [s for acc in variants.values() for s in acc.spans]
        first = min((s[0] for s in spans), default=None)
        last = max((s[1] for s in spans), default=None)
        months = _months(first, last) if first and last else []
        prov_items = [i for i in items if i["providerSlug"] == base]
        slugs = {s for acc in variants.values() for s in acc.slugs}

        variant_out = []
        for name, acc in sorted(
            variants.items(), key=lambda kv: (-len(kv[1].slugs), kv[0])
        ):
            monitoring = [
                sum(1 for lo, hi in acc.spans if lo[:7] <= m <= hi[:7]) for m in months
            ]
            variant_out.append(
                {
                    "name": name,
                    "n_endpoints": len(acc.slugs),
                    "lt": _method_block(acc.lt_endpoints, acc.lt_years, acc.lt_changes),
                    "b3it": _method_block(
                        acc.b3it_endpoints, acc.b3it_years, acc.b3it_changes
                    ),
                    "monitoring": monitoring,
                }
            )

        def _total(attr: str) -> float:
            return sum(getattr(acc, attr) for acc in variants.values())

        views[slugify(base)] = {
            "name": base,
            "slug": slugify(base),
            "n_endpoints": len(slugs),
            "n_models": len(models[base]),
            "n_variants": len(variants),
            "first": first,
            "last": last,
            "months": months,
            "lt": _method_block(
                int(_total("lt_endpoints")), _total("lt_years"), int(_total("lt_changes"))
            ),
            "b3it": _method_block(
                int(_total("b3it_endpoints")),
                _total("b3it_years"),
                int(_total("b3it_changes")),
            ),
            "variants": variant_out,
            "changes": prov_items,
            "endpoints": sorted(
                (rows_by_slug[s] for s in slugs if s in rows_by_slug),
                key=lambda r: (-r["nChanges"], r["model"]),
            ),
        }
    return views


def overview_rows(views: dict[str, dict]) -> list[dict]:
    """Compact provider rows for the Overview's providers section."""
    rows = []
    for view in views.values():
        rows.append(
            {
                "name": view["name"],
                "slug": view["slug"],
                "n_endpoints": view["n_endpoints"],
                "n_models": view["n_models"],
                "n_variants": view["n_variants"],
                "lt_years": view["lt"]["years"],
                "lt_changes": view["lt"]["changes"],
                "lt_rate": view["lt"]["rate"],
                "lt_ci": view["lt"]["ci"],
                "b3it_endpoints": view["b3it"]["endpoints"],
                "b3it_years": view["b3it"]["years"],
                "last_change": view["changes"][0]["date"] if view["changes"] else None,
            }
        )
    rows.sort(
        key=lambda r: (-(r["lt_rate"] if r["lt_rate"] is not None else -1), -r["lt_years"])
    )
    return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_generate_site_provider.py -q`
Expected: 8 passed

- [ ] **Step 5: Lint and commit**

```bash
prek run --all-files
git add src/trackllm_website/generate_site/provider.py tests/test_generate_site_provider.py
git commit --no-verify -m "feat(site): base-provider aggregation with per-variant rates"
```

---

## Task 5: Overview drops its own provider stats

**Files:**
- Modify: `src/trackllm_website/generate_site/overview.py` (delete `PROVIDER_MIN_ENDPOINT_YEARS`, `PROVIDER_CONF_FLOOR`, `PROVIDER_CONF_FULL_YEARS`, the `provider_stats` accumulator and the `providers` list build; add `provider_companies` to `stats`)
- Modify: `tests/test_generate_site_overview.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_overview(...)` now returns `{"stats", "feed", "endpoints"}` — no `providers` key. `stats` gains `provider_companies: int`. `render.py` (Task 6) injects `overview["providers"] = overview_rows(provider_views)`.

- [ ] **Step 1: Update the failing assertions in the existing test**

In `tests/test_generate_site_overview.py`, change `test_build_overview_shape`:

```python
def test_build_overview_shape(fake_site):
    ov = _build_overview(fake_site)
    assert set(ov) == {"stats", "feed", "endpoints"}
```

and add two tests — the stats counts, and the model slug the Model-page links in
Tasks 8–9 need (derived in Python so TypeScript never re-implements `slugify`):

```python
def test_endpoint_rows_carry_model_slug(fake_site):
    ov = _build_overview(fake_site)
    ep = next(e for e in ov["endpoints"] if e["slug"] == "m2fa23p")
    assert ep["modelSlug"] == slugify("m/a")


def test_stats_count_provider_companies_and_variants(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 31)]
    write_lt_endpoint(root, "a23p", "org/a", "p", dates=dates, changes=[], drift=[0.1] * 30)
    write_lt_endpoint(
        root, "b23p2ffp8", "org/b", "p/fp8", dates=dates, changes=[], drift=[0.1] * 30
    )
    (root / "data" / "changes.json").write_text(json.dumps([]))
    ov = _build_overview(root)
    assert ov["stats"]["providers"] == 2  # serving variants
    assert ov["stats"]["provider_companies"] == 1
```

Delete any existing test that asserts on `ov["providers"]` rows (the per-variant
`rate`/`conf` shape is gone); the replacement coverage lives in
`tests/test_generate_site_provider.py::test_overview_rows_are_one_per_provider_with_last_change`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_generate_site_overview.py -q`
Expected: FAIL — `providers` still present, `provider_companies` missing.

- [ ] **Step 3: Edit `overview.py`**

1. Delete the constants `PROVIDER_MIN_ENDPOINT_YEARS`, `PROVIDER_CONF_FLOOR`, `PROVIDER_CONF_FULL_YEARS`.
2. Delete the `provider_stats: dict[str, dict] = defaultdict(...)` declaration, the `ps = provider_stats[provider]` block inside the endpoint loop, the `for c in lt_changes: provider_stats[...]["n_changes"] += 1` loop, and the whole `providers = []` … `providers.sort(...)` block.
3. Add to `stats`:

```python
        "provider_companies": len({r["provider"].split("/")[0] for r in endpoint_recs}),
```

3b. Add `"modelSlug": slugify(full_model),` to each endpoint record, importing
`from trackllm_website.util import slugify`. This is what the Overview
directory, provider page and change log all use to link to model pages.

4. Change the return to:

```python
    return {
        "stats": stats,
        "feed": feed,
        "endpoints": endpoint_recs,
    }
```

5. Remove the now-unused `defaultdict` import if nothing else uses it.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_generate_site_overview.py -q`
Expected: all pass

- [ ] **Step 5: Lint and commit**

```bash
prek run --all-files
git add src/trackllm_website/generate_site/overview.py tests/test_generate_site_overview.py
git commit --no-verify -m "refactor(site): overview delegates provider stats to provider.py"
```

---

## Task 6: `changes_page.py` — data for the global change log

**Files:**
- Create: `src/trackllm_website/generate_site/changes_page.py`
- Test: `tests/test_generate_site_changes_page.py`

**Interfaces:**
- Consumes: `feed.build_feed_items`, `lt.load_lt_scores`, `b3it.B3ITView`.
- Produces: `build_changes_page(website_dir: Path, lt_endpoints: list[EndpointInfo], b3it_views: dict[str, B3ITView]) -> dict` returning `{"stats", "items", "months", "top_endpoints"}`:
  - `stats`: `{total, lt, b3it, endpoints_affected, providers_involved, changes_30d, largest_lt_drift, since, now}`
  - `items`: every enriched feed item, newest first
  - `months`: `[{"month": "2026-06", "lt": 3, "b3it": 1}, …]`, contiguous from the first monitored month to `now`
  - `top_endpoints`: 5 × `{slug, model, provider, providerSlug, modelSlug, n, last}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate_site_changes_page.py`:

```python
import json

import pytest

from conftest import write_lt_endpoint
from trackllm_website.generate_site.b3it import discover_b3it_views
from trackllm_website.generate_site.changes_page import build_changes_page
from trackllm_website.generate_site.lt import discover_lt_endpoints

TOP_N = 5


@pytest.fixture
def fake_site(tmp_path):
    root = tmp_path / "website"
    dates = [f"2026-0{m}-{d:02d}T00:00:00Z" for m in (4, 5, 6) for d in range(1, 11)]
    write_lt_endpoint(
        root,
        "a23p",
        "org/a",
        "p",
        dates=dates,
        changes=[{"index": 5, "sigma": 40.0}, {"index": 22, "sigma": 12.0}],
        drift=[0.1] * 5 + [1.4] * 17 + [2.0] * 8,
    )
    write_lt_endpoint(
        root,
        "b23q",
        "org/b",
        "q",
        dates=dates,
        changes=[{"index": 12, "sigma": 20.0}],
        drift=[0.1] * 12 + [0.9] * 18,
    )
    changes = [
        {
            "date": dates[i],
            "slug": slug,
            "model": model,
            "provider": provider,
            "method": "LT",
            "magnitude": 20.0,
            "magnitude_display": "20σ",
        }
        for i, slug, model, provider in (
            (5, "a23p", "org/a", "p"),
            (22, "a23p", "org/a", "p"),
            (12, "b23q", "org/b", "q"),
        )
    ]
    (root / "data" / "changes.json").write_text(json.dumps(changes))
    return root


def _build(root):
    return build_changes_page(
        root,
        list(discover_lt_endpoints(root / "data" / "lt")),
        discover_b3it_views(
            root / "data" / "b3it" / "state", root / "data" / "b3it" / "phase_2"
        ),
    )


def test_every_change_appears_exactly_once(fake_site):
    page = _build(fake_site)
    assert len(page["items"]) == 3
    assert page["stats"]["total"] == 3
    assert page["stats"]["lt"] == 3
    assert page["stats"]["b3it"] == 0


def test_items_sorted_newest_first(fake_site):
    dates = [i["date"] for i in _build(fake_site)["items"]]
    assert dates == sorted(dates, reverse=True)


def test_month_histogram_totals_equal_the_change_count(fake_site):
    page = _build(fake_site)
    assert sum(m["lt"] + m["b3it"] for m in page["months"]) == page["stats"]["total"]


def test_months_are_contiguous(fake_site):
    months = [m["month"] for m in _build(fake_site)["months"]]
    assert months == ["2026-04", "2026-05", "2026-06"]


def test_top_endpoints_ranked_by_change_count(fake_site):
    top = _build(fake_site)["top_endpoints"]
    assert len(top) <= TOP_N
    assert top[0]["slug"] == "a23p"
    assert top[0]["n"] == 2
    assert top[0]["last"] == max(
        i["date"] for i in _build(fake_site)["items"] if i["slug"] == "a23p"
    )


def test_stats_report_affected_endpoints_and_providers(fake_site):
    stats = _build(fake_site)["stats"]
    assert stats["endpoints_affected"] == 2
    assert stats["providers_involved"] == 2
    assert stats["largest_lt_drift"] == pytest.approx(2.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_generate_site_changes_page.py -q`
Expected: FAIL — no module named `changes_page`

- [ ] **Step 3: Write the implementation**

Create `src/trackllm_website/generate_site/changes_page.py`:

```python
"""data/changes_page.json: the complete change log, its month histogram and rankings.

Reads only already-generated data; the enrichment itself lives in feed.py so the
Overview's latest-changes slice and this log can never disagree.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.feed import build_feed_items
from trackllm_website.generate_site.lt import EndpointInfo, load_lt_scores

TOP_ENDPOINTS = 5
RECENT_DAYS = 30


def _month_range(first: str, last: str) -> list[str]:
    out, y, m = [], int(first[:4]), int(first[5:7])
    while f"{y:04d}-{m:02d}" <= last[:7]:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def build_changes_page(
    website_dir: Path,
    lt_endpoints: list[EndpointInfo],
    b3it_views: dict[str, B3ITView],
) -> dict:
    data_dir = website_dir / "data"
    lt_dir = data_dir / "lt"

    drift_by_slug: dict[str, list[tuple[datetime, float]]] = {}
    all_dates: list[str] = []
    for ep in lt_endpoints:
        d = load_lt_scores(lt_dir, ep.slug)
        if d is None:
            continue
        drift_by_slug[ep.slug] = [
            (datetime.fromisoformat(s), v)
            for s, v in zip(d.get("drift_dates", []), d.get("drift", []))
        ]
        all_dates += [d["dates"][0][:10], d["dates"][-1][:10]]
    for view in b3it_views.values():
        dates = view.tv_series["dates"]
        if dates:
            all_dates += [dates[0][:10], dates[-1][:10]]

    changes_path = data_dir / "changes.json"
    changes = json.loads(changes_path.read_text()) if changes_path.exists() else []

    now = datetime.fromisoformat(max(all_dates)) if all_dates else None
    items = build_feed_items(changes, drift_by_slug, b3it_views, now) if now else []

    months = _month_range(min(all_dates), max(all_dates)) if all_dates else []
    lt_counts = Counter(i["date"][:7] for i in items if i["method"] == "lt")
    b3it_counts = Counter(i["date"][:7] for i in items if i["method"] == "b3it")

    per_endpoint: dict[str, dict] = {}
    for item in items:  # items are newest first, so the first hit is the latest
        rec = per_endpoint.setdefault(
            item["slug"],
            {
                "slug": item["slug"],
                "model": item["model"],
                "provider": item["provider"],
                "providerSlug": item["providerSlug"],
                "modelSlug": item["modelSlug"],
                "n": 0,
                "last": item["date"],
            },
        )
        rec["n"] += 1

    lt_drifts = [i["magnitude"] for i in items if i["method"] == "lt" and i["magnitude"]]
    return {
        "stats": {
            "total": len(items),
            "lt": sum(1 for i in items if i["method"] == "lt"),
            "b3it": sum(1 for i in items if i["method"] == "b3it"),
            "endpoints_affected": len(per_endpoint),
            "providers_involved": len({i["providerSlug"] for i in items}),
            "changes_30d": sum(1 for i in items if i["daysAgo"] < RECENT_DAYS),
            "largest_lt_drift": max(lt_drifts, default=None),
            "since": min(all_dates) if all_dates else None,
            "now": now.strftime("%Y-%m-%d") if now else None,
        },
        "items": items,
        "months": [
            {"month": m, "lt": lt_counts.get(m, 0), "b3it": b3it_counts.get(m, 0)}
            for m in months
        ],
        "top_endpoints": sorted(
            per_endpoint.values(), key=lambda r: (-r["n"], r["slug"])
        )[:TOP_ENDPOINTS],
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_generate_site_changes_page.py -q`
Expected: 6 passed

- [ ] **Step 5: Lint and commit**

```bash
prek run --all-files
git add src/trackllm_website/generate_site/changes_page.py tests/test_generate_site_changes_page.py
git commit --no-verify -m "feat(site): build data for the global change log"
```

---

## Task 7: Render provider pages, the Changes page, and the nav

**Files:**
- Modify: `src/trackllm_website/generate_site/render.py`
- Create: `website/templates/provider.html.j2`, `website/templates/changes.html.j2`
- Modify: `website/templates/base.html.j2` (nav), `website/templates/endpoint.html.j2` (provider link)
- Modify: `tests/test_generate_site_render.py`

**Interfaces:**
- Consumes: `provider.build_provider_views`, `provider.overview_rows`, `changes_page.build_changes_page`.
- Produces: `data/providers/<slug>.json`, `providers/<slug>.html`, `data/changes_page.json`, `changes.html`; `data/overview.json` regains a `providers` key holding `overview_rows(...)`; endpoint template context gains `provider_slug`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generate_site_render.py`:

```python
def test_render_emits_provider_pages_and_data(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path)
    view = json.loads((tmp_path / "data" / "providers" / "p.json").read_text())
    assert view["name"] == "p"
    assert view["n_endpoints"] == 1
    assert (tmp_path / "providers" / "p.html").exists()
    assert 'id="providerData"' in (tmp_path / "providers" / "p.html").read_text()


def test_overview_providers_are_base_provider_rows(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path)
    overview = json.loads((tmp_path / "data" / "overview.json").read_text())
    (row,) = overview["providers"]
    assert row["name"] == "p"
    assert row["slug"] == "p"
    assert "lt_ci" in row


def test_render_emits_changes_page(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path)
    page = json.loads((tmp_path / "data" / "changes_page.json").read_text())
    assert set(page) == {"stats", "items", "months", "top_endpoints"}
    assert (tmp_path / "changes.html").exists()
    assert 'id="log"' in (tmp_path / "changes.html").read_text()


def test_nav_links_to_changes(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path)
    assert 'href="changes.html"' in (tmp_path / "index.html").read_text()


def test_endpoint_page_links_to_its_provider(tmp_path):
    _scaffold(tmp_path)
    render_site(tmp_path)
    html = (tmp_path / "endpoints" / "m2fa23p.html").read_text()
    assert 'href="../providers/p.html"' in html
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_generate_site_render.py -q`
Expected: FAIL — provider data/pages missing.

- [ ] **Step 3: Create `website/templates/provider.html.j2`**

```jinja
{% extends "base.html.j2" %}

{% block title %}{{ provider }} | TrackLLM{% endblock %}

{% block content %}
<div class="wrap">
  <div class="crumb">
    <a href="{{ nav_prefix }}index.html">Home</a><span class="sep">/</span>
    <span style="color:var(--text-muted)">{{ provider }}</span>
  </div>
  <div class="head">
    <div>
      <div class="eyebrow">Provider</div>
      <h1>{{ provider }}</h1>
      <p class="lede" id="lede"></p>
      <div class="summary" id="summary"></div>
    </div>
  </div>
</div>

<main class="wrap">
  <section class="block">
    <div class="sec-head">
      <h2>Drift rate</h2>
      <span class="hint">the two methods are kept apart: different sensitivity, very different monitoring lengths &mdash; pooling them would move the number when coverage grows rather than when behaviour does</span>
    </div>
    <div class="ratecards" id="ratecards"></div>
  </section>

  <section class="block">
    <div class="sec-head">
      <h2>Monitoring &amp; changes over time</h2>
      <span class="hint">the grey area is how many endpoints were under monitoring each month &mdash; the exposure the rate divides by; dots are detected changes, sized by how far the endpoint moved</span>
    </div>
    <div class="panel" id="timeline"></div>
    <div class="legend">
      <span class="k"><i class="bar" style="background:var(--text-dim);opacity:0.4"></i> endpoints under monitoring</span>
      <span class="k"><i style="background:var(--accent)"></i> LT change (size = drift in nats)</span>
      <span class="k"><i style="background:var(--b3it)"></i> B3IT change (size = peak TV)</span>
    </div>
  </section>

  <section class="block">
    <div class="sec-head">
      <h2>Serving variants</h2>
      <span class="hint">a provider's quantizations are separate serving stacks and drift separately</span>
    </div>
    <div class="table-wrap"><table class="dir">
      <thead><tr>
        <th>Variant</th><th class="r">Endpoints</th><th>LT drift rate</th>
        <th class="col-hide">Monitoring</th><th class="r col-hide">Changes</th><th class="r col-hide">B3IT</th>
      </tr></thead>
      <tbody id="variantBody"></tbody>
    </table></div>
  </section>

  <section class="block">
    <div class="sec-head"><h2>Endpoints</h2><span class="hint" id="epCount"></span></div>
    <div class="toolbar">
      <label class="search">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4-4"/></svg>
        <input id="epq" type="text" placeholder="Search model or org…" autocomplete="off">
      </label>
      <div class="chips" id="epChips">
        <span class="chip state-changed" data-f="changed">Ever changed</span>
        <span class="chip state-retired" data-f="retired">Retired</span>
        <span class="chip" data-f="b3it">B3IT</span>
      </div>
    </div>
    <div class="table-wrap"><div class="table-scroll"><table class="dir">
      <thead><tr>
        <th data-sort="model">Model <span class="arr"></span></th>
        <th data-sort="provider" class="col-hide">Variant <span class="arr"></span></th>
        <th data-sort="status">Status <span class="arr"></span></th>
        <th data-sort="nChanges" class="r">Changes <span class="arr"></span></th>
        <th class="col-hide">Methods</th>
        <th data-sort="stableDays" class="r col-hide">Stable for <span class="arr"></span></th>
        <th class="col-hide">Drift from baseline</th>
      </tr></thead>
      <tbody id="epBody"></tbody>
    </table></div><div class="dir-foot" id="epFoot"></div></div>
  </section>
</main>
{% endblock %}

{% block scripts %}
    <script type="application/json" id="providerData">{{ provider_slug|tojson }}</script>
    <script type="module" src="../js/provider.js"></script>
{% endblock %}
```

- [ ] **Step 4: Create `website/templates/changes.html.j2`**

```jinja
{% extends "base.html.j2" %}

{% block title %}Changes | TrackLLM{% endblock %}

{% block content %}
<div class="wrap">
  <div class="crumb">
    <a href="index.html">Home</a><span class="sep">/</span>
    <span style="color:var(--text-muted)">Changes</span>
  </div>
  <div class="head">
    <div>
      <div class="eyebrow">Change log</div>
      <h1>Every change we have detected.</h1>
      <p class="lede" id="lede"></p>
      <div class="summary" id="summary"></div>
    </div>
  </div>
</div>

<main class="wrap">
  <section class="block">
    <div class="sec-head">
      <h2>Timeline</h2>
      <span class="hint">click a month to filter the log below</span>
    </div>
    <div class="panel"><div class="hist" id="hist"></div></div>
    <div class="legend">
      <span class="k"><i class="bar" style="background:var(--accent)"></i> LT — logprob tracking</span>
      <span class="k"><i class="bar" style="background:var(--b3it)"></i> B3IT — border inputs</span>
      <span class="k" id="monthSel" style="color:var(--text-dim)"></span>
    </div>
  </section>

  <section class="block">
    <div class="sec-head">
      <h2>Most-changed endpoints</h2>
      <span class="hint">the overview directory sorts the same field</span>
    </div>
    <div class="board" id="topEndpoints"></div>
  </section>

  <section class="block">
    <div class="sec-head"><h2>Log</h2><span class="hint" id="logCount"></span></div>
    <div class="toolbar">
      <label class="search">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4-4"/></svg>
        <input id="q" type="text" placeholder="Search model, provider, or org…" autocomplete="off">
      </label>
      <div class="chips" id="chips">
        <span class="chip" data-f="lt">LT</span>
        <span class="chip" data-f="b3it">B3IT</span>
        <span class="chip state-changed" data-f="recent">Last 90 days</span>
        <span class="chip state-changed" data-f="big">Large drift</span>
      </div>
    </div>
    <div class="feed" id="log"></div>
  </section>
</main>

<div class="note">Magnitude is the drift level reached after the changepoint — nats for LT, total variation for B3IT. σ is confidence, not size: a change can be certain and behaviourally small.</div>
{% endblock %}

{% block scripts %}
    <script type="module" src="js/changes.js"></script>
{% endblock %}
```

- [ ] **Step 5: Add Changes to the nav in `website/templates/base.html.j2`**

Replace the `.nav-links` block with:

```jinja
        <div class="nav-links">
            <a href="{{ nav_prefix }}index.html">Overview</a>
            <a href="{{ nav_prefix }}changes.html">Changes</a>
            <a href="{{ nav_prefix }}spend.html">Spend</a>
            <a href="https://tchauvin.com/change-detection-llm-apis" target="_blank" rel="noopener">Methodology</a>
            <a href="https://github.com/timothee-chauvin/trackllm_website" target="_blank" rel="noopener">GitHub</a>
            <a href="https://arxiv.org/abs/2512.03816" target="_blank" rel="noopener">Paper</a>
        </div>
```

- [ ] **Step 6: Add the provider link to `website/templates/endpoint.html.j2`**

In the `.compare` block, after the existing model link, add the provider link:

```jinja
  <div class="compare">
    <div class="t"><b>{{ model_name }}</b> <span>is served by {{ n_providers }} provider{{ "s" if n_providers != 1 else "" }} — compare {{ "its" if n_providers == 1 else "their" }} drift side by side.</span></div>
    <div style="display:flex;gap:1rem;flex-wrap:wrap">
      {% if model_slug %}<a class="go" href="{{ nav_prefix }}models/{{ model_slug }}.html">View model page →</a>{% endif %}
      <a class="go" href="{{ nav_prefix }}providers/{{ provider_slug }}.html">All of {{ provider.split("/")[0] }} →</a>
    </div>
  </div>
```

- [ ] **Step 7: Wire `render.py`**

In `src/trackllm_website/generate_site/render.py`:

1. Add imports:

```python
from trackllm_website.generate_site import changes_page as changes_page_mod
from trackllm_website.generate_site import provider as provider_mod
```

2. Load the two new templates next to the others:

```python
    provider_template = env.get_template("provider.html.j2")
    changes_template = env.get_template("changes.html.j2")
```

3. After the `overview.json` write, replace it with the assembled version and emit provider data + pages:

```python
    overview = overview_mod.build_overview(website_dir, endpoints, b3it_views)
    provider_views = provider_mod.build_provider_views(
        website_dir, endpoints, b3it_views, overview["endpoints"]
    )
    overview["providers"] = provider_mod.overview_rows(provider_views)
    (website_dir / "data" / "overview.json").write_text(json.dumps(overview))

    providers_data_dir = website_dir / "data" / "providers"
    providers_data_dir.mkdir(parents=True, exist_ok=True)
    for pslug, view in provider_views.items():
        (providers_data_dir / f"{pslug}.json").write_text(json.dumps(view))

    provider_pages_dir = website_dir / "providers"
    provider_pages_dir.mkdir(parents=True, exist_ok=True)
    for f in provider_pages_dir.glob("*.html"):
        f.unlink()
    for pslug, view in provider_views.items():
        (provider_pages_dir / f"{pslug}.html").write_text(
            provider_template.render(
                provider=view["name"],
                provider_slug=pslug,
                css_path="../style.css",
                body_class="provider",
                nav_prefix="../",
            )
        )
    print(f"Generated {len(provider_views)} provider pages in providers/")

    changes_page = changes_page_mod.build_changes_page(
        website_dir, endpoints, b3it_views
    )
    (website_dir / "data" / "changes_page.json").write_text(json.dumps(changes_page))
    (website_dir / "changes.html").write_text(
        changes_template.render(css_path="style.css", body_class="changes")
    )
    print("Generated changes.html")
```

4. In the endpoint-page loop, pass the provider slug (import `slugify` is already present in `render.py`'s module graph via `util`; add `from trackllm_website.util import slugify` if absent):

```python
            provider_slug=slugify(provider.split("/")[0]),
```

- [ ] **Step 8: Run the render tests**

Run: `uv run pytest tests/test_generate_site_render.py -q`
Expected: all pass

- [ ] **Step 9: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 10: Lint and commit**

```bash
prek run --all-files
git add -A src/trackllm_website/generate_site website/templates tests
git commit --no-verify -m "feat(site): render provider pages and the changes page"
```

---

## Task 8: `components.ts` + `provider.ts`

There is no JS test runner in this repo; correctness is gated by
`bunx tsc --noEmit` plus a real `make build` and an eyeball of the output.

**Files:**
- Create: `website/src/components.ts`, `website/src/provider.ts`
- Modify: `website/package.json`, `website/style.css`

**Interfaces:**
- Consumes: `data/providers/<slug>.json` (Task 3 shape).
- Produces (from `components.ts`, all exported):
  - `sparkline(trace: number[], cap: number, color: string, frac: number | null): string`
  - `rateBar(years: number, rate: number | null, ci: [number, number] | null, max: number): string`
  - `volGrid(years: number): string`
  - `methodBadges(methods: string[]): string`
  - `statusPill(status: string): string`
  - `LT_CAP = 1.5`, `B3IT_CAP = 1.0`, `MIN_ENDPOINT_YEARS = 0.5`
  - `esc(s: string): string`, `relDays(n: number): string`

- [ ] **Step 1: Add the bundler entrypoints**

In `website/package.json`, both scripts become:

```json
    "build": "bun build src/endpoint.ts src/spend.ts src/overview.ts src/model.ts src/provider.ts src/changes.ts --outdir=js --minify --splitting",
    "watch": "bun build src/endpoint.ts src/spend.ts src/overview.ts src/model.ts src/provider.ts src/changes.ts --outdir=js --watch --splitting"
```

- [ ] **Step 2: Write `website/src/components.ts`**

This is the contract Task 9 also compiles against, so it is given in full. The
sparkline geometry is copied from `overview.ts` unchanged (viewBox `0 0 120 34`,
3px padding, 0.13 area fill, 1.5px stroke, `vector-effect="non-scaling-stroke"`)
so directory rows keep rendering identically.

```typescript
export const LT_CAP = 1.5; // nats
export const B3IT_CAP = 1.0; // total variation
export const MIN_ENDPOINT_YEARS = 0.5; // mirrors rates.py

export function esc(s: string): string {
  return String(s).replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] as string
  );
}

export function sparkline(
  trace: number[],
  cap: number,
  color: string,
  frac: number | null
): string {
  if (!trace.length) return '<svg viewBox="0 0 120 34"></svg>';
  const W = 120, H = 34, pad = 3;
  const pts = trace.map((v, i): [number, number] => [
    trace.length === 1 ? W / 2 : (i / (trace.length - 1)) * W,
    H - pad - Math.min(1, Math.max(0, v / cap)) * (H - 2 * pad),
  ]);
  const line = pts
    .map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1))
    .join(" ");
  const mark =
    frac === null
      ? ""
      : `<line x1="${(frac * W).toFixed(1)}" y1="0" x2="${(frac * W).toFixed(1)}" y2="${H}" stroke="${color}" stroke-width="1" stroke-dasharray="2 2" opacity="0.65"/>`;
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
    <path d="${line} L${W} ${H} L0 ${H} Z" fill="${color}" opacity="0.13"/>${mark}
    <path d="${line}" fill="none" stroke="${color}" stroke-width="1.5" stroke-linejoin="round" vector-effect="non-scaling-stroke"/></svg>`;
}

export function rateBar(
  years: number,
  rate: number | null,
  ci: [number, number] | null,
  max: number
): string {
  if (rate === null || years < MIN_ENDPOINT_YEARS) {
    return '<div class="rbar nd"><b>not enough monitoring</b></div>';
  }
  if (rate === 0 && ci) {
    return `<div class="rbar zero"><b>none in ${years.toFixed(1)} ep-yr &middot; &lt;${ci[1].toFixed(2)}/yr</b></div>`;
  }
  const pc = (v: number): number => Math.min(100, (v / max) * 100);
  const band = ci
    ? `<i style="left:${pc(ci[0])}%;width:${Math.max(1, pc(ci[1]) - pc(ci[0]))}%"></i>`
    : "";
  return `<div class="rbar">${band}
    <span style="width:${pc(rate)}%"></span><u style="left:${pc(rate)}%"></u>
    <b>${rate.toFixed(2)}</b></div>`;
}

/** 1 square = one endpoint-month; 12 per row, 3 rows per column group. */
export function volGrid(years: number): string {
  const months = Math.round(years * 12);
  const groups: string[] = [];
  for (let g = 0; g * 36 < months; g++) {
    const rows: string[] = [];
    for (let y = 0; y < 3; y++) {
      const base = g * 36 + y * 12;
      if (base >= months) break;
      rows.push(
        '<span class="yr">' +
          '<i class="sq"></i>'.repeat(Math.min(12, months - base)) +
          "</span>"
      );
    }
    groups.push('<span class="grp">' + rows.join("") + "</span>");
  }
  const low = years < MIN_ENDPOINT_YEARS ? " low" : "";
  return `<span class="vol"><span class="grid">${groups.join("")}</span>
    <span class="lbl${low}">${years.toFixed(1)} ep-yr</span></span>`;
}

export function methodBadges(methods: string[]): string {
  return methods
    .map((m) => `<span class="badge ${m}">${m === "lt" ? "LT" : "B3IT"}</span>`)
    .join("");
}

export function statusPill(status: string): string {
  return `<span class="pill ${status}"><span class="led"></span>${status}</span>`;
}

export function relDays(n: number): string {
  if (n < 1) return "today";
  if (n < 30) return `${n}d ago`;
  if (n < 365) return `${Math.round(n / 30)}mo ago`;
  return `${(n / 365).toFixed(1)}y ago`;
}
```

Note the `rateBar(years, rate, ci, max)` argument order — it takes the
already-computed `rate`/`ci` from the JSON rather than recomputing them, so the
Python gate in `rates.py` stays the single source of truth.

- [ ] **Step 3: Add the new CSS**

Append to `website/style.css` the classes the new pages need, following the
existing token vocabulary — `.rbar` (+ `.zero`, `.nd`, and its `i/span/u/b`
children), `.boards`, `.board`, `.brow`, `.ratecards`, `.ratecard`, `.tl`,
`.tlrow`, `.tlaxis`, `.hist`, `.mohead`, `.delta-note`. Use only existing custom
properties; define no new colours.

- [ ] **Step 4: Write `website/src/provider.ts`**

```typescript
export {};

interface MethodBlock {
  endpoints: number;
  years: number;
  changes: number;
  rate: number | null;
  ci: [number, number] | null;
}

interface Variant {
  name: string;
  n_endpoints: number;
  lt: MethodBlock;
  b3it: MethodBlock;
  monitoring: number[];
}

interface ProviderChange {
  date: string;
  method: "lt" | "b3it";
  magnitude: number | null;
  model: string;
  provider: string;
  slug: string;
}

interface EndpointRow {
  slug: string;
  model: string;
  org: string;
  provider: string;
  methods: string[];
  status: string;
  stableDays: number | null;
  nChanges: number;
  trace: number[];
}

interface ProviderData {
  name: string;
  slug: string;
  n_endpoints: number;
  n_models: number;
  n_variants: number;
  first: string | null;
  last: string | null;
  months: string[];
  lt: MethodBlock;
  b3it: MethodBlock;
  variants: Variant[];
  changes: ProviderChange[];
  endpoints: EndpointRow[];
}
```

Then an `init()` that fetches `../data/providers/${slug}.json` (slug read from
`#providerData`) and renders, in order:

1. `#lede` — endpoint / model / variant counts and the monitored range. When two
   variants both have a rate and the highest exceeds the lowest by more than
   1.8×, append the sentence naming both, since that contrast is the page's most
   useful finding.
2. `#summary` — endpoints, still active (`status !== "retired"`), changes
   detected, endpoints affected (`nChanges > 0`).
3. `#ratecards` — one card per method: rate to 2dp with unit
   "changes / endpoint-year", the 95% interval (append " (rule of three)" when
   `changes === 0`), and the exposure line. Under the threshold render the
   dashed `.ratecard.nd` variant reading "Not enough monitoring" with the
   accumulated endpoint-years and changes-so-far.
4. `#timeline` — one `.tlrow` per variant: label (`name` or the bare provider
   for the default variant) + endpoint count, an SVG whose grey area is
   `variant.monitoring` across `months`, change dots on top coloured by method
   and sized `2.6 + min(1, magnitude / cap) * 3.2`, and the change count.
   Follow with a `.tlaxis` labelling every third month.
5. `#variantBody` — variant, endpoints, `rateBar`, `volGrid`, changes, B3IT endpoints.
6. `#epBody` — the endpoint directory with search (`#epq`), chips (`#epChips`),
   sortable headers (`th[data-sort]`), model names linking to
   `../models/${slugify(org + "/" + model)}.html` and rows to
   `../endpoints/${slug}.html`. Reuse the sort/filter shape from `overview.ts`.

- [ ] **Step 5: Typecheck and build**

Run: `cd website && bunx tsc --noEmit && cd .. && make build`
Expected: no type errors; build prints "Generated N provider pages in providers/".

- [ ] **Step 6: Verify a real page**

Run: `uv run python -c "import json;d=json.load(open('website/data/providers/chutes.json'));print(d['name'], d['lt'], d['n_variants'])"`
Expected: `chutes {'endpoints': 82, ...}` with a non-null `rate` and `ci`.

- [ ] **Step 7: Lint and commit**

```bash
prek run --all-files
git add -A website src tests
git commit --no-verify -m "feat(site): provider page frontend"
```

---

## Task 9: `changes.ts` — the change log frontend

**Files:**
- Create: `website/src/changes.ts`
- Modify: `website/src/overview.ts`

**Interfaces:**
- Consumes: `data/changes_page.json` (Task 5 shape), `components.ts` (Task 7).
- Produces: no exports; hydrates `#lede`, `#summary`, `#hist`, `#topEndpoints`, `#log`, `#logCount`, `#monthSel`.

- [ ] **Step 1: Write `website/src/changes.ts`**

Fetch `data/changes_page.json`, then:

1. `#lede` and `#summary` from `stats` — changes, endpoints affected, providers
   involved, last 30 days, largest LT drift (nats).
2. `#hist` — one `.mo` column per entry in `months`, stacking a `.b3` block over
   a `.lt` block scaled to the tallest month, with the total above and the month
   label below. Clicking a column filters the log to that month and dims the
   others; clicking the same column again clears the filter and updates
   `#monthSel`.
3. `#topEndpoints` — five `.brow` rows linking to
   `models/${modelSlug}.html`, showing the change count and last change date.
4. `#log` — every item, grouped by `date.slice(0,7)` with a sticky `.mohead` per
   month carrying that month's count under the current filters. Each row is the
   `.event` markup already used by `overview.ts`: date + relative age, model
   linking to `models/${modelSlug}.html` and `@ provider` linking to
   `providers/${providerSlug}.html`, the sparkline (`LT_CAP` / `B3IT_CAP`, dashed
   marker at `changeFrac`), the method badge, `primary` as the magnitude and
   `secondary` as confidence. Severity drives `--sev` from `sevKey`.
5. Search over model + provider + org; chips for LT, B3IT, last 90 days
   (`daysAgo <= 90`), and large drift (`magnitude >= 0.8` for LT,
   `>= 0.6` for B3IT). LT and B3IT chips filter only when exactly one is active.

- [ ] **Step 2: Point the Overview at the new pages**

In `website/src/overview.ts`:

1. Update the `ProviderRate` interface to the Task 3 `overview_rows` shape
   (`name, slug, n_endpoints, n_models, n_variants, lt_years, lt_changes,
   lt_rate, lt_ci, b3it_endpoints, b3it_years, last_change`) and render the
   provider section from it: the two ranked boards (`Most drift-prone` by
   `lt_rate` desc among rows with `lt_changes > 0`; `Nothing detected yet` among
   rows with `lt_changes === 0 && lt_years >= 1`, ranked by `lt_years` desc),
   then the sortable table with the `has changes` / `rateable` / `runs B3IT`
   chips. Rows with `lt_rate === null` sort to the bottom and render the
   "not enough monitoring" bar.
2. Link the directory: model name → `models/${modelSlug}.html`, provider name →
   `providers/${providerSlug}.html`. **Never derive the slug in TypeScript.**
   Provider pages are keyed `slugify(base_provider(provider))` in Python; a
   `provider.split("/")[0]` in TS reintroduces the 404 that Task 7 fixed. Read
   `slug` from the provider row, and for directory rows use the `providerSlug`
   the endpoint record carries (add it in `overview.py` alongside `modelSlug` if
   it is not there yet).
3. Point the two placeholder `href="#"` links in `website/templates/index.html.j2`
   at `changes.html` ("Full changelog →") and remove the "All providers →" link,
   since the section below it is now the full list.
4. Replace the imports of the local sparkline/badge helpers with
   `components.ts`.

- [ ] **Step 3: Typecheck and build**

Run: `cd website && bunx tsc --noEmit && cd .. && make build`
Expected: clean; `changes.html` and `js/changes.js` exist.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 5: Lint and commit**

```bash
prek run --all-files
git add -A website
git commit --no-verify -m "feat(site): change log frontend and overview provider section"
```

---

## Task 10: Model page edits

**Files:**
- Modify: `src/trackllm_website/generate_site/model.py`, `website/src/model.ts`
- Modify: `tests/test_generate_site_model.py`

**Interfaces:**
- Consumes: `provider.base_provider`.
- Produces: each model endpoint gains `base: str` (provider company) and
  `providerSlug: str`; the model view gains
  `changes: list[{date, method, provider}]` — every change for the model,
  flattened and sorted ascending.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generate_site_model.py`:

The file has no shared fixture — every test builds its tree inline from
`tmp_path` and looks the view up by `slugify("m/a")`. Follow that:

```python
def _two_variant_model(root: Path):
    dates = [f"2026-06-{d:02d}T00:00:00Z" for d in range(1, 21)]
    write_lt_endpoint(
        root,
        "m2fa23chutes",
        "m/a",
        "chutes",
        dates=dates,
        changes=[{"index": 15, "sigma": 12.0}],
        drift=[0.1] * 15 + [1.2] * 5,
    )
    write_lt_endpoint(
        root,
        "m2fa23chutes2ffp8",
        "m/a",
        "chutes/fp8",
        dates=dates,
        changes=[{"index": 10, "sigma": 30.0}],
        drift=[0.1] * 10 + [0.9] * 10,
    )


def test_model_endpoints_carry_base_provider(tmp_path):
    root = tmp_path / "website"
    _two_variant_model(root)
    view = _build_model_views(root)[slugify("m/a")]
    assert {e["base"] for e in view["endpoints"]} == {"chutes"}
    assert {e["providerSlug"] for e in view["endpoints"]} == {"chutes"}
    assert {e["provider"] for e in view["endpoints"]} == {"chutes", "chutes/fp8"}


def test_model_view_has_flattened_change_list(tmp_path):
    root = tmp_path / "website"
    _two_variant_model(root)
    view = _build_model_views(root)[slugify("m/a")]
    dates = [c["date"] for c in view["changes"]]
    assert dates == sorted(dates)
    assert sum(e["n_changes"] for e in view["endpoints"]) == len(view["changes"])
    assert {c["method"] for c in view["changes"]} == {"lt"}
    assert {c["provider"] for c in view["changes"]} == {"chutes", "chutes/fp8"}
```

Add `from conftest import write_lt_endpoint` to the imports (Task 3 put it there).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_generate_site_model.py -q`
Expected: FAIL — `KeyError: 'base'`

- [ ] **Step 3: Edit `model.py`**

In `_build_endpoint`, add to the returned dict:

```python
            "base": base_provider(provider),
            "providerSlug": slugify(base_provider(provider)),
```

and in `build_model_views`, after `endpoints.sort(...)`, build the flattened list:

```python
        changes = sorted(
            [
                {"date": c["date"], "method": "lt", "provider": e["provider"]}
                for e in endpoints
                if e["lt"]
                for c in e["lt"]["changes"]
            ]
            + [
                {"date": c["date"], "method": "b3it", "provider": e["provider"]}
                for e in endpoints
                if e["b3it"]
                for c in e["b3it"]["changes"]
            ],
            key=lambda c: c["date"],
        )
```

and add `"changes": changes,` to the emitted view. Import `base_provider` from
`provider.py` and `slugify` from `util`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_generate_site_model.py -q`
Expected: all pass

- [ ] **Step 5: Apply the five `model.ts` edits**

1. **Dots at their drift level.** In `strip()`, replace the fixed `cy = H / 2`
   with the y of the change's own magnitude on that strip's scale — reuse the
   same expression the line path uses:
   `const y = (v: number): number => H - pad - (Math.min(v, dmax) / dmax) * (H - 2 * pad);`
   then `cy = y(c.drift)` for LT and `cy = y(c.peakTV)` for B3IT, with a faint
   vertical stem from the baseline to the dot.
2. **All-providers strip.** Above the provider rows, render a `.allrow`: the
   label "All providers" with the total change count, an SVG marking every entry
   of the new `changes` array on the shared x-axis (LT accent, B3IT purple), and
   the `n_changed / n_providers` ratio.
3. **Group by provider company.** Group `endpoints` by `base`; emit a `.grp-h`
   header per group linking to `../providers/${providerSlug}.html`, and only
   when the group has more than one endpoint or its provider string carries a
   variant. Order groups by total change count descending.
4. **Provider names link to the provider page** in each row's `.pv`, keeping the
   existing endpoint-page link reachable from the row.
5. **Meta column shows the last change date** when `n_changes > 0`, the peak
   drift otherwise.

Add the matching interface fields to `ModelEndpoint` (`base`, `providerSlug`)
and `ModelData` (`changes: {date: string; method: "lt" | "b3it"; provider: string}[]`).

- [ ] **Step 6: Typecheck, build, run everything**

Run: `cd website && bunx tsc --noEmit && cd .. && make build && uv run pytest -q`
Expected: clean typecheck, successful build, all tests pass.

- [ ] **Step 7: Lint and commit**

```bash
prek run --all-files
git add -A src website tests
git commit --no-verify -m "feat(site): model page groups providers and places dots at drift level"
```

---

## Final verification

- [ ] `uv run pytest -q` — full suite green.
- [ ] `cd website && bunx tsc --noEmit` — clean.
- [ ] `make build` — completes, printing provider page and changes page lines.
- [ ] `prek run --all-files` — clean.
- [ ] Spot-check the built output:
  - `website/providers/chutes.html` exists; `website/data/providers/chutes.json` has a non-null `lt.rate` and `lt.ci`, and more than one variant.
  - `website/changes.html` exists; `website/data/changes_page.json`'s `stats.total` equals `len(website/data/changes.json)`.
  - `website/data/overview.json`'s `providers` has one row per company, each with `lt_ci`.
  - An endpoint page contains both a `models/` and a `providers/` link.
