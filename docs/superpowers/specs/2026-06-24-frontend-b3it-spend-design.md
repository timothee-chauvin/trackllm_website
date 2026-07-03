# Design: frontend for B3IT, change feed, and spend

Goal: surface the backend work of PRs 1–17 on the website. Today the site is
LT-only — an index of LT-tracked endpoints plus per-endpoint logprob and
anomaly-score charts. Three backend data sources are emitted but shown nowhere:
B3IT epoch state (`website/data/b3it/state/*.json`), the LT change feed
(`website/data/lt/lt_changes.json`), and the universal spend ledger
(`website/data/spend/{slug}/{YYYY-MM}.jsonl`).

This implements the "Website" step deferred from the 2026-06-11 change-detection
design (per-endpoint B3IT section, merged change feed, instability badge,
unified index) **plus** a spend dashboard (deferred as YAGNI in the
2026-06-22 ledger design, now buildable since the ledger exists).

## Core principle: build-time derivation (facts vs derivations)

Consistent with the two prior designs: state files and the ledger store only
**facts**. Every displayed B3IT value (TV-over-time series, change markers,
instability flag, current BI count, normalized status) and every spend
aggregate is **derived at build time** by the site generator, using the current
algorithm in `bi/detection.py` and the helpers in `spend.py`. The browser only
fetches small pre-derived JSON and renders it — it never sees raw reference
samples, runs the detection algorithm, or aggregates the ledger.

Rejected alternatives: client-side derivation (TV needs tokenizers + the
detection algorithm, which are Python-only, and ships large reference payloads
to the browser); a hybrid that assembles the change feed in JS (Python already
holds both sources — no reason to split the merge across the boundary).

## Build pipeline: split `generate_site.py` into a package

`generate_site.py` (246 lines, LT-only) becomes a `generate_site/` package so
each method's derivation is an independently testable, focused module
(per project practice — many small modules over one growing file):

- `generate_site/lt.py` — today's LT endpoint discovery + per-prompt manifest
  logic, moved verbatim.
- `generate_site/b3it.py` — read `b3it/state/*.json`, derive per-endpoint
  display data (below) via `bi.detection`.
- `generate_site/changes.py` — merge LT + B3IT change events into one feed.
- `generate_site/spend.py` — aggregate the ledger for the strip and `/spend`.
- `generate_site/render.py` — Jinja rendering + JSON emission.
- `generate_site/__main__.py` — orchestration (`uv run python -m
  trackllm_website.generate_site`); `make build` updated to match.

The endpoint set rendered is the **union** of LT-discovered and B3IT-state
endpoints, keyed by slug. Each endpoint page renders only the sections it has
data for.

## Emitted artifacts (small, derived, git-committed by the deploy/run pipeline)

1. **`website/data/b3it/<slug>/b3it.json`** — per endpoint, analogous to
   `lt_scores.json`:
   ```
   {
     "status": "monitoring" | "retired",
     "retired_reason": "stalled" | "no_bis" | "delisted" | null,
     "n_bis": 18,
     "unstable": false,
     "epochs": [{"start", "end", "end_reason", "change_date"}],
     "tv_series": {"dates": [...], "values": [...]},   // current epoch, may be empty
     "changes": [{"date", "kind": "onset"}]
   }
   ```
   Derivation: for the current (open) epoch, `tv_series = epoch_tv_series(
   epoch.reference, results)` where `results` is loaded from
   `b3it/phase_2/<slug>/` exactly as `monitor.py` loads it (reuse that loader,
   do not reimplement); `changes = adaptive_transitions(tv_series)`;
   `unstable = is_unstable(tv_series)`; `n_bis = len(current_epoch.border_inputs)`.
   `epochs` is the full epoch timeline (for boundary markers) regardless of
   whether each has a reconstructable TV series.

2. **`website/data/changes.json`** — merged feed, newest first:
   ```
   [{"date", "slug", "model", "provider", "method": "LT"|"B3IT", "magnitude"}]
   ```
   LT events come from `lt/lt_changes.json` (`magnitude` = `sigma`, displayed
   `∞` when non-finite, as the existing LT UI already does). B3IT events are
   epoch boundaries with a `change_date` (`end_reason == "change_detected"`);
   `magnitude` is the TV jump at onset where derivable, else null.

3. **`website/data/spend.json`** — `{cumulative, last_30d, daily, by_endpoint}`,
   each split into kind-groups: **onboarding** (`onboard`+`recheck`+`reinit`),
   **monitoring** (`monitor`), **LT** (`lt`), **vetting** (`vetting`).
   `cumulative`/`last_30d` reuse `spend.cumulative_by_kind` / `today_by_kind`
   shapes; `daily` (date→group→$) and `by_endpoint` (slug→group→$) need a small
   new build-time aggregator in `generate_site/spend.py` that walks
   `spend/*/*.jsonl` once (same glob as the existing helpers).

## Status vocabulary (normalized for the UI)

The two methods have different raw status words. The UI normalizes both to a
shared vocabulary: an endpoint is **monitoring** under a method when that method
is actively querying it (LT: last query within `INACTIVE_THRESHOLD_DAYS`; B3IT:
`status == "monitoring"`), otherwise **retired** (LT: stale; B3IT: the
`retired.reason`). "—" means the method does not cover that endpoint.

## Index (homepage)

Top-to-bottom:

1. **Recent changes feed** (from `changes.json`): rows of date, endpoint
   (linked), method tag, magnitude — newest first, capped to a recent window.
   Empty-state copy when there are none.
2. **Spend strip**: cumulative + last-30-day totals, split by kind-group,
   linking to `/spend`.
3. **Unified endpoint table**: one row per endpoint (alphabetical) with `LT` and
   `B3IT` columns showing the normalized status (`monitoring` /
   `retired(reason)` / `—`). Rows with a change in the last N days are
   highlighted/badged. Replaces today's two LT-only Currently/Previously tables.

## Endpoint page

`endpoint.ts` gains a B3IT section rendered from `b3it.json`: a TV-over-time
Plotly chart (mean TV vs the epoch reference) with epoch-boundary lines and
change markers — reusing the existing `makeChangeShapes` helper for visual
consistency with the LT anomaly chart — plus current BI count, normalized
status, and an instability badge. The existing LT sections are unchanged. The
page header shows which method(s) cover the endpoint; sections with no data are
omitted (LT-only, B3IT-only, or both).

## `/spend` page

A new top-level page (new template + a small TS entry that fetches
`spend.json`): total cumulative spend; a split-by-kind-group breakdown; a daily
spend time-series chart (stacked by group); and a sortable per-endpoint table.
Linked from the index spend strip.

## Empty / partial-data handling (first-class)

Current reality: all 53 B3IT state files are migrated → `retired`, epoch 0 only,
`end_reason: "gap"`, `reference` empty (`n_ref == 0`), `change_date: null`; the
spend ledger so far holds only `vetting` rows. So:

- `epoch_tv_series` returns `[]` for reference-less epochs → `b3it.json` carries
  an empty `tv_series`. The B3IT section then shows the epoch timeline + status
  + "no TV data for this epoch" rather than an empty chart or an error.
- The changes feed and `/spend` render meaningfully with sparse/zero data and
  light up automatically as live monitoring and onboarding accumulate.
- No endpoint page, the index, or `make build` may error on missing/empty
  artifacts — absence is a normal state, never a failure (per global guidance:
  never silence a *real* error, but empty data here is expected, not an error).

## Testing (TDD, tests first)

Per project practice, with fixtures extracted from the real data:

- `b3it.py`: TV/epoch/status/instability derivation against a real monitoring
  fixture and against a reference-less migrated fixture (→ empty `tv_series`,
  full epoch timeline, correct retired reason).
- `changes.py`: merge + newest-first sort across LT and B3IT events; non-finite
  LT sigma passthrough; empty inputs → empty feed.
- `spend.py` (generator): `daily` and `by_endpoint` aggregation across multiple
  slug/month JSONL files; kind→group mapping; empty ledger → zeroed structure.
- `render.py`: union endpoint set; section omission for single-method
  endpoints; index renders with zero changes / zero spend.
- `make build` smoke-passes end-to-end on the committed data.

## Rollout order (independently shippable PRs)

1. **Build refactor**: split `generate_site.py` into the package, LT behavior
   unchanged (pure refactor, existing LT tests green). No visible change.
2. **B3IT endpoint section**: `b3it.py` + `b3it.json` + endpoint-page TV chart +
   normalized status on the (still LT-shaped) index.
3. **Unified index + change feed**: `changes.py` + `changes.json` + homepage
   feed + unified endpoint table + recent-change badge.
4. **Spend dashboard**: `generate_site/spend.py` aggregator + `spend.json` +
   index strip + `/spend` page.

## Out of scope (YAGNI)

- New backend data or algorithm changes — the frontend consumes only what
  PRs 1–17 already emit.
- The broader information-architecture redesign (considered and declined).
- LT logprob/anomaly chart changes — left exactly as they are.
- Per-BI drill-downs, individual border-input prompt displays.
