# Implementation plan: endpoint chart readability

Spec: `docs/superpowers/specs/2026-07-31-endpoint-chart-readability-design.md` — read it first.

## Global constraints

- Front-end only. Python pipeline, generated JSON, and `make build` inputs are untouched.
- Tests first, then code to green. `cd website && bun test`.
- After editing code run `prek run --all-files`; commit with `--no-verify` (stale pre-commit hook).
- Succinct code, ~10% comments, no repeated blocks, no default argument values — one source
  of truth per constant, at module top level.
- `website/test/mobile.test.ts` imports `chartSvg` from `../src/endpoint`; that import path
  must keep working after the split.
- Geometry constants (`LANE_H`, `TOP1`, `TOP2`, `VH`, `MONTH_LABEL_W`, `CHAR_W`, `ROW_H`,
  `DESIGN_VW`, `NARROW_VW`, `MIN_VW`) live in `endpoint/chart.ts` and are imported by the
  other modules — never redeclared.
- No new tab stops: hit rects are `aria-hidden="true"`.

## Task 0: split `endpoint.ts` into modules — behaviour-preserving

Files: new `website/src/endpoint/{data,chart,ticks,marks,hover}.ts`, rewritten
`website/src/endpoint.ts`.

This task lands **the interfaces the parallel tasks build against**, with stub bodies where
the behaviour is new. Nothing about the rendered output changes.

1. Move code as the spec's table says. No logic edits. `endpoint.ts` keeps `init`,
   `renderStatusCard`, `renderChart`, `renderChangesTable`, and re-exports `chartSvg`,
   `buildLT`, `buildB3IT` so existing tests and the bundler entrypoint are unaffected.
2. `ticks.ts` ships the real signature and a stub body:
   ```ts
   export interface Tick { t: number; label: string }
   export function timeTicks(d0: number, d1: number, maxLabels: number): Tick[]
   ```
   Stub delegates to `monthTicks` with the existing `Mon YY` label, so output is unchanged.
   `chart.ts` calls it for both gridlines and axis labels, dropping its own `step` thinning
   with `maxLabels = Math.floor(PW / MONTH_LABEL_W)`.
3. `marks.ts` ships the real signature and a stub body:
   ```ts
   export interface Mark { x: number; y: number; labelX: number; labelY: number | null; col: string; lab: string }
   // labelY null => label dropped (dot and rule still drawn)
   export function packMarks(
     lanes: { series: [string, number][]; changes: { date: string; lab: string }[];
              topY: number; maxV: number; col: string; titleW: number }[],
     fx: (s: string) => number, vw: number
   ): Mark[]
   ```
   Stub reproduces today's behaviour exactly: `y = topY - 4`, `labelY` from the existing
   fixed-baseline `rowEnd` packing.
4. `hover.ts` ships the real signatures and inert bodies:
   ```ts
   export function nearestPoint(series: [string, number][], xPx: number, fx: (s: string) => number): number
   export function hitRects(lanes: HoverLane[]): string  // "" for now
   export function bindHover(chartEl: HTMLElement, tipEl: HTMLElement, lanes: HoverLane[]): void  // no-op for now
   ```
5. Verify: `bun test` green with **no test edits**, and `chartSvg` output byte-identical for
   the `mobile.test.ts` fixtures (assert this by capturing the string before and after).

**Commit:** `refactor(website): split endpoint chart into modules`

## Task 1: adaptive x axis

Files: `website/src/endpoint/ticks.ts`, new `website/test/ticks.test.ts`.

1. Tests first:
   - 12-day span, `maxLabels = 18` → ≥3 ticks, every label matches `/^[A-Z][a-z]{2} \d{1,2}$/`.
   - 3-month span, `maxLabels = 18` → ≥3 ticks, ≤ `maxLabels`, ticks strictly increasing.
   - 2-year span, `maxLabels = 18` → ≤ `maxLabels`, every label matches `/^[A-Z][a-z]{2} '\d{2}$/`.
   - Single-day span → ≥1 tick.
   - Narrow budget (`maxLabels = 3`) over 12 days → ≤3 ticks, still non-empty.
   - Every case: ticks lie within `[d0, d1]` and are sorted.
2. Implement the step ladder from the spec (1d, 2d, 3d, 7d, 14d, 1mo, 2mo, 3mo, 6mo, 12mo)
   as a module-level constant. First step whose count ≤ `maxLabels` wins; the last rung is the
   fallback so the result is never empty.
3. Day/week steps: aligned to whole UTC days from `d0`, label `Mon D`. Month steps:
   first-of-month, label `Mon 'YY` (reuse the `Mon 'YY` formatter that already exists near
   `components.ts:349`, or lift it into `ticks.ts` if that keeps the caller simpler).
4. `chart.ts` needs no edit — Task 0 already routes both gridlines and labels through
   `timeTicks`. Confirm the reported endpoint's 12-day span now yields day labels.

**Commit:** `fix(website): date the endpoint chart's x axis on short spans`

## Task 2: change marks anchored to the curve

Files: `website/src/endpoint/marks.ts`, new `website/test/marks.test.ts`.

1. Tests first, using a B3IT-shaped fixture (12 daily points rising 0.14 → 0.5355, one change
   on the peak day):
   - The mark's `y` equals the lane's `yv(0.5355)` within 1px, and is **not** `topY - 4`.
   - A change dated after the last sample keeps the lane-top fallback (`y === topY - 4`).
   - A lane with an empty series keeps the lane-top fallback for all its changes.
   - Two changes on adjacent days with overlapping labels: the second's `labelY` differs from
     the first's by `ROW_H` (lifted) or sits below its dot — never equal to the first's.
   - A change near the peak whose label would clear the lane top is placed below its dot.
   - When neither above nor below is free, `labelY` is `null` and the caller still gets `x`.
   - `labelX` stays inside `[0, vw]`.
2. Implement: anchor = value of the sampled point nearest the change date, `labelY =`
   `y - LABEL_DY`; then the packing rule from the spec. Keep placed label boxes in a per-lane
   list of `{x0, x1, y}`; overlap = x-intervals intersect **and** `|Δy| < ROW_H`. Lane title
   occupies `{x0: PL, x1: PL + titleW, y: topY - 8}` when the lane has a series.
3. `chart.ts` consumes `Mark[]` unchanged from Task 0 — verify the dashed rule and dot are
   still drawn for a dropped label.

**Commit:** `fix(website): anchor endpoint change marks to their lane's curve`

## Task 3: pointer readout

Files: `website/src/endpoint/hover.ts`, `website/templates/endpoint.html.j2`,
`website/style.css`, new `website/test/hover.test.ts`.

1. Tests first:
   - `nearestPoint`: exact hit on a sample; midpoint between two samples resolves to one of
     them deterministically; x left of the first sample → 0; x right of the last → last index.
   - `hitRects` emits one rect per lane **with data**, each `aria-hidden="true"`, and none for
     a lane without a series.
   - Glue: build the chart in happy-dom, dispatch a `pointermove` on the B3IT hit rect at the
     peak day's x, assert the tip is no longer `hidden` and its text contains the date and
     `TV 0.536`-shaped value.
   - A `pointerleave` re-hides the tip.
2. Template: `.chartwrap` gains a `chart-tip` div (`id="charttip"`, `hidden`).
   `style.css`: `.chartwrap { position: relative }` and `.chart-tip` modelled on `.hero-tip`
   (`style.css:130`) — absolute, `pointer-events: none`, `z-index: 2`.
3. Implement `hitRects` (transparent rect per lane over its plot area, drawn last in the SVG)
   and `bindHover`: `pointermove` for mouse/pen, `pointerdown` for touch, `pointerleave` to
   hide, a `pointerdown` outside any lane to dismiss a pinned tip. Marker dot is drawn into a
   dedicated `<g>` the handler rewrites, not by re-rendering the chart. Tip text is
   `${date} · ${fmtTV(v)}` / `${date} · ${fmtDrift(v)}`; tip position clamped to the chart box
   like `hero-tip`.
4. `endpoint.ts`: merge draw + bind into one function the resize handler calls whole, so
   wiring survives a redraw.

**Commit:** `feat(website): show date and value on hover or tap in the endpoint chart`

## Task 4: integrate and verify

1. `bun test` fully green; `prek run --all-files`.
2. `make build`, then headless-chromium screenshots of
   `website/endpoints/z-ai2fglm-5.223cloudflare.html` at 1280px and 375px. Check: the
   `TV 0.535` label sits on the curve; the axis reads dates; hovering the B3IT lane shows a
   date and value; nothing overlaps at 375px.
3. Check a second endpoint with **both** lanes and a long span (pick one from
   `website/data/lt/` that also has `b3it.json`) to confirm month labels still read `Jul '26`
   and both lanes' readouts work.
4. Screenshots to the user before the PR.

**Commit:** `test(website): visual acceptance for the endpoint chart fixes`

## Subagent dispatch

Tasks 1, 2 and 3 own **disjoint files** by construction, which is the whole point of Task 0.
That makes them safe to run concurrently in the same worktree.

| step | who | why |
| --- | --- | --- |
| Task 0 | main session, alone | It rewrites every file the others touch. Nothing may run in parallel with it. |
| Tasks 1, 2, 3 | three subagents, one message, concurrent | Disjoint files, disjoint tests, all built against signatures Task 0 froze. |
| Task 4 | main session, alone | Only the integrator sees all three landed. |

Each subagent gets: the spec path, the plan path, its task's section verbatim, its file list,
and the instruction to write tests first and stop at green. Explicit prohibition in every
brief: **do not edit `endpoint/chart.ts`, `endpoint.ts`, or any file outside your list** — a
signature that turns out wrong is reported back, not fixed unilaterally.

Task 3 is the odd one out: it also touches `endpoint.html.j2`, `style.css`, and (in step 4)
`endpoint.ts`. Step 4 of Task 3 is therefore **pulled into Task 4** and done by the main
session, so no subagent writes `endpoint.ts`.

Cost of the split: Task 0 is ~20 minutes of pure refactor before any bug is fixed. It pays for
itself here because all three fixes otherwise land in the same 130-line function, and because
`endpoint.ts` at 421 lines is already past the point where it should have been split.

If the three tasks are run sequentially instead, drop Task 0 and edit `endpoint.ts` in place —
the refactor is worth doing either way, but it is not a prerequisite without the fan-out.
