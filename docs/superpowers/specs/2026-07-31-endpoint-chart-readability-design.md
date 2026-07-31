# Endpoint chart readability: anchored change marks, dated axis, pointer readout

Reported on `https://www.trackllm.net/endpoints/z-ai2fglm-5.223cloudflare.html`:

1. The `TV 0.535` label floats above the curve instead of sitting on it.
2. Hovering the plot reveals nothing — no dates, no y-values.
3. The x axis is blank.

## Diagnosis

**Change label.** `endpoint.ts` draws every changepoint label on a fixed baseline at
the top of its lane (`TOP1 - 8` / `TOP2 - 8`) and its dot 4px below that. The B3IT
lane is scaled 0→1, so a change of 0.535 belongs at mid-lane; the mark is at the
ceiling. The number itself is correct: `tv_series` reaches 0.5355 on 2026-07-25/26
and the manifest records `peakTV: 0.535`.

**Blank axis.** `monthTicks(d0, d1)` (`components.ts`) emits first-of-month ticks and
keeps the month start preceding `d0` only when it is within 15 days. This endpoint
spans 2026-07-17 → 2026-07-29; Jul 1 is 16 days before the start, so the function
returns an empty list. No gridlines, no labels — and every endpoint whose observed
span is short enough to fall inside one month hits the same hole.

**No readout.** The chart has never had a pointer interaction. The SVG carries no
`<title>` and no hit layer, so neither hover nor touch reveals a value.

## Design

### 1. Change marks anchored to the curve

A change mark's dot moves onto its lane's curve at the change date; its label sits
`LABEL_DY` above the dot.

- The anchor value is the sampled point nearest the change date. If the change date
  falls outside the series' span (a change recorded after the last sample), or the
  lane has no series at all, the mark keeps today's lane-top baseline.
- The label keeps the manifest's number (`peakTV`), which is canonical — see the
  `ManifestData` comment. The dot uses the series value. On the reported endpoint the
  two differ only by rounding (0.535 vs 0.5355).
- Collision handling replaces the current per-baseline `rowEnd` map, which assumes
  all labels in a lane share one y. New rule, per lane: a label whose x-interval
  overlaps an already-placed label and whose y is within `ROW_H` of it lifts by
  `ROW_H`; if lifting would clear the lane top or hit the lane title, it is placed
  below the dot instead; if that is blocked too, it is dropped — as today, its dashed
  rule and dot still mark the day, and the changes table below lists every change
  with its magnitude.

### 2. Adaptive x axis

New `timeTicks(d0, d1, maxLabels)` in `components.ts`, returning
`{ t: number; label: string }[]`.

- Step ladder, first step whose tick count is ≤ `maxLabels`:
  1d, 2d, 3d, 7d, 14d, 1mo, 2mo, 3mo, 6mo, 12mo.
- `maxLabels = floor(PW / MONTH_LABEL_W)`, the same width budget the current
  thinning uses. The last rung always fits, so the list is never empty.
- Day and week steps are aligned to whole UTC days from `d0` and labelled `Jul 18`.
  Month steps land on the first of the month and are labelled `Jul '26` — the
  apostrophe is what stops a day label from being read as a year.
- Gridlines and labels both come from this one list, so `chartSvg`'s separate
  `step`-thinning of month labels goes away and the axis can no longer be empty.
- `model.ts` keeps calling `monthTicks`; only the endpoint chart switches.

### 3. Pointer readout, per lane, hover and touch

Each lane with data gets its own readout showing the date and that lane's value.
Lanes are read independently — there is no crosshair spanning both.

- `.chartwrap` becomes a positioned box holding a `chart-tip` div, the same idiom as
  the hero card's `hero-tip` (`overview.ts`, `style.css`).
- Each lane with data gets a transparent hit `<rect>` covering its plot area, drawn
  after the traces so it receives the pointer events.
- Pure `nearestPoint(series, xPx, fx)` returns the index of the sampled day nearest
  the pointer — testable with no layout engine.
- A mouse or pen `pointermove` over a lane, or a tap on a touch device, draws a
  marker dot on that lane's curve at the nearest sample and places the tip beside the
  pointer reading `Jul 26 · TV 0.536` or `Jul 26 · 0.81 nats`. The tip is clamped
  inside the chart box.
- `pointerleave` hides it. On touch it stays pinned until the next tap; a tap outside
  any lane dismisses it.
- The changes table below the chart remains the keyboard and screen-reader path;
  the hit rects are `aria-hidden`, adding no tab stops.

### 4. Module split

`website/src/endpoint.ts` is 421 lines carrying data shaping, chart geometry, tick
maths, label packing, and DOM glue. The work above adds to three of those at once, so
it is split first, along the seams the work already follows:

| module | contents |
| --- | --- |
| `endpoint/data.ts` | `LTScoresData`/`B3ITData`/`Focus*` types, `buildLT`, `buildB3IT`, `fetchJSON`, `downsamplePairs` |
| `endpoint/ticks.ts` | `timeTicks` and its formatters |
| `endpoint/marks.ts` | change-mark anchoring and label packing |
| `endpoint/hover.ts` | hit rects, `nearestPoint`, tip positioning |
| `endpoint/chart.ts` | `chartSvg`, `lane`, `dims`, the geometry constants |
| `endpoint.ts` | `init`, `renderStatusCard`, `renderChart`, `renderChangesTable` |

`chart.ts` is the only integration point; the other three are pure or self-contained.

The split is a behaviour-preserving move — existing tests (`mobile.test.ts` imports
`chartSvg` from `../src/endpoint`) must stay green, so `endpoint.ts` re-exports
`chartSvg`.

### 5. Redraw

`renderChart` currently rewrites `chartEl.innerHTML` on resize, which would discard
any hover wiring. Drawing and binding merge into one function that the resize handler
calls whole.

## Testing

Bun + happy-dom, in `website/test/`. Tests first.

- `timeTicks`: a 12-day span yields day labels and at least three of them; a 3-month
  span yields week or month steps; a 2-year span yields month steps and never exceeds
  `maxLabels`; a single-day span yields at least one tick.
- `chartSvg` on a 12-day B3IT-only series emits `Jul`-plus-day axis labels —
  regression for the blank axis.
- `chartSvg` change mark: the label's `y` is within a few px of the curve's y at
  0.5355, not at `TOP2 - 8` — regression for the reported bug.
- `nearestPoint`: exact hit, midpoint between two samples, and clamping past either
  end.
- Hover glue: a dispatched `pointermove` on a lane's hit rect unhides the tip with
  the expected date and value text.

happy-dom cannot catch overlap or layout, so a headless-chromium render of the
reported endpoint page at desktop width and at 375px is part of the acceptance check.

## Out of scope

- The `model.ts`, `provider.ts` and hero charts.
- The duplicate `2026-07-29T21:48:33+00:00` entry in this endpoint's `tv_series`
  (two values, 0.4825 and 0.1105, for one timestamp). That is a pipeline data
  question, tracked separately.
