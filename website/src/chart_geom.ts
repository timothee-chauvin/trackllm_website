// Geometry of the endpoint page's stacked-lane chart, shared by the renderer
// (endpoint.ts) and the pointer readout (chart_hover.ts). Everything here is pure:
// no DOM, no measurement beyond the container width it is handed.
import { DAY_MS, td } from "./components";

export interface LTChange {
  date: string;
  sigma: string;
  drift: number | null; // null when the drift lane has no data yet to look up a level in
}

export interface B3ITChange {
  date: string;
  peakTV: number | null;
}

export interface FocusLT {
  drift: [string, number][];
  breaks: number[]; // see downsampleRuns: where the days the endpoint was not observed fall
  changes: LTChange[];
  // raw LT observation range from lt_scores.json's own `dates`, independent of the
  // drift trace (which may be empty/absent pre-backfill) -- see buildLT.
  firstDate: string;
  lastDate: string;
}

export interface FocusB3IT {
  tv: [string, number][];
  breaks: number[];
  changes: B3ITChange[];
  // raw tv_series observation range, independent of the (possibly downsampled) `tv` trace.
  firstDate: string;
  lastDate: string;
}

export const last = <T,>(arr: T[]): T | undefined => arr[arr.length - 1];

// ---- missing days --------------------------------------------------------------
// A day the endpoint was not sampled on is a hole in the series, and the area under
// the curve is where the reader sees it: neither the line nor the fill may cross a hole.
// The holes have to be found before the series is thinned for drawing -- afterwards
// every step looks like a gap -- so `breaks` travels with the thinned points: the
// indices into them that start a new run of consecutive observed days. timeline.py
// publishes the same field for the shared timeline, thinned at build time.

/** The runs of `series` delimited by `breaks`. */
export function segments<T>(series: T[], breaks: number[]): T[][] {
  const cuts = [0, ...breaks, series.length];
  return cuts.slice(1).map((end, i) => series.slice(cuts[i], end)).filter((s) => s.length);
}

/** One `M...L...` subpath per run, so the stroke stops at every hole. A one-point
 *  run has no length to stroke and is skipped, like its zero-width fill. */
export function strokePath<T>(runs: T[][], at: (p: T) => string): string {
  return runs
    .filter((run) => run.length > 1)
    .map((run) => run.map((p, i) => `${i ? "L" : "M"}${at(p)}`).join(" "))
    .join(" ");
}

/** One closed area subpath per run, closed on the run's own first and last x so the
 *  fill covers exactly the observed days. A one-point run gets a hairline tick of
 *  fill -- with no line and no dot, an isolated observed day would otherwise vanish. */
export function areaPath<T>(
  runs: T[][],
  x: (p: T) => number,
  y: (p: T) => number,
  base: number
): string {
  const b = base.toFixed(1);
  return runs
    .map((run) => {
      if (run.length === 1) {
        const [cx, cy] = [x(run[0]), y(run[0]).toFixed(1)];
        const [x0, x1] = [(cx - 0.75).toFixed(1), (cx + 0.75).toFixed(1)];
        return `M${x0} ${b} L${x0} ${cy} L${x1} ${cy} L${x1} ${b} Z`;
      }
      const pts = run.map((p) => `L${x(p).toFixed(1)} ${y(p).toFixed(1)}`).join(" ");
      return `M${x(run[0]).toFixed(1)} ${b} ${pts} L${x(last(run)!).toFixed(1)} ${b} Z`;
    })
    .join(" ");
}

/** `k` evenly spaced points of `run`, both ends kept so the fill under it stops
 *  exactly where the observations do. */
function pick(run: [string, number][], k: number): [string, number][] {
  if (k >= run.length) return run;
  if (k === 1) return run.slice(0, 1);
  return Array.from({ length: k }, (_, i) => run[Math.round((i * (run.length - 1)) / (k - 1))]);
}

/** Thin a daily series to about `n` points, and say where its missing days are. */
export function downsampleRuns(
  pairs: [string, number][],
  n: number
): { series: [string, number][]; breaks: number[] } {
  const runs: [string, number][][] = [];
  let prev: number | null = null;
  for (const p of pairs) {
    const t = td(p[0]);
    if (prev === null || t - prev > DAY_MS) runs.push([]);
    last(runs)!.push(p);
    prev = t;
  }
  const thinned =
    pairs.length > n
      ? runs.map((r) => pick(r, Math.max(1, Math.round((n * r.length) / pairs.length))))
      : runs;
  const series: [string, number][] = [];
  const breaks: number[] = [];
  for (const r of thinned) {
    if (series.length) breaks.push(series.length);
    series.push(...r);
  }
  return { series, breaks };
}

export function round(v: number, n: number): number {
  const f = 10 ** n;
  return Math.round(v * f) / f;
}

// The series are stored to 3 dp; a readout of a value that has not been through
// buildLT/buildB3IT is rounded the same way rather than spilling float noise.
export const fmtDrift = (v: number | null): string => (v === null ? "—" : `${round(v, 3)} nats`);
export const fmtTV = (v: number | null): string => (v === null ? "TV —" : `TV ${round(v, 3)}`);

// ---- lane layout ---------------------------------------------------------------
// The SVG is laid out in CSS pixels of the container it is about to fill: the
// viewBox width is the measured content width, so a `font-size="10"` label is 10px
// on a phone exactly as on a desktop. Drawing at a fixed 1000-unit width and
// letting `.chart svg { width: 100% }` scale it down is what made the axis and
// sigma labels render at ~2.6px in a 264px phone column.
export const DESIGN_VW = 1000; // width to draw at when nothing can be measured (no layout engine)
export const NARROW_VW = 560; // below this the side margins are trimmed and lane titles shortened
export const MIN_VW = 160; // floor: below it the margins would eat the whole plot area
export const LANE_H = 108;
export const GAP = 34;
export const TOP1 = 34;
export const TOP2 = TOP1 + LANE_H + GAP;
export const VH = TOP2 + LANE_H + 40;
// "Jul '26" in var(--mono) at font-size 10.5 measures ~41px; the rest is the gap
// that keeps two neighbouring labels from touching.
export const MONTH_LABEL_W = 52;

export interface Dims {
  vw: number;
  pl: number;
  pr: number;
  pw: number;
  narrow: boolean;
}

/** Plot margins for a given container width. At DESIGN_VW these are the values the
 *  chart has always used, so a desktop-width render is unchanged. */
export function dims(vw: number): Dims {
  const w = Math.max(MIN_VW, vw);
  const narrow = w < NARROW_VW;
  const pl = narrow ? 30 : 50;
  const pr = narrow ? 10 : 20;
  return { vw: w, pl, pr, pw: w - pl - pr, narrow };
}

export interface Axis extends Dims {
  d0: number;
  d1: number;
  fx: (s: string) => number;
}

/** The shared time axis, or null when there is nothing at all to plot. Every date
 *  either lane can put on the chart is an anchor -- including the changepoint rules,
 *  since a change can be recorded after the last sampled point (the same reason
 *  model.py folds them into its date_range). */
export function chartAxis(lt: FocusLT | null, b3it: FocusB3IT | null, vw: number): Axis | null {
  const anchors = [
    lt?.drift[0]?.[0], last(lt?.drift ?? [])?.[0],
    b3it?.tv[0]?.[0], last(b3it?.tv ?? [])?.[0],
    ...(lt?.changes ?? []).map((c) => c.date),
    ...(b3it?.changes ?? []).map((c) => c.date),
  ]
    .filter((d): d is string => !!d)
    .sort();
  if (!anchors.length) return null;

  const d = dims(vw);
  const d0 = td(anchors[0]);
  const d1 = td(last(anchors)!);
  const span = Math.max(1, d1 - d0);
  return { ...d, d0, d1, fx: (s: string): number => d.pl + ((td(s) - d0) / span) * d.pw };
}

/** One lane's plot: what it draws, where, and how a value off it reads aloud. */
export interface LaneGeom {
  key: "lt" | "b3it";
  series: [string, number][];
  breaks: number[];
  topY: number;
  maxV: number;
  col: string;
  fill: string;
  fmt: (v: number) => string;
}

// headroom above the tallest drift sample, so the trace never touches the lane top
const LT_HEADROOM = 1.08;

export function laneGeoms(lt: FocusLT | null, b3it: FocusB3IT | null): LaneGeom[] {
  const drift = lt?.drift ?? [];
  return [
    {
      key: "lt",
      series: drift,
      breaks: lt?.breaks ?? [],
      topY: TOP1,
      maxV: drift.length ? Math.max(1, ...drift.map(([, v]) => v)) * LT_HEADROOM : 1,
      col: "var(--accent)",
      fill: "var(--accent-fill)",
      fmt: (v) => fmtDrift(v),
    },
    {
      key: "b3it",
      series: b3it?.tv ?? [],
      breaks: b3it?.breaks ?? [],
      topY: TOP2,
      maxV: 1,
      col: "var(--b3it)",
      fill: "var(--b3it-fill)",
      fmt: (v) => fmtTV(v),
    },
  ];
}

export const laneY = (lane: LaneGeom, v: number): number =>
  lane.topY + LANE_H * (1 - v / lane.maxV);

// ---- change marks -------------------------------------------------------------
// measured with getBBox in Chromium and rounded up: reserving a little too much
// only ever moves a label that would have just fitted
export const CHAR_W = 6.4; // var(--mono) at font-size 10.5
export const TITLE_CHAR_W = 7.2; // ... and at the 11.5 the lane titles use
// 14, not the 12 the two font sizes suggest: at 12 a moved label's box still
// shared a pixel row with the lane title's
const ROW_H = 14;
const LABEL_DY = 10; // an anchored label clears its own dot by this much
const CEIL_DOT_DY = 4; // a mark with nothing to anchor to sits this far above the lane
const CEIL_LABEL_DY = 8; // ... and its label on the lane title's baseline
const BELOW_DY = 12; // the one place left to try once everything above is taken
const MAX_LIFT = 2; // rows a label may climb before it is put below its dot instead

export interface CpLane {
  lane: LaneGeom;
  title: string; // "" when the lane draws a placeholder instead of a trace
  changes: { date: string; lab: string }[];
}

export interface Mark {
  x: number;
  y: number;
  labelX: number;
  labelY: number | null; // null: nowhere free, so only the rule and dot are drawn
  col: string;
  lab: string;
}

interface Box {
  x0: number;
  x1: number;
  y: number;
}

const boxesCollide = (a: Box, b: Box): boolean =>
  a.x0 < b.x1 && b.x0 < a.x1 && Math.abs(a.y - b.y) < ROW_H;

/** The value the curve shows on `date`, or null when the date is outside the
 *  sampled span — the series is downsampled, so the nearest sample is the answer,
 *  but extrapolating past either end would invent a level. */
export function sampleAt(series: [string, number][], date: string): number | null {
  if (!series.length) return null;
  const t = td(date);
  if (t < td(series[0][0]) || t > td(last(series)![0])) return null;
  let best = series[0];
  for (const p of series) {
    if (Math.abs(td(p[0]) - t) < Math.abs(td(best[0]) - t)) best = p;
  }
  return best[1];
}

/** Place one lane's change marks: dot on the curve, label above it, moved out of
 *  the way of the lane title and of labels already placed. A label with nowhere
 *  free is dropped — its dashed rule and dot still mark the day, and the changes
 *  table below the chart lists every change with its magnitude. */
export function packMarks(
  cp: CpLane,
  fx: (s: string) => number,
  vw: number,
  pl: number
): Mark[] {
  const { lane } = cp;
  const ceiling = lane.topY - CEIL_LABEL_DY - ROW_H; // one clear row above the title
  const floor = lane.topY + LANE_H;
  const placed: Box[] = cp.title
    ? [{ x0: pl, x1: pl + cp.title.length * TITLE_CHAR_W, y: lane.topY - CEIL_LABEL_DY }]
    : [];
  return cp.changes
    .map((c) => ({ c, x: fx(c.date) }))
    .sort((a, b) => a.x - b.x)
    .map(({ c, x }) => {
      const v = sampleAt(lane.series, c.date);
      const y = v === null ? lane.topY - CEIL_DOT_DY : laneY(lane, v);
      const half = (c.lab.length * CHAR_W) / 2;
      const labelX = Math.min(Math.max(x, half), vw - half);
      const box = (ly: number): Box => ({ x0: labelX - half, x1: labelX + half, y: ly });
      const start = v === null ? lane.topY - CEIL_LABEL_DY : y - LABEL_DY;
      const rows = Array.from({ length: MAX_LIFT + 1 }, (_, k) => start - k * ROW_H)
        .filter((ly) => ly >= ceiling)
        .concat(y + BELOW_DY);
      const labelY =
        rows.find((ly) => ly <= floor && !placed.some((p) => boxesCollide(box(ly), p))) ?? null;
      if (labelY !== null) placed.push(box(labelY));
      return { x, y, labelX, labelY, col: lane.col, lab: c.lab };
    });
}
