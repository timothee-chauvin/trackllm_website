// `export {}` makes this a module so its top-level names (init, ...)
// don't collide with the same names in other bundler-entrypoint scripts (overview.ts, model.ts)
// when type-checked together as one tsc program.
export {};

import { showLoadError } from "./components";
import { readingCaption } from "./caption";
import { MONTH_NAMES, esc, monthTicks, td } from "./components";

interface ManifestData {
  slug: string;
  // Both canonical, stamped at build time (generate_site/manifest.py): the status
  // verdict is aged against the build's clock, never the reader's, and the changes
  // are the merged list's own -- not the changepoints lt_scores.json recomputes,
  // which double-detect some changes on adjacent days.
  state: Status | null;
  changes: { lt: LTChange[]; b3it: B3ITChange[] };
}

interface LTScoresData {
  n_per_test: number;
  dates: string[];
  // absent entirely on endpoints whose lt_scores.json predates the Task 1 drift backfill
  drift_dates?: string[];
  drift?: number[];
}

interface B3ITData {
  status: string;
  retired_reason: string | null;
  n_bis: number;
  unstable: boolean;
  tv_series: { dates: string[]; values: number[] };
}

interface LTChange {
  date: string;
  sigma: string;
  drift: number | null; // null when the drift lane has no data yet to look up a level in
}

interface B3ITChange {
  date: string;
  peakTV: number | null;
}

interface FocusLT {
  drift: [string, number][];
  changes: LTChange[];
  // raw LT observation range from lt_scores.json's own `dates`, independent of the
  // drift trace (which may be empty/absent pre-backfill) -- see buildLT.
  firstDate: string;
  lastDate: string;
}

interface FocusB3IT {
  tv: [string, number][];
  changes: B3ITChange[];
  // raw tv_series observation range, independent of the (possibly downsampled) `tv` trace.
  firstDate: string;
  lastDate: string;
}

type Status = "stable" | "changed" | "retired";

const TRACE_LEN = 110;

const fmtMon = (s: string): string => {
  const d = new Date(td(s));
  return `${MONTH_NAMES[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
};
const round = (v: number, n: number): number => {
  const f = 10 ** n;
  return Math.round(v * f) / f;
};
const last = <T,>(arr: T[]): T | undefined => arr[arr.length - 1];

function downsamplePairs(pairs: [string, number][], n: number): [string, number][] {
  if (pairs.length <= n) return pairs;
  return Array.from({ length: n }, (_, i) => pairs[Math.floor((i * pairs.length) / n)]);
}

const fmtDrift = (v: number | null): string => (v === null ? "—" : `${v} nats`);
const fmtTV = (v: number | null): string => (v === null ? "TV —" : `TV ${v}`);

async function fetchJSON<T>(url: string): Promise<T | null> {
  const res = await fetch(url);
  // 404 is a real state -- the file is absent because that method never ran for
  // this endpoint. Any other failure must throw rather than masquerade as
  // "not monitored".
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

// Read drift straight from lt_scores.json's own drift/drift_dates -- this is
// already computed by the pipeline (Task 1); never recompute it from raw logprobs here.
// The changes come from the manifest, not from this file: they are a different part of
// the LT pipeline than the drift trace, so a still-empty drift/drift_dates backfill
// (see brief) leaves them intact, only their per-change drift *level* unknown.
export function buildLT(scores: LTScoresData | null, changes: LTChange[]): FocusLT | null {
  if (!scores || !scores.dates.length) return null;
  const driftDates = scores.drift_dates ?? [];
  const drift = scores.drift ?? [];
  const pairs: [string, number][] = driftDates.map((d, i) => [d.slice(0, 10), round(drift[i], 3)]);
  return {
    drift: downsamplePairs(pairs, TRACE_LEN),
    changes,
    firstDate: scores.dates[0].slice(0, 10),
    lastDate: last(scores.dates)!.slice(0, 10),
  };
}

export function buildB3IT(data: B3ITData | null, changes: B3ITChange[]): FocusB3IT | null {
  if (!data) return null;
  const pairs: [string, number][] = data.tv_series.dates.map((d, i) => [
    d.slice(0, 10),
    round(data.tv_series.values[i], 3),
  ]);
  return {
    tv: downsamplePairs(pairs, TRACE_LEN),
    changes,
    firstDate: data.tv_series.dates.length ? data.tv_series.dates[0].slice(0, 10) : "",
    lastDate: data.tv_series.dates.length ? last(data.tv_series.dates)!.slice(0, 10) : "",
  };
}

/** The observed span is the one thing here the series alone can answer; the verdict
 *  and the change counts are the build's (see ManifestData). */
function renderStatusCard(lt: FocusLT | null, b3it: FocusB3IT | null, state: Status | null): void {
  const el = document.getElementById("statuscard");
  if (!el) return;

  const lastDates = [lt?.lastDate, b3it?.lastDate].filter((d): d is string => !!d);
  const firstDates = [lt?.firstDate, b3it?.firstDate].filter((d): d is string => !!d);
  const lastObserved = lastDates.length ? last(lastDates.sort())! : null;
  const first = firstDates.length ? firstDates.sort()[0] : null;

  const LABEL: Record<Status, string> = { stable: "Stable", changed: "Changed", retired: "Retired" };
  const nLT = lt?.changes.length ?? 0;
  const nB3 = b3it?.changes.length ?? 0;
  const monitored = first && lastObserved ? `${fmtMon(first)} – ${fmtMon(lastObserved)}` : "—";

  el.innerHTML = `
    ${state ? `<div><div class="k">Status</div><div class="v"><span class="pill ${esc(state)}"><span class="led"></span>${LABEL[state]}</span></div></div>` : ""}
    <div><div class="k">Monitored</div><div class="v">${monitored}</div></div>
    <div><div class="k">Changes</div><div class="v">${nLT + nB3} <small>(${nLT} LT · ${nB3} B3IT)</small></div></div>`;
}

// ---- stacked-lane drift chart: LT drift (nats) and B3IT total variation (0-1)
// share one time axis so a real change reads as an aligned step in both lanes.
//
// The SVG is laid out in CSS pixels of the container it is about to fill: the
// viewBox width is the measured content width, so a `font-size="10"` label is 10px
// on a phone exactly as on a desktop. Drawing at a fixed 1000-unit width and
// letting `.chart svg { width: 100% }` scale it down is what made the axis and
// sigma labels render at ~2.6px in a 264px phone column.
const DESIGN_VW = 1000; // width to draw at when nothing can be measured (no layout engine)
const NARROW_VW = 560; // below this the side margins are trimmed and month labels thinned
const MIN_VW = 160; // floor: below it the margins would eat the whole plot area
const LANE_H = 108, GAP = 34, TOP1 = 34;
const TOP2 = TOP1 + LANE_H + GAP;
const VH = TOP2 + LANE_H + 40;
// "Jul 26" in var(--mono) at font-size 10.5 measures ~41px; the rest is the gap
// that keeps two neighbouring labels from touching.
const MONTH_LABEL_W = 52;
const RESIZE_DEBOUNCE_MS = 150;

interface Dims {
  vw: number;
  pl: number;
  pr: number;
  pw: number;
  narrow: boolean;
}

/** Plot margins for a given container width. At DESIGN_VW these are the values the
 *  chart has always used, so a desktop-width render is unchanged. */
function dims(vw: number): Dims {
  const w = Math.max(MIN_VW, vw);
  const narrow = w < NARROW_VW;
  const pl = narrow ? 30 : 50;
  const pr = narrow ? 10 : 20;
  return { vw: w, pl, pr, pw: w - pl - pr, narrow };
}

/** Content width of the chart's own box, in CSS pixels. */
function chartWidth(el: HTMLElement): number {
  const cs = getComputedStyle(el);
  const w = Math.round(
    el.clientWidth - parseFloat(cs.paddingLeft || "0") - parseFloat(cs.paddingRight || "0")
  );
  return w > 0 ? w : DESIGN_VW;
}

/** Redraw when the container's width actually changed. A phone fires `resize` for
 *  every scroll-driven browser-chrome collapse, where only the height moves. */
function onWidthChange(el: HTMLElement, draw: () => void): void {
  let width = chartWidth(el);
  let timer: ReturnType<typeof setTimeout>;
  window.addEventListener("resize", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      const w = chartWidth(el);
      if (w === width) return;
      width = w;
      draw();
    }, RESIZE_DEBOUNCE_MS);
  });
}

/** The whole chart as one SVG, a pure function of the data and the width to fill.
 *  Returns "" when there is nothing to plot. */
export function chartSvg(lt: FocusLT | null, b3it: FocusB3IT | null, vw: number): string {
  const anchors = [
    lt?.drift[0]?.[0], last(lt?.drift ?? [])?.[0],
    b3it?.tv[0]?.[0], last(b3it?.tv ?? [])?.[0],
    // the changepoint rules are on this axis too, and a change can fall outside the
    // observed span (one recorded after the last sampled point) -- same reason
    // model.py folds them into its date_range
    ...(lt?.changes ?? []).map((c) => c.date),
    ...(b3it?.changes ?? []).map((c) => c.date),
  ].filter((d): d is string => !!d);
  if (!anchors.length) return "";

  const { pl: PL, pr: PR, pw: PW, vw: VW, narrow } = dims(vw);
  const d0 = td(anchors.sort()[0]);
  const d1 = td(last(anchors.sort())!);
  const span = Math.max(1, d1 - d0);
  const fx = (s: string): number => PL + ((td(s) - d0) / span) * PW;

  function lane(
    series: [string, number][],
    topY: number,
    maxV: number,
    color: string,
    fill: string,
    label: string,
    axisFmt: (v: number) => string
  ): string {
    const yv = (v: number): number => topY + LANE_H * (1 - v / maxV);
    const pts = series
      .map(([d, v], i) => `${i ? "L" : "M"}${fx(d).toFixed(1)} ${yv(v).toFixed(1)}`)
      .join(" ");
    const area =
      `M${fx(series[0][0]).toFixed(1)} ${(topY + LANE_H).toFixed(1)} ` +
      series.map(([d, v]) => `L${fx(d).toFixed(1)} ${yv(v).toFixed(1)}`).join(" ") +
      ` L${fx(last(series)![0]).toFixed(1)} ${(topY + LANE_H).toFixed(1)} Z`;
    const grid = monthTicks(d0, d1)
      .map((d) => {
        const x = fx(d.toISOString().slice(0, 10));
        return `<line x1="${x.toFixed(1)}" y1="${topY}" x2="${x.toFixed(1)}" y2="${topY + LANE_H}" stroke="var(--border-soft)" stroke-width="1"/>`;
      })
      .join("");
    const ticks = [0, maxV / 2, maxV]
      .map(
        (v) =>
          `<line x1="${PL}" y1="${yv(v).toFixed(1)}" x2="${VW - PR}" y2="${yv(v).toFixed(1)}" stroke="var(--border-soft)" stroke-width="0.7" opacity="0.6"/><text x="${PL - 8}" y="${(yv(v) + 3).toFixed(1)}" fill="${color}" font-size="10" font-family="var(--mono)" text-anchor="end">${axisFmt(v)}</text>`
      )
      .join("");
    return `${grid}${ticks}<path d="${area}" fill="${fill}" stroke="none"/><path d="${pts}" fill="none" stroke="${color}" stroke-width="1.7" vector-effect="non-scaling-stroke"/>
      <text x="${PL}" y="${topY - 8}" fill="${color}" font-size="11.5" font-family="var(--mono)" font-weight="600">${label}</text>`;
  }

  function placeholder(topY: number, text: string): string {
    return `<text x="${PL}" y="${topY + LANE_H / 2}" fill="var(--text-dim)" font-size="11" font-family="var(--mono)">${text}</text>`;
  }

  // SVG text does not wrap, so every string here has to fit the plot it is drawn
  // in: at ~30 px of margin and ~7 px per character, a phone column has room for
  // roughly 30 of them. The lane's meaning is spelled out in .sec-desc above.
  const say = (full: string, short: string): string => (narrow ? short : full);
  const ltTitle = say("LT · drift from baseline (nats)", "LT · drift (nats)");
  const b3Title = say("B3IT · total variation from baseline (0–1)", "B3IT · TV (0–1)");
  const ltMax = lt && lt.drift.length ? Math.max(1, ...lt.drift.map(([, v]) => v)) * 1.08 : 1;
  const ltSvg =
    lt && lt.drift.length
      ? lane(lt.drift, TOP1, ltMax, "var(--accent)", "var(--accent-fill)", ltTitle, (v) => v.toFixed(1))
      : placeholder(TOP1, lt ? say("LT · drift trace not available for this endpoint yet", "LT · no drift trace yet") : say("LT · not monitored for this endpoint", "LT · not monitored"));
  const b3Svg =
    b3it && b3it.tv.length
      ? lane(b3it.tv, TOP2, 1, "var(--b3it)", "var(--b3it-fill)", b3Title, (v) => v.toFixed(1))
      : placeholder(TOP2, b3it ? say("B3IT · no reference data in this window", "B3IT · no reference data") : say("B3IT · not monitored for this endpoint", "B3IT · not monitored"));

  const cps = [
    ...(lt?.changes ?? []).map((c) => ({ x: fx(c.date), col: "var(--accent)", lab: c.sigma, baseY: TOP1 - 8 })),
    ...(b3it?.changes ?? []).map((c) => ({ x: fx(c.date), col: "var(--b3it)", lab: fmtTV(c.peakTV), baseY: TOP2 - 8 })),
  ];
  // A changepoint label is centred on its dot and shares a baseline with the lane
  // title, so on a plot this narrow they overprint each other and the title. Three
  // corrections, all no-ops while everything still fits: pull a label at either
  // end of the span back inside the SVG (which clips), lift one that would land
  // on an already-occupied stretch of the baseline into the row above, and drop the
  // one that finds every row taken -- its dashed rule and dot still mark the day,
  // and the changes table below the chart lists every change with its magnitude.
  // measured with getBBox in Chromium and rounded up: reserving a little too much
  // only ever lifts a label that would have just fitted
  const CHAR_W = 6.4; // var(--mono) at font-size 10.5
  const TITLE_CHAR_W = 7.2; // ... and at the 11.5 the lane titles use
  // 14, not the 12 the two font sizes suggest: at 12 a lifted label's box still
  // shared a pixel row with the lane title's
  const ROW_H = 14;
  // one row up is all the clear space there is above either baseline: row 2 would
  // leave the SVG over the LT lane and cut into the drift plot over the B3IT one
  const MAX_ROW = 1;
  // per baseline, the right edge already occupied in each row -- row 0 starts at
  // the end of the lane title
  const rowEnd = new Map<number, number[]>([
    [TOP1 - 8, [PL + (lt?.drift.length ? ltTitle.length * TITLE_CHAR_W : 0)]],
    [TOP2 - 8, [PL + (b3it?.tv.length ? b3Title.length * TITLE_CHAR_W : 0)]],
  ]);
  const cpSvg = cps
    .sort((a, b) => a.x - b.x)
    .map((c) => {
      const half = (c.lab.length * CHAR_W) / 2;
      const labX = Math.min(Math.max(c.x, half), VW - half);
      const ends = rowEnd.get(c.baseY)!;
      let row = 0;
      while (row <= MAX_ROW && labX - half < (ends[row] ?? 0) + 4) row++;
      if (row <= MAX_ROW) ends[row] = labX + half;
      const label =
        row <= MAX_ROW
          ? `<text x="${labX.toFixed(1)}" y="${c.baseY - row * ROW_H}" fill="${c.col}" font-size="10.5" font-family="var(--mono)" font-weight="600" text-anchor="middle">${esc(c.lab)}</text>`
          : "";
      return `<line x1="${c.x.toFixed(1)}" y1="${TOP1 - 4}" x2="${c.x.toFixed(1)}" y2="${TOP2 + LANE_H}" stroke="${c.col}" stroke-width="1" stroke-dasharray="3 3" opacity="0.55"/>
        <circle cx="${c.x.toFixed(1)}" cy="${(c.baseY + 4).toFixed(1)}" r="2.6" fill="${c.col}"/>${label}`;
    })
    .join("");

  // Every month keeps its gridline; only the labels thin out, and only once they
  // would no longer fit side by side -- which on a phone is from the second one.
  const ticks = monthTicks(d0, d1);
  const step = Math.max(1, Math.ceil((ticks.length * MONTH_LABEL_W) / PW));
  const xlabels = ticks
    .filter((_, i) => i % step === 0)
    .map((d) => {
      const x = fx(d.toISOString().slice(0, 10));
      return `<text x="${x.toFixed(1)}" y="${VH - 14}" fill="var(--text-dim)" font-size="10.5" font-family="var(--mono)" text-anchor="middle">${MONTH_NAMES[d.getUTCMonth()]} ${String(d.getUTCFullYear()).slice(2)}</text>`;
    })
    .join("");

  return `<svg viewBox="0 0 ${VW} ${VH}" preserveAspectRatio="xMidYMid meet">
    ${ltSvg}${b3Svg}${cpSvg}${xlabels}
    <line x1="${PL}" y1="${TOP2 + LANE_H}" x2="${VW - PR}" y2="${TOP2 + LANE_H}" stroke="var(--border)" stroke-width="1"/></svg>`;
}

function renderChart(lt: FocusLT | null, b3it: FocusB3IT | null): void {
  const chartEl = document.getElementById("mainchart");
  const footEl = document.getElementById("footnote");
  if (!chartEl) return;

  const svg = chartSvg(lt, b3it, chartWidth(chartEl));
  if (!svg) {
    chartEl.innerHTML = `<div style="padding:2rem 1rem;color:var(--text-dim);font-size:0.85rem">No monitoring data available yet for this endpoint.</div>`;
    if (footEl) footEl.innerHTML = "";
    return;
  }
  chartEl.innerHTML = svg;
  onWidthChange(chartEl, () => {
    chartEl.innerHTML = chartSvg(lt, b3it, chartWidth(chartEl));
  });

  if (footEl) {
    let note = readingCaption(!!lt, !!b3it);
    if (lt?.drift.length && b3it?.tv.length && b3it.tv[0][0] > lt.drift[0][0]) {
      note += ` B3IT only has reference data from ${b3it.tv[0][0]} onward, so its lane starts there.`;
    }
    footEl.innerHTML = note;
  }
}

function renderChangesTable(lt: FocusLT | null, b3it: FocusB3IT | null): void {
  const el = document.getElementById("changerows");
  if (!el) return;
  const rows: { date: string; method: "lt" | "b3it"; mag: string; conf: string }[] = [];
  (lt?.changes ?? []).forEach((c) =>
    rows.push({ date: c.date, method: "lt", mag: fmtDrift(c.drift), conf: c.sigma })
  );
  (b3it?.changes ?? []).forEach((c) =>
    rows.push({ date: c.date, method: "b3it", mag: fmtTV(c.peakTV), conf: "—" })
  );
  rows.sort((a, b) => td(b.date) - td(a.date));
  el.innerHTML = rows
    .map(
      (r) => `<tr>
    <td class="date">${esc(r.date)}</td>
    <td><span class="badge ${r.method}">${r.method}</span></td>
    <td class="r mag" style="color:${r.method === "lt" ? "var(--accent)" : "var(--b3it)"}">${r.mag}</td>
    <td class="r num" style="color:var(--text-muted)">${esc(r.conf)}</td></tr>`
    )
    .join("");
}

export async function init(): Promise<void> {
  const manifestEl = document.getElementById("manifest");
  if (!manifestEl) {
    console.error("Manifest element not found");
    return;
  }
  const manifest: ManifestData = JSON.parse(manifestEl.textContent || "{}");

  let scores: LTScoresData | null, b3itData: B3ITData | null;
  try {
    [scores, b3itData] = await Promise.all([
      fetchJSON<LTScoresData>(`../data/lt/${manifest.slug}/lt_scores.json`),
      fetchJSON<B3ITData>(`../data/b3it/${manifest.slug}/b3it.json`),
    ]);
  } catch (err) {
    showLoadError("mainchart", "this endpoint's monitoring data");
    throw err;
  }

  const lt = buildLT(scores, manifest.changes.lt);
  const b3it = buildB3IT(b3itData, manifest.changes.b3it);

  renderStatusCard(lt, b3it, manifest.state);
  renderChart(lt, b3it);
  renderChangesTable(lt, b3it);
}

init();
