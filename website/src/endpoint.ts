// `export {}` makes this a module so its top-level names (init, ...)
// don't collide with the same names in other bundler-entrypoint scripts (overview.ts, model.ts)
// when type-checked together as one tsc program.
export {};

import { showLoadError } from "./components";
import { readingCaption } from "./caption";
import { MONTH_NAMES, esc, td, timeTicks } from "./components";
import { bindHover, hitRects } from "./chart_hover";
import {
  type B3ITChange,
  type CpLane,
  type FocusB3IT,
  type FocusLT,
  type LTChange,
  type LaneGeom,
  DESIGN_VW,
  LANE_H,
  MONTH_LABEL_W,
  TOP1,
  TOP2,
  VH,
  chartAxis,
  downsampleRuns,
  fmtDrift,
  fmtTV,
  laneGeoms,
  laneY,
  last,
  packMarks,
  round,
  segments,
} from "./chart_geom";

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

type Status = "stable" | "changed" | "retired";

const TRACE_LEN = 110;
const RESIZE_DEBOUNCE_MS = 150;

const fmtMon = (s: string): string => {
  const d = new Date(td(s));
  return `${MONTH_NAMES[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
};

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
  const { series, breaks } = downsampleRuns(pairs, TRACE_LEN);
  return {
    drift: series,
    breaks,
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
  const { series, breaks } = downsampleRuns(pairs, TRACE_LEN);
  return {
    tv: series,
    breaks,
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
// The geometry itself lives in chart_geom.ts, shared with the pointer readout.

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
  const axis = chartAxis(lt, b3it, vw);
  if (!axis) return "";
  const { pl: PL, pr: PR, pw: PW, vw: VW, narrow, d0, d1, fx } = axis;
  // One tick list for the gridlines and the labels both, at whatever granularity
  // this many pixels can spell out -- days on a fortnight-old endpoint, months on
  // a two-year one.
  const ticks = timeTicks(d0, d1, Math.max(1, Math.floor(PW / MONTH_LABEL_W)));
  const [ltGeom, b3Geom] = laneGeoms(lt, b3it);

  function lane(g: LaneGeom, label: string): string {
    const { series, breaks, topY, maxV, col, fill } = g;
    const yv = (v: number): number => laneY(g, v);
    const pts = series
      .map(([d, v], i) => `${i ? "L" : "M"}${fx(d).toFixed(1)} ${yv(v).toFixed(1)}`)
      .join(" ");
    // one closed area per run of observed days: the fill is what tells the reader
    // which days this endpoint was actually sampled on
    const base = (topY + LANE_H).toFixed(1);
    const area = segments(series, breaks)
      .map(
        (run) =>
          `M${fx(run[0][0]).toFixed(1)} ${base} ` +
          run.map(([d, v]) => `L${fx(d).toFixed(1)} ${yv(v).toFixed(1)}`).join(" ") +
          ` L${fx(last(run)![0]).toFixed(1)} ${base} Z`
      )
      .join(" ");
    const grid = ticks
      .map(({ t }) => {
        const x = fx(new Date(t).toISOString().slice(0, 10));
        return `<line x1="${x.toFixed(1)}" y1="${topY}" x2="${x.toFixed(1)}" y2="${topY + LANE_H}" stroke="var(--border-soft)" stroke-width="1"/>`;
      })
      .join("");
    const yticks = [0, maxV / 2, maxV]
      .map(
        (v) =>
          `<line x1="${PL}" y1="${yv(v).toFixed(1)}" x2="${VW - PR}" y2="${yv(v).toFixed(1)}" stroke="var(--border-soft)" stroke-width="0.7" opacity="0.6"/><text x="${PL - 8}" y="${(yv(v) + 3).toFixed(1)}" fill="${col}" font-size="10" font-family="var(--mono)" text-anchor="end">${v.toFixed(1)}</text>`
      )
      .join("");
    return `${grid}${yticks}<path d="${area}" fill="${fill}" stroke="none"/><path d="${pts}" fill="none" stroke="${col}" stroke-width="1.7" vector-effect="non-scaling-stroke"/>
      <text x="${PL}" y="${topY - 8}" fill="${col}" font-size="11.5" font-family="var(--mono)" font-weight="600">${label}</text>`;
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
  const ltSvg = ltGeom.series.length
    ? lane(ltGeom, ltTitle)
    : placeholder(TOP1, lt ? say("LT · drift trace not available for this endpoint yet", "LT · no drift trace yet") : say("LT · not monitored for this endpoint", "LT · not monitored"));
  const b3Svg = b3Geom.series.length
    ? lane(b3Geom, b3Title)
    : placeholder(TOP2, b3it ? say("B3IT · no reference data in this window", "B3IT · no reference data") : say("B3IT · not monitored for this endpoint", "B3IT · not monitored"));

  // A change mark's dot sits on its lane's curve at the change date, so the reader
  // can see the level the label is quoting. Only when the curve cannot answer -- an
  // empty lane, or a change recorded outside the sampled span -- does the mark fall
  // back to the lane ceiling, where every mark used to be drawn.
  const cpLanes: CpLane[] = [
    {
      lane: ltGeom,
      title: ltGeom.series.length ? ltTitle : "",
      changes: (lt?.changes ?? []).map((c) => ({ date: c.date, lab: c.sigma })),
    },
    {
      lane: b3Geom,
      title: b3Geom.series.length ? b3Title : "",
      changes: (b3it?.changes ?? []).map((c) => ({ date: c.date, lab: fmtTV(c.peakTV) })),
    },
  ];
  const cpSvg = cpLanes
    .flatMap((l) => packMarks(l, fx, VW, PL))
    .map(
      (m) => `<line x1="${m.x.toFixed(1)}" y1="${TOP1 - 4}" x2="${m.x.toFixed(1)}" y2="${TOP2 + LANE_H}" stroke="${m.col}" stroke-width="1" stroke-dasharray="3 3" opacity="0.55"/>
        <circle cx="${m.x.toFixed(1)}" cy="${m.y.toFixed(1)}" r="2.6" fill="${m.col}"/>${
          m.labelY === null
            ? ""
            : `<text x="${m.labelX.toFixed(1)}" y="${m.labelY.toFixed(1)}" fill="${m.col}" font-size="10.5" font-family="var(--mono)" font-weight="600" text-anchor="middle">${esc(m.lab)}</text>`
        }`
    )
    .join("");

  const xlabels = ticks
    .map(({ t, label }) => {
      const x = fx(new Date(t).toISOString().slice(0, 10));
      return `<text x="${x.toFixed(1)}" y="${VH - 14}" fill="var(--text-dim)" font-size="10.5" font-family="var(--mono)" text-anchor="middle">${label}</text>`;
    })
    .join("");

  return `<svg viewBox="0 0 ${VW} ${VH}" preserveAspectRatio="xMidYMid meet">
    ${ltSvg}${b3Svg}${cpSvg}${xlabels}
    <line x1="${PL}" y1="${TOP2 + LANE_H}" x2="${VW - PR}" y2="${TOP2 + LANE_H}" stroke="var(--border)" stroke-width="1"/>
    ${hitRects([ltGeom, b3Geom], PL, PW)}</svg>`;
}

function renderChart(lt: FocusLT | null, b3it: FocusB3IT | null): void {
  const chartEl = document.getElementById("mainchart");
  const tipEl = document.getElementById("charttip");
  const footEl = document.getElementById("footnote");
  if (!chartEl) return;

  // Draw and wire together: the resize redraw replaces the SVG, and a readout bound
  // to the elements it threw away is a chart that goes inert on the first rotation.
  const draw = (): boolean => {
    const svg = chartSvg(lt, b3it, chartWidth(chartEl));
    if (!svg) return false;
    chartEl.innerHTML = svg;
    if (tipEl) {
      tipEl.hidden = true;
      bindHover(chartEl, tipEl, lt, b3it, () => chartWidth(chartEl));
    }
    return true;
  };

  if (!draw()) {
    chartEl.innerHTML = `<div style="padding:2rem 1rem;color:var(--text-dim);font-size:0.85rem">No monitoring data available yet for this endpoint.</div>`;
    if (footEl) footEl.innerHTML = "";
    return;
  }
  onWidthChange(chartEl, draw);

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
  // no detected changes yet: the section (heading included) is omitted rather
  // than left as an empty table, matching the rest of the site's convention.
  if (!rows.length) {
    document.getElementById("changesSection")?.remove();
    return;
  }
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
