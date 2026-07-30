// `export {}` makes this a module so its top-level names (init, ...)
// don't collide with the same names in other bundler-entrypoint scripts (overview.ts, model.ts)
// when type-checked together as one tsc program.
export {};

import { showLoadError } from "./components";
import { readingCaption } from "./caption";
import { DAY_MS, MONTH_NAMES, monthTicks, td } from "./components";

interface ManifestData {
  slug: string;
}

interface LTScoresData {
  n_per_test: number;
  dates: string[];
  scores: number[];
  sigmas: (number | null)[];
  changes: { index: number; sigma: number | null }[];
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
  changes: { date: string; kind: string }[];
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

// downsampling/windowing constants mirror generate_site/model.py and peaks.py so
// the endpoint page and the model page read the same drift level for the same
// changepoint.
const TRACE_LEN = 110;
const LT_PEAK_WINDOW = 20;
const B3IT_PEAK_WINDOW = 8;
const RETIRED_GAP_DAYS = 14;
const RECENT_CHANGE_DAYS = 60;
const SIGMA_INF_THRESHOLD = 1e4;

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

// first point on/after `day`, max over the next `window` points -- same rule as
// generate_site/peaks.py's peak_from, so drift levels agree across pages.
function peakFrom(day: string, pairs: [string, number][], window: number): number | null {
  const onOrAfter = pairs.filter(([d]) => d >= day).slice(0, window);
  return onOrAfter.length ? Math.max(...onOrAfter.map(([, v]) => v)) : null;
}

function sigmaDisplay(sigma: number | null): string {
  if (sigma === null || !Number.isFinite(sigma) || Math.abs(sigma) >= SIGMA_INF_THRESHOLD) {
    return "∞σ";
  }
  return `${Math.round(sigma)}σ`;
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
// `changes` (sigma-based changepoints) come from a different part of the LT pipeline than
// the drift trace, so a still-empty drift/drift_dates backfill (see brief) must not hide them --
// only the per-change drift *level* lookup is unavailable until the backfill lands.
function buildLT(scores: LTScoresData | null): FocusLT | null {
  if (!scores || !scores.dates.length) return null;
  const driftDates = scores.drift_dates ?? [];
  const drift = scores.drift ?? [];
  const pairs: [string, number][] = driftDates.map((d, i) => [d.slice(0, 10), round(drift[i], 3)]);
  const changes: LTChange[] = scores.changes.map((c) => {
    const day = scores.dates[c.index].slice(0, 10);
    const level = peakFrom(day, pairs, LT_PEAK_WINDOW);
    return { date: day, sigma: sigmaDisplay(c.sigma), drift: level === null ? null : round(level, 3) };
  });
  return {
    drift: downsamplePairs(pairs, TRACE_LEN),
    changes,
    firstDate: scores.dates[0].slice(0, 10),
    lastDate: last(scores.dates)!.slice(0, 10),
  };
}

function buildB3IT(data: B3ITData | null): FocusB3IT | null {
  if (!data) return null;
  const pairs: [string, number][] = data.tv_series.dates.map((d, i) => [
    d.slice(0, 10),
    round(data.tv_series.values[i], 3),
  ]);
  const changes: B3ITChange[] = data.changes.map((c) => {
    const day = c.date.slice(0, 10);
    const peak = peakFrom(day, pairs, B3IT_PEAK_WINDOW);
    return { date: day, peakTV: peak === null ? null : round(peak, 3) };
  });
  return {
    tv: downsamplePairs(pairs, TRACE_LEN),
    changes,
    firstDate: data.tv_series.dates.length ? data.tv_series.dates[0].slice(0, 10) : "",
    lastDate: data.tv_series.dates.length ? last(data.tv_series.dates)!.slice(0, 10) : "",
  };
}

function computeStatus(lastObserved: string | null, lastChange: string | null): Status {
  if (!lastObserved) return "stable";
  const gapDays = (Date.now() - td(lastObserved)) / DAY_MS;
  if (gapDays > RETIRED_GAP_DAYS) return "retired";
  if (lastChange && (Date.now() - td(lastChange)) / DAY_MS <= RECENT_CHANGE_DAYS) {
    return "changed";
  }
  return "stable";
}

function renderStatusCard(lt: FocusLT | null, b3it: FocusB3IT | null): void {
  const el = document.getElementById("statuscard");
  if (!el) return;

  const lastDates = [lt?.lastDate, b3it?.lastDate].filter((d): d is string => !!d);
  const firstDates = [lt?.firstDate, b3it?.firstDate].filter((d): d is string => !!d);
  const lastObserved = lastDates.length ? last(lastDates.sort())! : null;
  const first = firstDates.length ? firstDates.sort()[0] : null;

  const changeDates = [...(lt?.changes ?? []), ...(b3it?.changes ?? [])].map((c) => c.date);
  const lastChange = changeDates.length ? last(changeDates.sort())! : null;

  const status = computeStatus(lastObserved, lastChange);
  const LABEL: Record<Status, string> = { stable: "Stable", changed: "Changed", retired: "Retired" };
  const nLT = lt?.changes.length ?? 0;
  const nB3 = b3it?.changes.length ?? 0;
  const monitored = first && lastObserved ? `${fmtMon(first)} – ${fmtMon(lastObserved)}` : "—";

  el.innerHTML = `
    <div><div class="k">Status</div><div class="v"><span class="pill ${status}"><span class="led"></span>${LABEL[status]}</span></div></div>
    <div><div class="k">Monitored</div><div class="v">${monitored}</div></div>
    <div><div class="k">Changes</div><div class="v">${nLT + nB3} <small>(${nLT} LT · ${nB3} B3IT)</small></div></div>`;
}

// ---- stacked-lane drift chart: LT drift (nats) and B3IT total variation (0-1)
// share one time axis so a real change reads as an aligned step in both lanes.
const VW = 1000, PL = 50, PR = 20;
const LANE_H = 108, GAP = 34, TOP1 = 34;
const TOP2 = TOP1 + LANE_H + GAP;
const VH = TOP2 + LANE_H + 40;
const PW = VW - PL - PR;

function renderChart(lt: FocusLT | null, b3it: FocusB3IT | null): void {
  const chartEl = document.getElementById("mainchart");
  const footEl = document.getElementById("footnote");
  if (!chartEl) return;

  const anchors = [
    lt?.drift[0]?.[0], last(lt?.drift ?? [])?.[0],
    b3it?.tv[0]?.[0], last(b3it?.tv ?? [])?.[0],
  ].filter((d): d is string => !!d);
  if (!anchors.length) {
    chartEl.innerHTML = `<div style="padding:2rem 1rem;color:var(--text-dim);font-size:0.85rem">No monitoring data available yet for this endpoint.</div>`;
    if (footEl) footEl.innerHTML = "";
    return;
  }
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

  const ltMax = lt && lt.drift.length ? Math.max(1, ...lt.drift.map(([, v]) => v)) * 1.08 : 1;
  const ltSvg =
    lt && lt.drift.length
      ? lane(lt.drift, TOP1, ltMax, "var(--accent)", "var(--accent-fill)", "LT · drift from baseline (nats)", (v) => v.toFixed(1))
      : placeholder(TOP1, lt ? "LT · drift trace not available for this endpoint yet" : "LT · not monitored for this endpoint");
  const b3Svg =
    b3it && b3it.tv.length
      ? lane(b3it.tv, TOP2, 1, "var(--b3it)", "var(--b3it-fill)", "B3IT · total variation from baseline (0–1)", (v) => v.toFixed(1))
      : placeholder(TOP2, b3it ? "B3IT · no reference data in this window" : "B3IT · not monitored for this endpoint");

  const cps: { x: number; col: string; lab: string; lane: "lt" | "b3" }[] = [];
  (lt?.changes ?? []).forEach((c) => cps.push({ x: fx(c.date), col: "var(--accent)", lab: c.sigma, lane: "lt" }));
  (b3it?.changes ?? []).forEach((c) => cps.push({ x: fx(c.date), col: "var(--b3it)", lab: fmtTV(c.peakTV), lane: "b3" }));
  const cpSvg = cps
    .map((c) => {
      const labY = c.lane === "lt" ? TOP1 - 8 : TOP2 - 8;
      return `<line x1="${c.x.toFixed(1)}" y1="${TOP1 - 4}" x2="${c.x.toFixed(1)}" y2="${TOP2 + LANE_H}" stroke="${c.col}" stroke-width="1" stroke-dasharray="3 3" opacity="0.55"/>
        <circle cx="${c.x.toFixed(1)}" cy="${(labY + 4).toFixed(1)}" r="2.6" fill="${c.col}"/>
        <text x="${c.x.toFixed(1)}" y="${labY}" fill="${c.col}" font-size="10.5" font-family="var(--mono)" font-weight="600" text-anchor="middle">${c.lab}</text>`;
    })
    .join("");

  const xlabels = monthTicks(d0, d1)
    .map((d) => {
      const x = fx(d.toISOString().slice(0, 10));
      return `<text x="${x.toFixed(1)}" y="${VH - 14}" fill="var(--text-dim)" font-size="10.5" font-family="var(--mono)" text-anchor="middle">${MONTH_NAMES[d.getUTCMonth()]} ${String(d.getUTCFullYear()).slice(2)}</text>`;
    })
    .join("");

  chartEl.innerHTML = `<svg viewBox="0 0 ${VW} ${VH}" preserveAspectRatio="xMidYMid meet">
    ${ltSvg}${b3Svg}${cpSvg}${xlabels}
    <line x1="${PL}" y1="${TOP2 + LANE_H}" x2="${VW - PR}" y2="${TOP2 + LANE_H}" stroke="var(--border)" stroke-width="1"/></svg>`;

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
    <td class="date">${r.date}</td>
    <td><span class="badge ${r.method}">${r.method}</span></td>
    <td class="r mag" style="color:${r.method === "lt" ? "var(--accent)" : "var(--b3it)"}">${r.mag}</td>
    <td class="r num" style="color:var(--text-muted)">${r.conf}</td></tr>`
    )
    .join("");
}

async function init(): Promise<void> {
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

  const lt = buildLT(scores);
  const b3it = buildB3IT(b3itData);

  renderStatusCard(lt, b3it);
  renderChart(lt, b3it);
  renderChangesTable(lt, b3it);
}

init();
