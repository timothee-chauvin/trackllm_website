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

export const MONTH_NAMES = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** "2026-07" -> "Jul '26" — axis ticks and month headers. */
export function monthLabel(month: string): string {
  return MONTH_NAMES[+month.slice(5, 7) - 1] + " '" + month.slice(2, 4);
}

/** "2026-07-24" -> "Jul 2026"; null -> em dash. */
export function prettyDate(date: string | null): string {
  return date ? MONTH_NAMES[+date.slice(5, 7) - 1] + " " + date.slice(0, 4) : "—";
}

export function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

/** How far the endpoint moved, in each method's own unit. */
export function magnitudeLabel(method: string, magnitude: number | null): string {
  if (magnitude === null) return "";
  return method === "lt" ? `${magnitude} nats` : `TV ${magnitude}`;
}

/** One detected change, as feed.py enriches it for both the Overview and the change log. */
export interface FeedItem {
  date: string;
  iso: string;
  daysAgo: number;
  slug: string;
  model: string;
  org: string;
  modelSlug: string;
  provider: string;
  providerSlug: string;
  method: "lt" | "b3it";
  desc: string;
  primary: string;
  secondary: string;
  sevKey: "alert" | "changed" | "stable";
  trace: number[];
  changeFrac: number;
  magnitude: number | null;
}

/** A change-feed row. Both surfaces that render it (index.html, changes.html) sit at
 *  the site root, so the link paths are root-relative with no prefix. */
export function eventRow(e: FeedItem): string {
  const isLT = e.method === "lt";
  const color = isLT ? "var(--accent)" : "var(--b3it)";
  return `<div class="event" style="--sev:var(--${e.sevKey})">
    <div class="when">${esc(e.date)}<span class="rel">${relDays(e.daysAgo)}</span></div>
    <div class="what">
      <div><a class="model" href="models/${esc(e.modelSlug)}.html">${esc(e.model)}</a>
        <span class="at">@ <a class="at" href="providers/${esc(e.providerSlug)}.html">${esc(e.provider)}</a></span></div>
      <div class="desc">${esc(e.desc)}</div>
    </div>
    <div class="spark">${sparkline(e.trace, isLT ? LT_CAP : B3IT_CAP, color, e.changeFrac)}</div>
    <div class="mag">${methodBadges([e.method])}
      <div class="delta">${esc(e.primary)}</div>
      <div class="conf">${esc(e.secondary)}</div>
    </div>
  </div>`;
}
