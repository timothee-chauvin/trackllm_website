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

/** Mirrors status.py::HEADLINE_ORDER — the priority chain, strongest first. */
export const HEADLINE_ORDER = [
  "tracked",
  "retired",
  "untrackable",
  "too_expensive",
  "not_selected",
  "errors_out",
  "pending",
  "free_excluded",
];

/** esc() the text, wrapping the first case-insensitive match of q in <mark>. */
export function highlight(text: string, q: string): string {
  const i = q ? text.toLowerCase().indexOf(q.toLowerCase()) : -1;
  if (i < 0) return esc(text);
  return (
    esc(text.slice(0, i)) +
    "<mark>" +
    esc(text.slice(i, i + q.length)) +
    "</mark>" +
    esc(text.slice(i + q.length))
  );
}

/** Render an explicit "failed to load" card into `mountId`. A fetch failure must
 *  read as a failure — never as a blank page or, worse, a "no data" claim. */
export function showLoadError(mountId: string, what: string): void {
  const el = document.getElementById(mountId);
  if (!el) return;
  el.innerHTML = `<div class="empty load-error">Failed to load ${esc(what)}.
    This is a loading error, not an absence of data — try reloading the page.</div>`;
}

/** Headline status badge; must stay in sync with the Jinja macro (_macros.html.j2). */
export function headlineBadge(headline: string): string {
  return `<span class="badge st st-${headline.replace(/_/g, "-")}">${esc(headline.replace(/_/g, " "))}</span>`;
}

/** Sort rank for a directory's Status column: tracked rows by trace status,
 *  untracked rows after them in headline priority order. */
export function statusRank(r: {
  methods: string[];
  status: string | null;
  headline: string;
}): number {
  const TRACE: Record<string, number> = { changed: 0, stable: 1, retired: 2 };
  return r.methods.length
    ? (TRACE[r.status ?? ""] ?? 3)
    : 3 + HEADLINE_ORDER.indexOf(r.headline);
}

/** The five directory cells after model/provider for a row with no series:
 *  headline badge in the status column, the one-line reason in the trace column. */
export function untrackedDirCells(
  r: { slug: string; headline: string; reason: string },
  root: string
): string {
  return `<td><a href="${root}endpoints/${esc(r.slug)}.html">${headlineBadge(r.headline)}</a></td>
    <td class="r"><span class="cc zero">—</span></td>
    <td class="col-hide"></td>
    <td class="r col-hide"><span class="org-cell">—</span></td>
    <td class="col-hide reason-cell"><span class="reason" title="${esc(r.reason)}">${esc(r.reason)}</span></td>`;
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

const MINUTE_MS = 60_000;
const HOUR_MS = 60 * MINUTE_MS;
export const DAY_MS = 24 * HOUR_MS;
const COARSE_DAYS = 30; // past a month, the hour is noise

/** Age of an absolute instant: "14m ago", "3h07m ago", "2d 4h ago", "45d ago".
 *  Takes `now` from the caller because a static build must never bake in an age.
 *  A timestamp from the future (clock skew) reads "0m ago", not a countdown. */
export function relativeAge(iso: string, now: number): string {
  const then = Date.parse(iso);
  if (!Number.isFinite(then)) throw new Error(`unparseable timestamp: ${iso}`);
  const age = Math.max(0, now - then);
  const d = Math.floor(age / DAY_MS);
  const h = Math.floor((age % DAY_MS) / HOUR_MS);
  const m = Math.floor((age % HOUR_MS) / MINUTE_MS);
  if (age < HOUR_MS) return `${m}m ago`;
  if (age < DAY_MS) return `${h}h${String(m).padStart(2, "0")}m ago`;
  if (d < COARSE_DAYS) return `${d}d ${h}h ago`;
  return `${d}d ago`;
}

export const MONTH_NAMES = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** Day of a date/datetime string as ms at UTC midnight. */
export const td = (s: string): number => Date.parse(s.slice(0, 10) + "T00:00:00Z");

/** First-of-month tick instants for a [d0, d1] ms range; the month start just
 *  before d0 is kept only when it is within 15 days, so the first label never
 *  sits far outside the plotted span. */
export function monthTicks(d0: number, d1: number): Date[] {
  const out: Date[] = [];
  const d = new Date(d0);
  d.setUTCDate(1);
  while (d.getTime() <= d1) {
    if (d.getTime() >= d0 - 15 * DAY_MS) out.push(new Date(d));
    d.setUTCMonth(d.getUTCMonth() + 1);
  }
  return out;
}

/** Delegated activation for the widgets that are not native buttons: a pointer
 *  click, plus the Enter/Space contract a role="button" element owes a keyboard.
 *  Space is prevented so activating a control never scrolls the page instead. */
export function bindActivation(
  el: Element,
  selector: string,
  handler: (target: HTMLElement) => void
): void {
  const target = (e: Event): HTMLElement | null =>
    (e.target as HTMLElement).closest(selector);
  el.addEventListener("click", (e) => {
    const t = target(e);
    if (t) handler(t);
  });
  el.addEventListener("keydown", (e) => {
    const ev = e as KeyboardEvent;
    if (ev.key !== "Enter" && ev.key !== " ") return;
    const t = target(ev);
    if (!t) return;
    ev.preventDefault();
    handler(t);
  });
}

/** Toggle `f` in `set`, mirroring membership in the chip's .on class and, for
 *  assistive tech, in its aria-pressed state. */
export function toggleChip(chip: HTMLElement, set: Set<string>, f: string): void {
  if (set.has(f)) { set.delete(f); chip.classList.remove("on"); }
  else { set.add(f); chip.classList.add("on"); }
  chip.setAttribute("aria-pressed", String(set.has(f)));
}

/** How many marks a strip spells out before it summarises the rest. */
const TIP_MAX = 6;

/** An SVG <title> only ever appears on hover, so a touch device never sees it --
 *  and inside a `preserveAspectRatio="none"` strip a change mark is squeezed to
 *  about a pixel wide on a phone, far too small to aim at anyway. So the strip as
 *  a whole carries the story: one tab stop, one full-width tap target, the same
 *  words as its accessible name and as the caption bindTips writes on activation.
 *  A strip with no marks stays decorative. */
export function stripTip(lead: string, parts: string[]): string {
  if (!parts.length) return ' aria-hidden="true"';
  const shown = parts.slice(0, TIP_MAX);
  const more = parts.length > TIP_MAX ? ` · +${parts.length - TIP_MAX} more` : "";
  const text = esc(`${lead}: ${shown.join(" · ")}${more}`);
  return ` tabindex="0" role="img" aria-label="${text}" data-tip="${text}"`;
}

/** The line a tapped strip writes into. aria-hidden because a screen reader has
 *  already read the same words off the strip that put them there. Empty until
 *  something is tapped, and styled to take no space until then. */
export const TIP_LINE = '<div class="tipline" aria-hidden="true"></div>';

/** Tapping or focusing a strip shows its text in the chart's caption line. */
export function bindTips(root: Element): void {
  const line = root.querySelector(".tipline");
  if (!line) return;
  const show = (e: Event): void => {
    const el = (e.target as Element).closest?.("[data-tip]");
    if (el) line.textContent = el.getAttribute("data-tip");
  };
  root.addEventListener("click", show);
  root.addEventListener("focusin", show);
}

/** The standard filter toolbar: each .chip toggles its data-f flag, then re-renders. */
export function bindFilterChips(el: Element, filters: Set<string>, render: () => void): void {
  bindActivation(el, ".chip", (chip) => {
    toggleChip(chip, filters, chip.dataset.f!);
    render();
  });
}

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
  endpointSlug: string;
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
 *  the site root, so the link paths are root-relative with no prefix.
 *  "model @ provider" names one endpoint, so it is one link to that endpoint's page. */
export function eventRow(e: FeedItem): string {
  const isLT = e.method === "lt";
  const color = isLT ? "var(--accent)" : "var(--b3it)";
  // no endpointSlug: the endpoint has left the fleet and no page was generated for
  // it (feed.py leaves its page slugs empty), so the name is text, not a 404 link
  const names = `<span class="model">${esc(e.model)}</span>
        <span class="at">@ ${esc(e.provider)}</span>`;
  return `<div class="event" style="--sev:var(--${e.sevKey})">
    <div class="when">${esc(e.date)}<span class="rel">${relDays(e.daysAgo)}</span></div>
    <div class="what">
      <div>${e.endpointSlug ? `<a href="endpoints/${esc(e.endpointSlug)}.html">${names}</a>` : names}</div>
      <div class="desc">${esc(e.desc)}</div>
    </div>
    <div class="spark">${sparkline(e.trace, isLT ? LT_CAP : B3IT_CAP, color, e.changeFrac)}</div>
    <div class="mag">${methodBadges([e.method])}
      <div class="delta">${esc(e.primary)}</div>
      <div class="conf">${esc(e.secondary)}</div>
    </div>
  </div>`;
}
