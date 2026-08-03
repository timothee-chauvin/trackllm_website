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

/** Why a headline is what it is, in the visitor's words -- the popover text for
 *  headlineBadge. Must stay in sync with STATUS_COPY (status.py): every
 *  HEADLINE_ORDER name is also a STATUS_COPY key, with the same meaning, so
 *  this is that subset copied by hand (the Jinja macro reads STATUS_COPY
 *  itself instead, via the Jinja global render.py registers). */
export const HEADLINE_COPY: Record<string, string> = {
  tracked: "This endpoint is actively tracked.",
  retired: "We tracked this endpoint, but tracking has since been retired.",
  untrackable:
    "This endpoint claims neither temperature control nor logprobs, so no tracking method can work — presumably to prevent distillation.",
  too_expensive: "This endpoint costs more than our tracking budget allows.",
  not_selected:
    "This endpoint vetted fine, but the selection budget went to more popular models.",
  errors_out: "Our probes of this endpoint error out.",
  pending: "This endpoint has not been evaluated for tracking yet.",
  free_excluded:
    "Free endpoints are excluded from tracking: their routing and rate limits are too unstable.",
};

/** Headline status badge; must stay in sync with the Jinja macro (_macros.html.j2).
 *  data-tip is bindTips' contract (see below): hover, tap, and Enter/Space all
 *  read it off the element itself, so one popover implementation serves every
 *  badge and chip in the site. */
export function headlineBadge(headline: string): string {
  return `<span class="badge st st-${headline.replace(/_/g, "-")}" role="button" tabindex="0" data-tip="${esc(HEADLINE_COPY[headline])}">${esc(headline.replace(/_/g, " "))}</span>`;
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
 *  headline badge in the status column, the one-line reason in the trace column.
 *  The badge is its own tap target (bindTips) and a real link to the endpoint page
 *  has to sit somewhere too -- an anchor cannot wrap a button, so .cell-go is a
 *  sibling stretched over the cell instead (.cell-tip in style.css), under the
 *  badge in z-order. See trackedDirCells (directory.ts) for the tracked twin. */
export function untrackedDirCells(
  r: { slug: string; headline: string; reason: string },
  root: string
): string {
  return `<td class="cell-tip">${headlineBadge(r.headline)}<a class="cell-go" href="${root}endpoints/${esc(r.slug)}.html" aria-label="View endpoint"></a></td>
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

/** The trace-status pill's popover text. Mirrors overview.py::_row_state: an
 *  endpoint is "retired" there once its pipeline verdict says so or its last
 *  observation is more than RETIRED_GAP_DAYS (14) old; otherwise "changed" if
 *  its last detected change is within RECENT_CHANGE_DAYS (60d), else "stable". */
const TRACE_COPY: Record<string, string> = {
  changed: "This endpoint is actively tracked and moved within the last 60 days.",
  stable: "This endpoint is actively tracked and has shown no change in the last 60 days.",
  retired:
    "This endpoint was tracked, but has gone quiet: either the pipeline retired it, or it hasn't answered in over 14 days.",
};

export function statusPill(status: string): string {
  return `<span class="pill ${status}" role="button" tabindex="0" data-tip="${esc(TRACE_COPY[status])}"><span class="led"></span>${status}</span>`;
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

export interface Tick {
  t: number;
  label: string;
}

// Coarsest first is what the caller wants and finest first is what reads well, so
// the ladder is climbed, not descended: the first rung whose ticks all fit is used.
const DAY_STEPS = [1, 2, 3, 7, 14];
const MONTH_STEPS = [1, 2, 3, 6, 12];
// Past this the day rungs are skipped even where they would fit: on a seven-month
// chart "Jul 3 · Jul 17 · Jul 31 · ..." spends sixteen labels saying less than
// seven monthly ones, and drops the year at the January the reader needs it at.
const DAY_TICK_MAX_SPAN = 70 * DAY_MS;

const dayLabel = (t: number): string => {
  const d = new Date(t);
  return `${MONTH_NAMES[d.getUTCMonth()]} ${d.getUTCDate()}`;
};
// monthLabel's apostrophe is load-bearing here: without it "Jul 26" is a day on one
// chart and a year on the next.
const tickMonthLabel = (t: number): string => monthLabel(new Date(t).toISOString().slice(0, 7));

/** Ticks every `step` days, aligned to the epoch so the same days are chosen
 *  whatever `d0` happens to be. */
function dayRungTicks(d0: number, d1: number, step: number): number[] {
  const ms = step * DAY_MS;
  const out: number[] = [];
  for (let t = Math.ceil(d0 / ms) * ms; t <= d1; t += ms) out.push(t);
  return out;
}

/** First-of-month ticks every `step` months, aligned to January for the same reason. */
function monthRungTicks(d0: number, d1: number, step: number): number[] {
  const out: number[] = [];
  const d = new Date(d0);
  d.setUTCDate(1);
  d.setUTCHours(0, 0, 0, 0);
  d.setUTCMonth(d.getUTCMonth() - (d.getUTCMonth() % step));
  while (d.getTime() < d0) d.setUTCMonth(d.getUTCMonth() + step);
  while (d.getTime() <= d1) {
    out.push(d.getTime());
    d.setUTCMonth(d.getUTCMonth() + step);
  }
  return out;
}

/** Dated x-axis ticks for a [d0, d1] ms range, at the finest granularity whose
 *  labels still fit in `maxLabels` slots.
 *
 *  monthTicks, which this replaces on the endpoint chart, could return nothing at
 *  all: an endpoint observed for under a month need contain no first-of-month, and
 *  its chart came out with no gridlines and a blank axis. Here the 1-day rung always
 *  yields at least one tick, and the coarsest rung is thinned rather than abandoned,
 *  so the result is never empty and never over budget. */
export function timeTicks(d0: number, d1: number, maxLabels: number): Tick[] {
  if (d1 - d0 <= DAY_TICK_MAX_SPAN) {
    for (const step of DAY_STEPS) {
      const ts = dayRungTicks(d0, d1, step);
      if (ts.length <= maxLabels) return ts.map((t) => ({ t, label: dayLabel(t) }));
    }
  }
  let coarsest: number[] = [];
  for (const step of MONTH_STEPS) {
    coarsest = monthRungTicks(d0, d1, step);
    if (coarsest.length <= maxLabels) break;
  }
  // Past a decade even yearly ticks can outrun a phone's budget.
  const keep = Math.ceil(coarsest.length / Math.max(1, maxLabels));
  return coarsest.filter((_, i) => i % keep === 0).map((t) => ({ t, label: tickMonthLabel(t) }));
}

/** The Enter/Space contract a role="button"/role="img" widget owes a keyboard.
 *  Space is prevented so activating a control never scrolls the page instead. */
export function bindKeyActivation(
  el: Element,
  selector: string,
  handler: (target: HTMLElement) => void
): void {
  el.addEventListener("keydown", (e) => {
    const ev = e as KeyboardEvent;
    if (ev.key !== "Enter" && ev.key !== " ") return;
    const t = (ev.target as HTMLElement).closest<HTMLElement>(selector);
    if (!t) return;
    ev.preventDefault();
    handler(t);
  });
}

/** Delegated activation for the widgets that are not native buttons: a pointer
 *  click, plus the keyboard contract above. */
export function bindActivation(
  el: Element,
  selector: string,
  handler: (target: HTMLElement) => void
): void {
  el.addEventListener("click", (e) => {
    const t = (e.target as HTMLElement).closest<HTMLElement>(selector);
    if (t) handler(t);
  });
  bindKeyActivation(el, selector, handler);
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

/** The caption a strip or badge writes into, built directly under the element that
 *  was activated: one caption at the foot of the panel sits below every other row,
 *  which on a model page listing dozens of endpoints puts it thousands of pixels
 *  down the page -- a reader who taps the first one sees nothing but a focus ring.
 *
 *  A strip's aria-label already says these same words (stripTip), so its caption is
 *  aria-hidden -- purely visual, for the sighted readers a hover-only <title> would
 *  have missed. A badge has no aria-label (its own visible text is its name), so
 *  its caption instead gets an id and becomes the accessible description: bindTips
 *  wires it to the badge via aria-describedby while it is open, which is how a
 *  screen reader user gets the same words a sighted tap or hover reveals. */
let tipId = 0;
function tipLine(text: string, described: boolean): HTMLElement {
  const el = document.createElement("div");
  el.className = "tipline";
  if (described) {
    el.id = `tip-${++tipId}`;
    el.setAttribute("role", "status");
  } else {
    el.setAttribute("aria-hidden", "true");
  }
  el.textContent = text;
  return el;
}

/** How far a press may travel and still count as a tap rather than a scroll. */
const TAP_SLOP = 10;

/** How a caption got opened, ranked by how deliberate the gesture was. A mouse
 *  moving onto a hover-capable element opens it "hover"-cheap, and a real click
 *  on that same element arrives right after (the browser fires mouseover before
 *  the pointer events a click resolves into) -- ORIGIN_RANK is what lets that
 *  click *pin* the caption open instead of reading as "close what hover just
 *  opened". Only a tap/click/Enter on an already-tap-pinned element closes it. */
const ORIGIN_RANK = { hover: 0, focus: 1, tap: 2 };
type TipOrigin = keyof typeof ORIGIN_RANK;

/** Tapping, hovering, or focusing a strip or badge captions it; tapping it again,
 *  leaving it, or moving the pointer or focus elsewhere takes the caption away.
 *  One binding (on `root`, typically document.body) serves every [data-tip]
 *  element on the page, present at bind time or added to the DOM afterwards. */
export function bindTips(root: Element): void {
  let open: { el: Element; origin: TipOrigin } | null = null;
  let press: { el: Element | null; x: number; y: number } | null = null;
  const close = (): void => {
    const line = root.querySelector(".tipline");
    if (line?.hasAttribute("role")) open?.el.removeAttribute("aria-describedby");
    line?.remove();
    open = null;
  };
  const show = (el: Element, origin: TipOrigin): void => {
    // already open on this element: a weaker origin (e.g. a hover while a tap
    // already pinned it) must not steal the pin, so only ever rank up
    if (open?.el === el) {
      if (ORIGIN_RANK[origin] > ORIGIN_RANK[open.origin]) open.origin = origin;
      return;
    }
    close();
    const described = !el.hasAttribute("aria-label");
    const line = tipLine(el.getAttribute("data-tip")!, described);
    el.insertAdjacentElement("afterend", line);
    if (described) el.setAttribute("aria-describedby", line.id);
    open = { el, origin };
  };
  const activate = (el: Element): void => {
    if (open?.el === el && open.origin === "tap") close();
    else show(el, "tap");
  };

  // The gesture is read from the pointer, not from `click`, because showing a
  // caption takes the previous one away: when that one sat above the strip being
  // pressed, the page reflows between press and release, the strip slides out from
  // under the finger, and the browser retargets the click to a common ancestor --
  // the tap is swallowed. What the press started on cannot be retargeted.
  root.addEventListener("pointerdown", (e) => {
    const p = e as PointerEvent;
    press = {
      el: (p.target as Element).closest?.("[data-tip]") ?? null,
      x: p.clientX,
      y: p.clientY,
    };
  });
  root.addEventListener("pointerup", (e) => {
    const p = e as PointerEvent;
    const from = press;
    press = null;
    if (!from) return;
    if (Math.abs(p.clientX - from.x) > TAP_SLOP || Math.abs(p.clientY - from.y) > TAP_SLOP) return;
    if (from.el) activate(from.el);
    // Deferred: closing here shrinks whatever cell the caption was in, and for a
    // badge sharing its cell with a stretched .cell-go link, a mouse's click
    // fires synchronously right after this same pointerup -- Chromium hit-tests
    // it fresh, against the now-shrunk layout, and can retarget it clean off the
    // link. A tick later, the link (if that is what was pressed) has already had
    // its click; nothing here is waiting on the caption being gone yet.
    else setTimeout(close, 0);
  });
  // a scroll that began on a strip is not a tap on it
  root.addEventListener("pointercancel", () => {
    press = null;
  });
  // A tap can grow or shrink the page (opening or closing a caption elsewhere,
  // e.g. a status chip's own popover further up the page) before the browser gets
  // around to synthesizing touch's compatibility click event, and Chromium then
  // retargets that click to whatever is now under the finger -- for a badge
  // sharing its cell with a stretched .cell-go link, that is a real navigation.
  // touchend, unlike pointerup, is what suppresses the click it would otherwise
  // produce; its own target is never retargeted, so it still names the element
  // the finger actually landed on.
  root.addEventListener(
    "touchend",
    (e) => {
      if ((e.target as Element).closest?.("[data-tip]")) e.preventDefault();
    },
    { passive: false }
  );
  bindKeyActivation(root, "[data-tip]", activate);
  // :focus-visible, because a pointer focuses too: a tap that dismisses a caption
  // reflows the page, and the strip that slides under the finger would otherwise
  // catch the focus the browser hands out afterwards and caption itself again.
  // (happy-dom answers false to it, so the keyboard path is browser-verified.)
  root.addEventListener("focusin", (e) => {
    const el = (e.target as Element).closest?.("[data-tip]");
    if (el && open?.el !== el && el.matches(":focus-visible")) show(el, "focus");
  });
  // leaving the element that owns the caption closes it outright, regardless of
  // how it was opened -- a tap-pinned caption should not survive its own blur.
  // Exception: a badge sharing its cell with a stretched .cell-go link (the
  // tracked/untracked directory rows) loses focus to that very link on the
  // press that is about to click it -- closing here would shrink the cell out
  // from under a click already in flight (the retargeting stripTip's own
  // comment above describes) for a real navigation this time, not a caption.
  // The link is about to take the page away regardless, so the caption is left
  // to whatever removes it next.
  root.addEventListener("focusout", (e) => {
    const to = (e as FocusEvent).relatedTarget;
    if (open?.el === e.target && !(to instanceof Node && open.el.parentElement?.contains(to))) {
      close();
    }
  });

  // Hover, for the desktop mice that never tap: gated on a real hover-capable
  // pointer so a touchscreen's compatibility mouse events (dispatched well after
  // the tap this same gesture already handled) never fight the tap-toggle above.
  if (matchMedia("(hover: hover) and (pointer: fine)").matches) {
    root.addEventListener("mouseover", (e) => {
      const el = (e.target as Element).closest?.("[data-tip]");
      if (el) show(el, "hover");
    });
    root.addEventListener("mouseout", (e) => {
      const from = (e.target as Element).closest?.("[data-tip]");
      const to = (e as MouseEvent).relatedTarget;
      // moving onto the caption itself, or deeper into the same element, is not a
      // leave; and a hover only ever closes what hover itself opened -- a click's
      // pin outlives the mouse moving away
      if (
        from &&
        open?.el === from &&
        open.origin === "hover" &&
        !(to instanceof Node && from.parentElement?.contains(to))
      ) {
        close();
      }
    });
  }
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
