// The model page's shared readout. The endpoint page reads one lane at a time,
// because its two lanes answer different questions; here every strip answers the
// same one -- how far has this provider moved -- so one pointer position reads all
// of them at once: a crosshair down the whole timeline, each row's own value in the
// column that held its counts, and the day itself in the tip the endpoint page uses.
import { DAY_MS, esc } from "./components";
import { type Cell, MARK_R, placeTip, tipHTML } from "./chart_tip";
import { fmtDrift, fmtTV, sampleAt } from "./chart_geom";

/** Every strip is drawn in these user units and stretched to the column's width. */
export const STRIP_VW = 1000;

const LT_COL = "var(--accent)";
const B3IT_COL = "var(--b3it)";

/** The series a model-page row can be read off; the shape model.ts's rows already have. */
export interface HoverSeries {
  lt: { drift: [string, number][] } | null;
  b3it: { tv: [string, number][] } | null;
}

/** What this endpoint measured on `date`, in each method's own units. A method
 *  whose series does not cover the day contributes nothing rather than its nearest
 *  sample: the point of reading every provider at one instant is lost if some of
 *  them answer with a different one. */
export function readCells(ep: HoverSeries, date: string): Cell[] {
  const cells: Cell[] = [];
  const lt = ep.lt ? sampleAt(ep.lt.drift, date) : null;
  const tv = ep.b3it ? sampleAt(ep.b3it.tv, date) : null;
  if (lt !== null) cells.push({ text: fmtDrift(lt), col: LT_COL });
  if (tv !== null) cells.push({ text: fmtTV(tv), col: B3IT_COL });
  return cells.length ? cells : [{ text: "—", col: "var(--text-dim)" }];
}

/** The day `frac` of the way along the axis. Snapped to a whole day so every strip
 *  is read at the same instant, whatever its own sampling. */
export function dayAt(d0: number, d1: number, frac: number): string {
  const t = d0 + frac * (d1 - d0);
  return new Date(Math.round(t / DAY_MS) * DAY_MS).toISOString().slice(0, 10);
}

/** One strip on the shared timeline: where to draw into, and how it reads. */
export interface StripRow {
  svg: Element; // measured for the pointer -> date mapping, and the crosshair's host
  mark: Element; // the <g class="hover-mark"> the crosshair is written to
  height: number; // its viewBox height
  // null on the all-endpoints strip: it counts changes rather than measuring a level
  read: HTMLElement | null;
  stat: HTMLElement | null; // what the readout stands in for while reading
  dot: (date: string) => { y: number; col: string } | null; // the point on its drawn curve
  cells: (date: string) => Cell[];
}

/** A rule the full height of the strip, and the sample it crosses. The rule is
 *  non-scaling and the dot an ellipse because the strips are drawn at STRIP_VW and
 *  squeezed to the column: a 1-unit line would come out hairline-thin, and a circle
 *  as a flattened oval. */
function crosshair(
  x: number,
  h: number,
  dot: { y: number; col: string } | null,
  xScale: number
): string {
  return (
    `<line x1="${x.toFixed(1)}" y1="0" x2="${x.toFixed(1)}" y2="${h}" stroke="var(--text-muted)" stroke-width="1" opacity="0.8" vector-effect="non-scaling-stroke"/>` +
    (dot === null
      ? ""
      : `<ellipse cx="${x.toFixed(1)}" cy="${dot.y.toFixed(1)}" rx="${(MARK_R * xScale).toFixed(1)}" ry="${MARK_R}" fill="${dot.col}" stroke="var(--surface-2)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`)
  );
}

const cellHTML = (cells: Cell[]): string =>
  cells.map((c) => `<div class="rv" style="color:${c.col}">${esc(c.text)}</div>`).join("");

/** Wire the shared readout. `wrap` is the positioned box the tip is absolute in,
 *  `xpos` the strips' own date -> user-unit mapping. */
export function bindSharedHover(
  cmpEl: HTMLElement,
  wrap: HTMLElement,
  tipEl: HTMLElement,
  rows: StripRow[],
  xpos: (date: string) => number,
  dateAt: (frac: number) => string
): void {
  if (!rows.length) return;
  // A press pins the readout: a touch fires no pointerleave, so it stays until the
  // next press lands somewhere else -- and survives the scroll that follows, which
  // is how a reader compares rows further down the page against it.
  let pinned = false;

  const hide = (): void => {
    tipEl.hidden = true;
    pinned = false;
    for (const r of rows) {
      r.mark.innerHTML = "";
      if (r.read && r.stat) {
        r.read.hidden = true;
        r.stat.hidden = false;
      }
    }
  };

  const show = (hit: StripRow, ev: PointerEvent): void => {
    const box = hit.svg.getBoundingClientRect();
    const frac = box.width ? (ev.clientX - box.left) / box.width : 0;
    const date = dateAt(Math.min(1, Math.max(0, frac)));
    const x = xpos(date);
    // the strips share one axis and one column width, so one measurement places
    // the crosshair on all of them
    const xScale = box.width ? STRIP_VW / box.width : 1;
    for (const r of rows) {
      r.mark.innerHTML = crosshair(x, r.height, r.dot(date), xScale);
      if (r.read && r.stat) {
        r.read.innerHTML = cellHTML(r.cells(date));
        r.read.hidden = false;
        r.stat.hidden = true;
      }
    }
    tipEl.innerHTML = tipHTML(date, hit.cells(date));
    tipEl.hidden = false;
    placeTip(tipEl, wrap, ev);
  };

  const hitOf = (ev: Event): StripRow | undefined => {
    const svg = (ev.target as Element).closest?.(".spark")?.querySelector("svg");
    return svg ? rows.find((r) => r.svg === svg) : undefined;
  };

  // A miss inside the panel keeps the last reading, unlike the endpoint chart's
  // per-lane readout: the rows are stacked with a seam of padding between their
  // strips, and clearing on every seam would blink the readout off exactly when the
  // reader is running down the column of values it just put up. Leaving the panel
  // still clears it.
  cmpEl.addEventListener("pointermove", (ev) => {
    const e = ev as PointerEvent;
    // touch: only a finger already down scrubs
    if (e.pointerType === "touch" && !pinned) return;
    const hit = hitOf(e);
    if (hit) show(hit, e);
  });
  cmpEl.addEventListener("pointerdown", (ev) => {
    const e = ev as PointerEvent;
    const hit = hitOf(e);
    if (!hit) return hide();
    pinned = e.pointerType === "touch";
    show(hit, e);
  });
  cmpEl.addEventListener("pointerleave", () => {
    if (!pinned) hide();
  });
}
