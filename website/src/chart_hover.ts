// The endpoint chart's pointer readout. Each lane is read on its own: hovering or
// tapping one names the day under the pointer and that lane's value on it, in that
// lane's units, over the raw data the value was computed from. There is no
// crosshair across both -- the two lanes answer different questions and are compared
// through the changes table, not by eye at a pixel. Change marks and epoch rules are
// targets of their own: they explain the mark, not the nearest day.
import { MARK_R, placeTip, tipHTML } from "./chart_tip";
import {
  type RawLoaders,
  b3itChangeDetail,
  b3itDayDetail,
  epochDetail,
  ltChangeDetail,
  ltDayDetail,
} from "./chart_detail";
import {
  type FocusB3IT,
  type FocusLT,
  type LaneGeom,
  LANE_H,
  TOP1,
  TOP2,
  chartAxis,
  laneGeoms,
  laneY,
} from "./chart_geom";

/** Index of the sample nearest `xPx` (SVG user units). The series is sorted, but
 *  short enough that a scan is cheaper to read than a bisection. */
export function nearestPoint(
  series: [string, number][],
  xPx: number,
  fx: (s: string) => number
): number {
  let best = 0;
  let bestD = Infinity;
  series.forEach(([d], i) => {
    const dist = Math.abs(fx(d) - xPx);
    if (dist < bestD) {
      bestD = dist;
      best = i;
    }
  });
  return best;
}

/** Where the change marks and epoch rules were drawn, for their own hit targets. */
export interface Targets {
  marks: { lane: LaneGeom["key"]; i: number; x: number }[];
  epochs: { i: number; x: number }[];
}

const STRIP_W = 8; // a rule is one pixel wide; its target is this wide

/** A transparent target over each lane that has a trace, a strip over every
 *  change mark and epoch rule, plus the group the readout marker is drawn into.
 *  Emitted last so nothing in the chart is above them, the strips last of all.
 *  aria-hidden: the changes table below the chart is the keyboard and screen-reader
 *  path, and these would otherwise add nameless tab stops. */
export function hitRects(lanes: LaneGeom[], pl: number, pw: number, targets: Targets): string {
  const rect = (cls: string, data: string, x: number, y: number, w: number, h: number): string =>
    `<rect class="${cls}" ${data} x="${x.toFixed(1)}" y="${y}" width="${w.toFixed(1)}" height="${h}" fill="transparent" aria-hidden="true"/>`;
  const strip = (cls: string, data: string, x: number, y: number, h: number): string =>
    rect(cls, data, x - STRIP_W / 2, y, STRIP_W, h);
  return (
    `<g class="hover-mark" aria-hidden="true"></g>` +
    lanes
      .filter((l) => l.series.length)
      .map((l) => rect("lane-hit", `data-lane="${l.key}"`, pl, l.topY, pw, LANE_H))
      .join("") +
    targets.epochs.map((e) => strip("epoch-hit", `data-epoch="${e.i}"`, e.x, TOP2, LANE_H)).join("") +
    targets.marks
      .map((m) =>
        strip("cp-hit", `data-lane="${m.lane}" data-cp="${m.i}"`, m.x, TOP1 - 4, TOP2 + LANE_H - (TOP1 - 4))
      )
      .join("")
  );
}

/** Wire the readout to an already-rendered chart. Called after every draw, so the
 *  resize redraw does not leave the chart inert. A readout whose raw data has not
 *  arrived yet (endpoint.ts fetches it as soon as the page is idle) shows the value
 *  at once and fills in when it lands. */
export function bindHover(
  chartEl: HTMLElement,
  tipEl: HTMLElement,
  lt: FocusLT | null,
  b3it: FocusB3IT | null,
  width: () => number,
  raw: RawLoaders
): void {
  const svg = chartEl.querySelector("svg");
  const mark = chartEl.querySelector(".hover-mark");
  if (!svg || !mark) return;
  const axis = chartAxis(lt, b3it, width());
  if (!axis) return;
  const lanes = new Map(laneGeoms(lt, b3it).map((l) => [l.key, l]));

  // A tap pins the readout: touch fires no pointerleave, so it stays until the next
  // press lands somewhere else.
  let pinned = false;
  // what the tip is showing, so a detail block that loads late lands on the readout
  // that asked for it and not on whatever the pointer moved to meanwhile
  let current: string | null = null;

  const hide = (): void => {
    tipEl.hidden = true;
    mark.innerHTML = "";
    pinned = false;
    current = null;
  };

  // Placed once per readout: a pointer drifting within one point's range, or a
  // detail block landing late, must not move a tip the reader is heading into.
  const render = (key: string, head: string, detail: string, ev: PointerEvent): void => {
    const moved = key !== current;
    current = key;
    tipEl.innerHTML = head + detail;
    tipEl.classList.toggle("has-detail", !!detail);
    tipEl.classList.toggle("pinned", pinned);
    tipEl.hidden = false;
    if (moved) placeTip(tipEl, chartEl, ev);
  };

  const showDay = (lane: LaneGeom, ev: PointerEvent): void => {
    const box = svg.getBoundingClientRect();
    // the SVG is drawn in the container's own pixels, so its user units and its CSS
    // pixels differ only by whatever `.chart svg { width: 100% }` had to scale
    const scale = box.width ? axis.vw / box.width : 1;
    const i = nearestPoint(lane.series, (ev.clientX - box.left) * scale, axis.fx);
    const [date, v] = lane.series[i];
    const x = axis.fx(date);
    mark.innerHTML = `<circle cx="${x.toFixed(1)}" cy="${laneY(lane, v).toFixed(1)}" r="${MARK_R}" fill="${lane.col}" stroke="var(--surface-2)" stroke-width="1.5"/>`;
    const head = tipHTML(date.slice(0, 10), [{ text: lane.fmt(v), col: lane.col }]);
    const key = `${lane.key}:${date}`;
    render(key, head, "", ev);
    if (lane.key === "b3it" && b3it) {
      const k = b3it.daily.findIndex(([d]) => d === date);
      if (k < 0) return;
      void raw.b3it().then((votes) => {
        if (votes && current === key) render(key, head, b3itDayDetail(b3it, votes, k), ev);
      });
    } else {
      void raw.lt().then((logprobs) => {
        if (logprobs && current === key) render(key, head, ltDayDetail(logprobs, date), ev);
      });
    }
  };

  const showMark = (laneKey: LaneGeom["key"], i: number, ev: PointerEvent): void => {
    mark.innerHTML = "";
    const col = lanes.get(laneKey)!.col;
    const head = (date: string): string => tipHTML(date, [{ text: "Change detected", col }]);
    if (laneKey === "lt" && lt) {
      render(`cp:lt:${i}`, head(lt.changes[i].date), ltChangeDetail(lt, lt.changes[i]), ev);
    } else if (laneKey === "b3it" && b3it) {
      render(`cp:b3it:${i}`, head(b3it.changes[i].date), b3itChangeDetail(b3it, b3it.changes[i]), ev);
    }
  };

  const showEpoch = (i: number, ev: PointerEvent): void => {
    if (!b3it) return;
    mark.innerHTML = "";
    const key = `ep:${i}`;
    const head = tipHTML(b3it.epochs[i].start.slice(0, 10), [
      { text: `Epoch ${i + 1}`, col: lanes.get("b3it")!.col },
    ]);
    render(key, head, epochDetail(b3it, null, i), ev);
    void raw.b3it().then((votes) => {
      if (votes && current === key) render(key, head, epochDetail(b3it, votes, i), ev);
    });
  };

  /** What the pointer is on, as the readout to show for it; null off every target. */
  const readoutAt = (ev: PointerEvent): (() => void) | null => {
    const el = (ev.target as Element).closest?.(".cp-hit, .epoch-hit, .lane-hit");
    if (!el) return null;
    const lane = el.getAttribute("data-lane") as LaneGeom["key"] | null;
    if (el.classList.contains("cp-hit") && lane) {
      return () => showMark(lane, +el.getAttribute("data-cp")!, ev);
    }
    if (el.classList.contains("epoch-hit")) return () => showEpoch(+el.getAttribute("data-epoch")!, ev);
    const geom = lane ? lanes.get(lane) : undefined;
    return geom ? () => showDay(geom, ev) : null;
  };

  chartEl.addEventListener("pointermove", (ev) => {
    const e = ev as PointerEvent;
    if (e.pointerType === "touch" || pinned) return;
    const show = readoutAt(e);
    if (show) show();
    else hide();
  });
  chartEl.addEventListener("pointerdown", (ev) => {
    const e = ev as PointerEvent;
    const show = readoutAt(e);
    if (!show) return hide();
    pinned = e.pointerType === "touch";
    show();
  });
  // The tip is the chart's sibling: leaving the chart for the tip (to select a
  // line of it) is not leaving the readout, so the box holding both is what hides.
  (tipEl.parentElement ?? chartEl).addEventListener("pointerleave", () => {
    if (!pinned) hide();
  });
}
