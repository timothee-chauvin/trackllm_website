// The endpoint chart's pointer readout. Each lane is read on its own: hovering or
// tapping one names the day under the pointer and that lane's value on it, in that
// lane's units. There is no crosshair across both -- the two lanes answer different
// questions and are compared through the changes table, not by eye at a pixel.
import { esc } from "./components";
import {
  type FocusB3IT,
  type FocusLT,
  type LaneGeom,
  LANE_H,
  chartAxis,
  laneGeoms,
  laneY,
} from "./chart_geom";

const MARK_R = 3.4;
const TIP_DX = 14; // the tip sits beside the pointer, never under it
const TIP_DY = -10;

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

/** A transparent target over each lane that has a trace, plus the group the readout
 *  marker is drawn into. Emitted last so nothing in the chart is above them.
 *  aria-hidden: the changes table below the chart is the keyboard and screen-reader
 *  path, and these would otherwise add two nameless tab stops. */
export function hitRects(lanes: LaneGeom[], pl: number, pw: number): string {
  return (
    `<g class="hover-mark" aria-hidden="true"></g>` +
    lanes
      .filter((l) => l.series.length)
      .map(
        (l) =>
          `<rect class="lane-hit" data-lane="${l.key}" x="${pl}" y="${l.topY}" width="${pw}" height="${LANE_H}" fill="transparent" aria-hidden="true"/>`
      )
      .join("")
  );
}

/** Wire the readout to an already-rendered chart. Called after every draw, so the
 *  resize redraw does not leave the chart inert. */
export function bindHover(
  chartEl: HTMLElement,
  tipEl: HTMLElement,
  lt: FocusLT | null,
  b3it: FocusB3IT | null,
  width: () => number
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

  const hide = (): void => {
    tipEl.hidden = true;
    mark.innerHTML = "";
    pinned = false;
  };

  const show = (lane: LaneGeom, ev: PointerEvent): void => {
    const box = svg.getBoundingClientRect();
    // the SVG is drawn in the container's own pixels, so its user units and its CSS
    // pixels differ only by whatever `.chart svg { width: 100% }` had to scale
    const scale = box.width ? axis.vw / box.width : 1;
    const i = nearestPoint(lane.series, (ev.clientX - box.left) * scale, axis.fx);
    const [date, v] = lane.series[i];
    const x = axis.fx(date);
    mark.innerHTML = `<circle cx="${x.toFixed(1)}" cy="${laneY(lane, v).toFixed(1)}" r="${MARK_R}" fill="${lane.col}" stroke="var(--surface-2)" stroke-width="1.5"/>`;
    tipEl.innerHTML = `<span class="d">${esc(date)}</span><span class="v" style="color:${lane.col}">${esc(lane.fmt(v))}</span>`;
    tipEl.hidden = false;

    const wrap = chartEl.getBoundingClientRect();
    const left = ev.clientX - wrap.left + TIP_DX;
    tipEl.style.left = `${Math.max(0, Math.min(left, wrap.width - tipEl.offsetWidth))}px`;
    tipEl.style.top = `${ev.clientY - wrap.top + TIP_DY}px`;
  };

  const laneOf = (ev: Event): LaneGeom | undefined => {
    const hit = (ev.target as Element).closest?.(".lane-hit");
    return hit ? lanes.get(hit.getAttribute("data-lane") as LaneGeom["key"]) : undefined;
  };

  chartEl.addEventListener("pointermove", (ev) => {
    const e = ev as PointerEvent;
    if (e.pointerType === "touch" || pinned) return;
    const lane = laneOf(e);
    if (lane) show(lane, e);
    else hide();
  });
  chartEl.addEventListener("pointerdown", (ev) => {
    const e = ev as PointerEvent;
    const lane = laneOf(e);
    if (!lane) return hide();
    pinned = e.pointerType === "touch";
    show(lane, e);
  });
  chartEl.addEventListener("pointerleave", () => {
    if (!pinned) hide();
  });
}
