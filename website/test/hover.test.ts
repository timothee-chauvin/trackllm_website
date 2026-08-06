/**
 * The chart's pointer readout: the date and value under the pointer, per lane.
 *
 * happy-dom has no layout engine, so the SVG's client rect is stubbed to the size
 * the chart was drawn at -- that is the one measurement the handler takes, and
 * stubbing it is what lets the mapping be exercised at all. Everything downstream
 * of it is real.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, beforeEach, describe, expect, test } from "bun:test";

const DESIGN_VW = 1000;
const VH = 324;
const PL = 50, PR = 20;
const PW = DESIGN_VW - PL - PR;

const B3IT = {
  tv: [
    ["2026-07-17", 0.141], ["2026-07-18", 0.114], ["2026-07-19", 0.1285],
    ["2026-07-20", 0.1165], ["2026-07-21", 0.14], ["2026-07-22", 0.159],
    ["2026-07-23", 0.148], ["2026-07-24", 0.139], ["2026-07-25", 0.5355],
    ["2026-07-26", 0.5355], ["2026-07-27", 0.5035], ["2026-07-28", 0.502],
    ["2026-07-29", 0.4825],
  ] as [string, number][],
  breaks: [], // already thinned: these fixtures are what the chart draws
  changes: [{ date: "2026-07-26", peakTV: 0.535 }],
  firstDate: "2026-07-17",
  lastDate: "2026-07-29",
};

const LT = {
  drift: [
    ["2026-07-17", 0.02], ["2026-07-22", 0.05], ["2026-07-26", 0.81], ["2026-07-29", 0.78],
  ] as [string, number][],
  breaks: [], // already thinned: these fixtures are what the chart draws
  changes: [{ date: "2026-07-26", sigma: "42σ", drift: 0.81 }],
  firstDate: "2026-07-17",
  lastDate: "2026-07-29",
};

beforeAll(() => GlobalRegistrator.register());
afterAll(() => GlobalRegistrator.unregister());

const attr = (el: Element, name: string): string => el.getAttribute(name) ?? "";

/** The chart mounted the way the page mounts it, with the client rect the page
 *  would have measured. Returns everything a test needs to poke at it. */
async function mount(
  lt: typeof LT | null,
  b3it: typeof B3IT | null
): Promise<{ chart: HTMLElement; tip: HTMLElement; hits: Element[] }> {
  const { chartSvg } = await import("../src/endpoint");
  const { bindHover } = await import("../src/chart_hover");
  document.body.innerHTML = `<div class="chartwrap chart" id="mainchart"></div>
    <div class="chart-tip" id="charttip" hidden></div>`;
  const chart = document.getElementById("mainchart")!;
  const tip = document.getElementById("charttip")!;
  chart.innerHTML = chartSvg(lt, b3it, DESIGN_VW);
  const svg = chart.querySelector("svg")!;
  svg.getBoundingClientRect = (): DOMRect =>
    ({ left: 0, top: 0, width: DESIGN_VW, height: VH, right: DESIGN_VW, bottom: VH, x: 0, y: 0 }) as DOMRect;
  chart.getBoundingClientRect = (): DOMRect =>
    ({ left: 0, top: 0, width: DESIGN_VW, height: VH, right: DESIGN_VW, bottom: VH, x: 0, y: 0 }) as DOMRect;
  bindHover(chart, tip, lt, b3it, () => DESIGN_VW);
  return { chart, tip, hits: [...chart.querySelectorAll(".lane-hit")] };
}

/** Fire a pointer event at an x in the SVG's own units (= client px here). */
function point(el: Element, type: string, x: number, pointerType: string): void {
  el.dispatchEvent(
    new window.PointerEvent(type, { clientX: x, clientY: 100, bubbles: true, pointerType })
  );
}

/** x of a date on the chart's axis: the span is 2026-07-17..29 in every fixture. */
const dayX = (day: number): number => PL + ((day - 17) / 12) * PW;

describe("nearestPoint", () => {
  test("lands on the sample under the pointer", async () => {
    const { nearestPoint } = await import("../src/chart_hover");
    const fx = (s: string): number => dayX(+s.slice(8, 10));
    expect(nearestPoint(B3IT.tv, dayX(17), fx)).toBe(0);
    expect(nearestPoint(B3IT.tv, dayX(26), fx)).toBe(9);
    expect(nearestPoint(B3IT.tv, dayX(29), fx)).toBe(12);
  });

  test("resolves the midpoint between two samples to one of them", async () => {
    const { nearestPoint } = await import("../src/chart_hover");
    const fx = (s: string): number => dayX(+s.slice(8, 10));
    expect([3, 4]).toContain(nearestPoint(B3IT.tv, dayX(20.5), fx));
  });

  test("clamps past either end of the series", async () => {
    const { nearestPoint } = await import("../src/chart_hover");
    const fx = (s: string): number => dayX(+s.slice(8, 10));
    expect(nearestPoint(B3IT.tv, -500, fx)).toBe(0);
    expect(nearestPoint(B3IT.tv, 5000, fx)).toBe(B3IT.tv.length - 1);
  });
});

describe("hit rects", () => {
  test("one per lane with a trace, and none for a lane without", async () => {
    const both = await mount(LT, B3IT);
    expect(both.hits.length).toBe(2);
    const b3only = await mount(null, B3IT);
    expect(b3only.hits.length).toBe(1);
    expect(attr(b3only.hits[0], "data-lane")).toBe("b3it");
  });

  test("adds no tab stop and no accessible name", async () => {
    const { hits } = await mount(LT, B3IT);
    for (const h of hits) {
      expect(attr(h, "aria-hidden")).toBe("true");
      expect(h.hasAttribute("tabindex")).toBe(false);
    }
  });
});

describe("readout", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
  });

  test("a hover names the day and that lane's value", async () => {
    const { tip, hits } = await mount(null, B3IT);
    point(hits[0], "pointermove", dayX(26), "mouse");
    expect(tip.hidden).toBe(false);
    expect(tip.textContent).toContain("2026-07-26");
    expect(tip.textContent).toContain("TV 0.536");
  });

  test("each lane reads its own units", async () => {
    const { tip, hits } = await mount(LT, B3IT);
    const lt = hits.find((h) => attr(h, "data-lane") === "lt")!;
    point(lt, "pointermove", dayX(26), "mouse");
    expect(tip.textContent).toContain("0.81 nats");
    expect(tip.textContent).not.toContain("TV");
  });

  test("leaving the lane takes the readout away", async () => {
    const { tip, hits } = await mount(null, B3IT);
    point(hits[0], "pointermove", dayX(26), "mouse");
    point(hits[0], "pointerleave", dayX(26), "mouse");
    expect(tip.hidden).toBe(true);
  });

  test("a tap pins the readout, and one outside dismisses it", async () => {
    const { chart, tip, hits } = await mount(null, B3IT);
    point(hits[0], "pointerdown", dayX(26), "touch");
    expect(tip.hidden).toBe(false);
    expect(tip.textContent).toContain("2026-07-26");
    // a touch never fires pointerleave on its own -- the tip must survive until
    // something else is touched
    point(hits[0], "pointerleave", dayX(26), "touch");
    expect(tip.hidden).toBe(false);
    point(chart, "pointerdown", 5, "touch");
    expect(tip.hidden).toBe(true);
  });

  test("marks the read sample on the curve", async () => {
    const { chart, hits } = await mount(null, B3IT);
    point(hits[0], "pointermove", dayX(26), "mouse");
    const dot = chart.querySelector(".hover-mark circle");
    expect(dot, "no marker drawn on the curve").not.toBeNull();
    expect(+attr(dot!, "cx")).toBeCloseTo(dayX(26), 0);
  });
});
