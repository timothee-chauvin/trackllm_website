/**
 * The endpoint chart's two reading bugs, guarded at the markup.
 *
 * Both were found on z-ai/glm-5.2 @ cloudflare, whose whole observed span is twelve
 * days in July 2026 and whose one change reaches TV 0.535. happy-dom has no layout
 * engine, so these assert coordinates, not appearance.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, describe, expect, test } from "bun:test";

const DESIGN_VW = 1000; // endpoint.ts's fallback width
const LANE_H = 108, TOP1 = 34, GAP = 34;
const TOP2 = TOP1 + LANE_H + GAP;

/** The reported endpoint's tv_series, verbatim: eight days near 0.13, then the step. */
const B3IT = {
  tv: [
    ["2026-07-17", 0.141], ["2026-07-18", 0.114], ["2026-07-19", 0.1285],
    ["2026-07-20", 0.1165], ["2026-07-21", 0.14], ["2026-07-22", 0.159],
    ["2026-07-23", 0.148], ["2026-07-24", 0.139], ["2026-07-25", 0.5355],
    ["2026-07-26", 0.5355], ["2026-07-27", 0.5035], ["2026-07-28", 0.502],
    ["2026-07-29", 0.4825],
  ] as [string, number][],
  breaks: [], // already thinned: these fixtures are what the chart draws
  changes: [{ date: "2026-07-26", shiftTV: 0.535 }],
  firstDate: "2026-07-17",
  lastDate: "2026-07-29",
};

/** A change stamped after the last sample: nothing on the curve to anchor to. */
const B3IT_LATE_CHANGE = { ...B3IT, changes: [{ date: "2026-08-14", shiftTV: 0.535 }] };

/** Two changes a day apart, both at the same height: their labels want one another's
 *  space, and the packing has to move the second somewhere. */
const B3IT_ADJACENT = {
  ...B3IT,
  changes: [
    { date: "2026-07-25", shiftTV: 0.535 },
    { date: "2026-07-26", shiftTV: 0.536 },
  ],
};

beforeAll(() => GlobalRegistrator.register());
afterAll(() => GlobalRegistrator.unregister());

const attr = (el: Element, name: string): string => el.getAttribute(name) ?? "";
const parse = (markup: string): Element => {
  const host = document.createElement("div");
  host.innerHTML = markup;
  return host.firstElementChild!;
};
const axisLabels = (svg: Element): string[] =>
  [...svg.querySelectorAll("text")]
    .filter((t) => /^\w{3} ('\d\d|\d{1,2})$/.test(t.textContent ?? ""))
    .map((t) => t.textContent!);
const cpLabels = (svg: Element): Element[] =>
  [...svg.querySelectorAll("text")].filter(
    (t) => attr(t, "font-weight") === "600" && attr(t, "font-size") === "10.5"
  );
/** The B3IT lane is fixed 0..1 over LANE_H, so a value's y needs no lookup. */
const b3itY = (v: number): number => TOP2 + LANE_H * (1 - v);

describe("x axis on a span shorter than a month", () => {
  test("labels days instead of coming out blank", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const labels = axisLabels(parse(chartSvg(null, B3IT, DESIGN_VW)));
    expect(labels.length, "the axis was blank: no first-of-month falls in the span")
      .toBeGreaterThanOrEqual(3);
    for (const l of labels) expect(l).toMatch(/^\w{3} \d{1,2}$/);
  });

  test("gridlines are drawn for the same ticks", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const svg = parse(chartSvg(null, B3IT, DESIGN_VW));
    const gridX = new Set(
      [...svg.querySelectorAll("line")]
        .filter((l) => attr(l, "stroke") === "var(--border-soft)" && attr(l, "stroke-width") === "1")
        .map((l) => attr(l, "x1"))
    );
    expect(gridX.size).toBe(axisLabels(svg).length);
  });
});

describe("change marks", () => {
  test("the dot sits on the curve, not at the top of the lane", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const svg = parse(chartSvg(null, B3IT, DESIGN_VW));
    const dot = svg.querySelector("circle")!;
    expect(+attr(dot, "cy")).toBeCloseTo(b3itY(0.5355), 0);
    expect(+attr(dot, "cy"), "still pinned to the lane ceiling").not.toBeCloseTo(TOP2 - 4, 0);
  });

  test("the label rides just above its dot", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const svg = parse(chartSvg(null, B3IT, DESIGN_VW));
    const dot = svg.querySelector("circle")!;
    const label = cpLabels(svg)[0];
    expect(label.textContent).toBe("TV 0.535");
    const above = +attr(dot, "cy") - +attr(label, "y");
    expect(above, "label should clear the dot by a few px").toBeGreaterThan(4);
    expect(above).toBeLessThan(20);
  });

  test("a change past the last sample keeps the lane-top fallback", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const svg = parse(chartSvg(null, B3IT_LATE_CHANGE, DESIGN_VW));
    expect(+attr(svg.querySelector("circle")!, "cy")).toBeCloseTo(TOP2 - 4, 0);
  });

  test("two changes a day apart do not print over each other", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const svg = parse(chartSvg(null, B3IT_ADJACENT, DESIGN_VW));
    const labels = cpLabels(svg);
    expect(labels.length).toBe(2);
    const box = (t: Element): [number, number, number] => {
      const half = ((t.textContent ?? "").length * 6.4) / 2;
      return [+attr(t, "x") - half, +attr(t, "x") + half, +attr(t, "y")];
    };
    const [a, b] = labels.map(box);
    const overlapX = a[0] < b[1] && b[0] < a[1];
    expect(overlapX && Math.abs(a[2] - b[2]) < 14, "labels overprint").toBe(false);
  });

  test("a dropped label still leaves the day marked", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const svg = parse(chartSvg(null, B3IT_ADJACENT, DESIGN_VW));
    const rules = [...svg.querySelectorAll("line")].filter((l) => attr(l, "stroke-dasharray"));
    expect(rules.length).toBe(B3IT_ADJACENT.changes.length);
    expect(svg.querySelectorAll("circle").length).toBe(B3IT_ADJACENT.changes.length);
  });
});
