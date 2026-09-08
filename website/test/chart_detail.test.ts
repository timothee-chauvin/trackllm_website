/**
 * The endpoint chart's readouts beyond the value: the raw votes and logprobs under a
 * day, what a change mark was called on, what an epoch rule marks -- and the epoch
 * rules themselves, which the TV line must not cross.
 *
 * happy-dom has no layout engine, so the chart's client rect is stubbed to the size
 * it was drawn at (as hover.test does); everything downstream of it is real.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import type { B3ITRaw } from "../src/chart_detail";

const DESIGN_VW = 1000;
const VH = 324;
const PL = 50, PR = 20;
const PW = DESIGN_VW - PL - PR;
const LANE_H = 108, TOP1 = 34, GAP = 34;
const TOP2 = TOP1 + LANE_H + GAP;

const day = (d: number, hh = "21"): string => `2026-07-${String(d).padStart(2, "0")}T${hh}:43:00+00:00`;

/** Two epochs: 17..22 against the first reference, then a re-initialisation on the
 *  22nd (after a change called on the 21st) and 23..29 against a new one. The batch
 *  of the 22nd closed epoch 1 and shares the instant of epoch 2's start. */
const DAILY: [string, number][] = [
  [day(17), 0.1], [day(18), 0.12], [day(19), 0.11], [day(20), 0.11], [day(21), 0.52], [day(22, "04"), 0.55],
  [day(23), 0.02], [day(24), 0.03], [day(25), 0.02], [day(26), 0.04], [day(27), 0.03], [day(28), 0.02], [day(29), 0.03],
];
const EPOCHS = [
  { start: "2026-07-16T04:00:00+00:00", end: day(22, "04"), endReason: "change_detected", changeDate: day(21), detector: "scan", nRef: 2 },
  { start: day(22, "04"), end: null, endReason: null, changeDate: null, detector: null, nRef: 2 },
];
const B3IT = {
  tv: DAILY,
  breaks: [6],
  daily: DAILY,
  changes: [{ date: "2026-07-21", shiftTV: 0.4, detector: "scan" }],
  epochs: EPOCHS,
  firstDate: "2026-07-17",
  lastDate: "2026-07-29",
};
const VOTES: B3ITRaw = {
  bis: ["What is the capital of <b>Assyria</b>? Answer in one word, please.", "Pick a number"],
  reference: [
    [[0, { Nineveh: 31, Assur: 19 }], [1, { "7": 30, "3": 20 }]],
    [[0, { Assur: 40, Nineveh: 10 }], [1, { "7": 25, "3": 25 }]],
  ],
  batches: DAILY.map((_, i) => [
    [0, { Nineveh: 7, Assur: 3 }, 0.08],
    [1, { "3": 6, "7": 4 }, i === 0 ? null : 0.2],
  ]),
};

const LT_DAILY: [string, number][] = Array.from({ length: 13 }, (_, i) => [
  `2026-07-${17 + i}`,
  i < 6 ? 0.1 : 0.9,
]);
const LT = {
  drift: LT_DAILY,
  breaks: [],
  daily: LT_DAILY,
  changes: [{ date: "2026-07-23", shift: 0.8 }],
  firstDate: "2026-07-17",
  lastDate: "2026-07-29",
};
const LOGPROBS = {
  drift_prompt: 0,
  prompts: [
    {
      text: "Hi",
      tokens: ["Hello", " there", "\n"],
      ref: [-0.1, -2.5, null],
      floor: -9.5,
      days: [["2026-07-23", 24, [[0, -0.15], [1, -2.2], [2, -4.4]]] as [string, number, [number, number][]]],
    },
    { text: "x", tokens: ["<x>"], ref: [-0.3], floor: -9.5, days: [] as [string, number, [number, number][]][] },
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
const dayX = (d: number): number => PL + ((d - 16) / 13) * PW;
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

describe("epoch boundaries", () => {
  test("a run ends at the batch that closed the epoch, even with no missing day", async () => {
    const { downsampleRuns } = await import("../src/chart_geom");
    const { breaks } = downsampleRuns(DAILY, 110, EPOCHS.map((e) => Date.parse(e.start)));
    expect(breaks).toEqual([6]);
  });

  test("epochOf scores the closing batch in the epoch it closed", async () => {
    const { epochOf } = await import("../src/chart_geom");
    expect(epochOf(EPOCHS, day(22, "04"))).toBe(0);
    expect(epochOf(EPOCHS, day(23))).toBe(1);
    expect(epochOf(EPOCHS, day(17))).toBe(0);
  });

  test("the TV line restarts at the re-initialisation instead of joining the epochs", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const svg = parse(chartSvg(null, B3IT, DESIGN_VW));
    const line = [...svg.querySelectorAll("path")].find(
      (p) => p.getAttribute("fill") === "none" && p.getAttribute("stroke") === "var(--b3it)"
    )!;
    expect(line.getAttribute("d")!.match(/M/g)?.length).toBe(2);
  });

  test("a full-height rule marks every epoch start, the initialisation included", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const svg = parse(chartSvg(null, B3IT, DESIGN_VW));
    const rules = [...svg.querySelectorAll("line")].filter((l) => attr(l, "stroke") === "var(--text)");
    expect(rules.length).toBe(2);
    for (const r of rules) {
      expect(+attr(r, "y1")).toBe(TOP2);
      expect(+attr(r, "y2")).toBe(TOP2 + LANE_H);
    }
    expect(+attr(rules[0], "x1")).toBeCloseTo(dayX(16), 0);
    expect(+attr(rules[1], "x1")).toBeCloseTo(dayX(22), 0);
    expect(svg.querySelectorAll(".epoch-hit").length).toBe(2);
    expect(svg.querySelectorAll(".cp-hit").length).toBe(1);
  });
});

describe("detail builders", () => {
  test("countsText puts the most voted token first even when tokens are digits", async () => {
    const { countsText } = await import("../src/chart_detail");
    expect(countsText({ "3": 6, "7": 4 })).toBe("3 6 · 7 4");
    expect(countsText({ a: 1, b: 5, c: 3, d: 2 })).toBe("b 5 · c 3 · d 2 · +1");
    expect(countsText({ " Yes": 2 })).toBe("&quot; Yes&quot; 2");
  });

  test("a B3IT day lists each border input's votes beside the reference's, with its TV", async () => {
    const { b3itDayDetail } = await import("../src/chart_detail");
    const html = parse(b3itDayDetail(B3IT, VOTES, 7)); // 2026-07-24, epoch 2
    const rows = [...html.querySelectorAll("tr")].slice(1).map((r) => [...r.children].map((c) => c.textContent));
    expect(rows).toEqual([
      ["What is the capital of <b>Assyria</b>? …", "Nineveh 7 · Assur 3", "Assur 40 · Nineveh 10", "0.08"],
      ["Pick a number", "3 6 · 7 4", "3 25 · 7 25", "0.20"],
    ]);
    expect(html.querySelector("td.p")!.getAttribute("title")).toBe(VOTES.bis[0]);
    expect(html.querySelector("b"), "prompt markup was rendered, not escaped").toBeNull();
    expect(html.textContent).toContain("Batch at 21:43 UTC");
  });

  test("an LT day tabulates the day's mean logprobs against the reference's, per prompt", async () => {
    const { ltDayDetail } = await import("../src/chart_detail");
    const html = parse(ltDayDetail(LOGPROBS, "2026-07-23"));
    const heads = [...html.querySelectorAll(".tip-h")].map((h) => h.textContent);
    expect(heads).toEqual(["Hi (the drift prompt) · 24 queries", "x · no queries this day"]);
    const rows = [...html.querySelectorAll("tr")].slice(1).map((r) => [...r.children].map((c) => c.textContent));
    expect(rows).toEqual([
      ["Hello", "-0.150", "-0.100"],
      ['" there"', "-2.200", "-2.500"],
      ["⏎", "-4.400", "-9.500*"], // never in the reference: scored at the floor, and said so
    ]);
    expect(html.textContent).toContain("* never returned in the reference period");
  });

  test("an LT change quotes the gate's shift and the two 7-day means behind it", async () => {
    const { ltChangeDetail } = await import("../src/chart_detail");
    const text = parse(ltChangeDetail(LT, LT.changes[0])).textContent!;
    expect(text).toContain("Level shift 0.8 nats");
    expect(text).toContain("mean drift 0.9 nats over the 7 days after vs 0.1 over the 6 days before");
  });

  test("a B3IT change names the split's levels within its epoch and the detector", async () => {
    const { b3itChangeDetail } = await import("../src/chart_detail");
    const text = parse(b3itChangeDetail(B3IT, B3IT.changes[0])).textContent!;
    expect(text).toContain("TV shift 0.4");
    expect(text).toContain("mean TV 0.535 after vs 0.11 before the split");
    expect(text).toContain("epoch 1");
    expect(text).toContain("scan permutation test");
  });

  test("a change in a fresh epoch says so instead of inventing a level before it", async () => {
    const { b3itChangeDetail } = await import("../src/chart_detail");
    const text = parse(
      b3itChangeDetail(B3IT, { date: "2026-07-23", shiftTV: 0.02, detector: "adaptive" })
    ).textContent!;
    expect(text).toContain("over the first days of epoch 2");
    expect(text).toContain("adaptive rule");
  });

  test("an epoch rule explains the (re-)initialisation and lists the reference votes", async () => {
    const { epochDetail } = await import("../src/chart_detail");
    const first = parse(epochDetail(B3IT, VOTES, 0)).textContent!;
    expect(first).toContain("Initialised");
    expect(first).toContain("Nineveh 31 · Assur 19");
    const second = parse(epochDetail(B3IT, VOTES, 1)).textContent!;
    expect(second).toContain("Re-initialised after the change detected on 2026-07-21");
    expect(second).toContain("TV restarts from 0");
    expect(second).toContain("Assur 40 · Nineveh 10");
    // before the votes arrive the count is known, the rows are not
    const pending = parse(epochDetail(B3IT, null, 1));
    expect(pending.textContent).toContain("2 border inputs");
    expect(pending.querySelectorAll("tr").length).toBe(0);
  });
});

describe("readout wiring", () => {
  async function mount(
    lt: typeof LT | null,
    b3it: typeof B3IT | null
  ): Promise<{ chart: HTMLElement; tip: HTMLElement }> {
    const { chartSvg } = await import("../src/endpoint");
    const { bindHover } = await import("../src/chart_hover");
    document.body.innerHTML = `<div class="chartwrap chart" id="mainchart"></div>
      <div class="chart-tip" id="charttip" hidden></div>`;
    const chart = document.getElementById("mainchart")!;
    const tip = document.getElementById("charttip")!;
    chart.innerHTML = chartSvg(lt, b3it, DESIGN_VW);
    const rect = { left: 0, top: 0, width: DESIGN_VW, height: VH, right: DESIGN_VW, bottom: VH, x: 0, y: 0 } as DOMRect;
    chart.querySelector("svg")!.getBoundingClientRect = (): DOMRect => rect;
    chart.getBoundingClientRect = (): DOMRect => rect;
    bindHover(chart, tip, lt, b3it, () => DESIGN_VW, {
      lt: () => Promise.resolve(LOGPROBS),
      b3it: () => Promise.resolve(VOTES),
    });
    return { chart, tip };
  }
  const point = (el: Element, type: string, x: number, pointerType: string): void => {
    el.dispatchEvent(new window.PointerEvent(type, { clientX: x, clientY: 100, bubbles: true, pointerType }));
  };

  test("a change mark reads as the change, not the nearest day", async () => {
    const { chart, tip } = await mount(null, B3IT);
    point(chart.querySelector(".cp-hit")!, "pointermove", dayX(21), "mouse");
    expect(tip.textContent).toContain("Change detected");
    expect(tip.textContent).toContain("2026-07-21");
    expect(tip.textContent).toContain("TV shift 0.4");
    expect(tip.classList.contains("has-detail")).toBe(true);
  });

  test("an epoch rule reads as the epoch, and its votes fill in when loaded", async () => {
    const { chart, tip } = await mount(null, B3IT);
    point(chart.querySelectorAll(".epoch-hit")[1], "pointermove", dayX(22), "mouse");
    expect(tip.textContent).toContain("Epoch 2");
    await tick();
    expect(tip.textContent).toContain("Assur 40 · Nineveh 10");
  });

  test("a day on the LT lane shows its value at once and the logprobs once loaded", async () => {
    const { chart, tip } = await mount(LT, B3IT);
    const lane = [...chart.querySelectorAll(".lane-hit")].find((h) => attr(h, "data-lane") === "lt")!;
    point(lane, "pointermove", dayX(23), "mouse");
    expect(tip.textContent).toContain("0.9 nats");
    await tick();
    expect(tip.textContent).toContain("Hello");
    expect(tip.textContent).toContain("24 queries");
  });

  test("a late-loaded detail does not land on a readout that has moved on", async () => {
    const { chart, tip } = await mount(LT, B3IT);
    const lane = [...chart.querySelectorAll(".lane-hit")].find((h) => attr(h, "data-lane") === "lt")!;
    point(lane, "pointermove", dayX(23), "mouse");
    point(chart, "pointerleave", 0, "mouse");
    await tick();
    expect(tip.hidden).toBe(true);
    expect(tip.textContent).not.toContain("Hello");
  });

  test("a tap pins the readout with its detail scrollable", async () => {
    const { chart, tip } = await mount(null, B3IT);
    point(chart.querySelectorAll(".lane-hit")[0], "pointerdown", dayX(24), "touch");
    await tick();
    expect(tip.hidden).toBe(false);
    expect(tip.classList.contains("pinned")).toBe(true);
    expect(tip.textContent).toContain("Nineveh 7 · Assur 3");
    point(chart, "pointerdown", 5, "touch");
    expect(tip.hidden).toBe(true);
  });
});

describe("status card", () => {
  const card = async (state: "stable" | "changed" | "retired"): Promise<string> => {
    const { renderStatusCard } = await import("../src/endpoint");
    document.body.innerHTML = `<div id="statuscard"></div>`;
    renderStatusCard(LT, B3IT, state);
    return document.getElementById("statuscard")!.textContent!;
  };

  test("an endpoint still monitored is monitored since, not from-to", async () => {
    expect(await card("stable")).toContain("Since Jul 2026");
    expect(await card("changed")).not.toContain("–");
  });

  test("a retired endpoint keeps its closed span", async () => {
    expect(await card("retired")).toContain("Jul 2026 – Jul 2026");
  });
});
