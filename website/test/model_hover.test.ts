/**
 * The model page's shared readout: one pointer position, every provider's value on
 * that day at once.
 *
 * The pure halves (which values a day reads, which day a fraction of the axis is)
 * are checked on fixtures; the wiring is checked on the generated page, with the
 * strips' client rects stubbed -- happy-dom has no layout engine, and that rect is
 * the one measurement the handler takes.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, beforeEach, describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

const SITE = resolve(import.meta.dir, "..");
const MODEL_PAGE = "models/qwen2fqwen3-coder.html";
const VW = 1000; // the strips' viewBox width, and the width they are stubbed at

function requireBuilt(path: string): string {
  const full = join(SITE, path);
  if (!existsSync(full)) {
    throw new Error(`${path} is missing -- run \`make build\` before these tests`);
  }
  return full;
}

beforeAll(() => GlobalRegistrator.register());
afterAll(() => GlobalRegistrator.unregister());

const EP = {
  lt: { drift: [["2026-07-17", 0.02], ["2026-07-22", 0.5], ["2026-07-26", 0.81]] as [string, number][] },
  b3it: { tv: [["2026-07-20", 0.11], ["2026-07-26", 0.53]] as [string, number][] },
};

describe("readCells", () => {
  test("a day both methods observed reads both, LT first, each in its own units", async () => {
    const { readCells } = await import("../src/model_hover");
    const cells = readCells(EP, "2026-07-26");
    expect(cells.map((c) => c.text)).toEqual(["0.81 nats", "TV 0.53"]);
    expect(cells[0].col).not.toBe(cells[1].col);
  });

  test("a day outside a series' span reads only the method that was there", async () => {
    const { readCells } = await import("../src/model_hover");
    // B3IT's first sample is the 20th: extrapolating it back to the 17th would
    // invent a level for a day it never measured
    expect(readCells(EP, "2026-07-17").map((c) => c.text)).toEqual(["0.02 nats"]);
  });

  test("a day neither method observed reads an em dash, not a stale value", async () => {
    const { readCells } = await import("../src/model_hover");
    expect(readCells(EP, "2026-08-30").map((c) => c.text)).toEqual(["—"]);
    expect(readCells({ lt: null, b3it: null }, "2026-07-26").map((c) => c.text)).toEqual(["—"]);
  });
});

describe("dayAt", () => {
  test("maps the ends of the axis to the ends of the span", async () => {
    const { dayAt } = await import("../src/model_hover");
    const d0 = Date.parse("2026-07-17T00:00:00Z");
    const d1 = Date.parse("2026-07-29T00:00:00Z");
    expect(dayAt(d0, d1, 0)).toBe("2026-07-17");
    expect(dayAt(d0, d1, 1)).toBe("2026-07-29");
  });

  test("snaps to a whole day, so every strip is read at the same instant", async () => {
    const { dayAt } = await import("../src/model_hover");
    const d0 = Date.parse("2026-07-17T00:00:00Z");
    const d1 = Date.parse("2026-07-29T00:00:00Z");
    for (const f of [0.1, 0.37, 0.5, 0.83]) {
      expect(dayAt(d0, d1, f)).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    }
    expect(dayAt(d0, d1, 0.5)).toBe("2026-07-23");
  });
});

/** Serve the page's fetches from disk (smoke.test.ts's stub). */
function stubFetch(pageDir: string): void {
  globalThis.fetch = (async (input: string | URL) => {
    const file = resolve(SITE, pageDir, String(input));
    if (!existsSync(file)) return { ok: false, status: 404, json: async () => null } as Response;
    return { ok: true, status: 200, json: async () => JSON.parse(readFileSync(file, "utf8")) } as Response;
  }) as typeof fetch;
}

const rect = (): DOMRect =>
  ({ left: 0, top: 0, width: VW, height: 40, right: VW, bottom: 40, x: 0, y: 0 }) as DOMRect;

async function mountPage(): Promise<void> {
  document.documentElement.innerHTML = readFileSync(requireBuilt(MODEL_PAGE), "utf8");
  stubFetch(dirname(MODEL_PAGE));
  await (await import("../src/model")).init();
  // the strips are all drawn on one axis at one width: stubbing them alike is what
  // the CSS grid does in a browser
  for (const el of [...document.querySelectorAll("#cmp .spark svg"), document.getElementById("cmp")!,
    document.querySelector(".chartbox")!]) {
    (el as HTMLElement).getBoundingClientRect = rect;
  }
}

/** A pointer event at an x in the strips' own units (= client px, stubbed 1:1). */
function point(el: Element, type: string, x: number, pointerType: string): void {
  el.dispatchEvent(
    new window.PointerEvent(type, { clientX: x, clientY: 20, bubbles: true, pointerType })
  );
}

const sparks = (): Element[] => [...document.querySelectorAll("#cmp .spark")];
const crossX = (): number[] =>
  [...document.querySelectorAll("#cmp .hover-mark line")].map((l) => +(l.getAttribute("x1") ?? "0"));

describe("the shared readout on the model page", () => {
  beforeEach(() => mountPage());

  test("a hover reads every strip at one date", () => {
    const tip = document.getElementById("cmptip")!;
    expect(tip.hidden, "the readout starts hidden").toBe(true);
    point(sparks()[1], "pointermove", VW * 0.6, "mouse");

    expect(tip.hidden).toBe(false);
    expect(tip.textContent).toMatch(/\d{4}-\d{2}-\d{2}/);
    // every strip -- not only the hovered one -- marks the same instant
    const xs = crossX();
    expect(xs.length).toBe(sparks().length);
    expect(new Set(xs.map((x) => x.toFixed(1))).size).toBe(1);
    expect(xs[0]).toBeGreaterThan(0);
  });

  test("every tracked row swaps its counts for the value on that day", () => {
    point(sparks()[1], "pointermove", VW * 0.6, "mouse");
    const reads = [...document.querySelectorAll("#cmp .row .meta .read")];
    expect(reads.length, "no row carries a readout").toBeGreaterThan(1);
    for (const r of reads) {
      expect((r as HTMLElement).hidden).toBe(false);
      expect(r.textContent).toMatch(/nats|TV|—/);
      const stat = r.parentElement!.querySelector(".static") as HTMLElement;
      expect(stat.hidden, "the static meta still shows under the readout").toBe(true);
    }
  });

  test("moving along the axis moves the date and the marks with it", () => {
    const tip = document.getElementById("cmptip")!;
    point(sparks()[1], "pointermove", VW * 0.2, "mouse");
    const early = tip.textContent;
    const earlyX = crossX()[0];
    point(sparks()[1], "pointermove", VW * 0.8, "mouse");
    expect(tip.textContent).not.toBe(early);
    expect(crossX()[0]).toBeGreaterThan(earlyX);
  });

  test("leaving the timeline puts every row back", () => {
    point(sparks()[1], "pointermove", VW * 0.6, "mouse");
    point(document.getElementById("cmp")!, "pointerleave", VW * 0.6, "mouse");
    expect(document.getElementById("cmptip")!.hidden).toBe(true);
    expect(crossX().length).toBe(0);
    for (const s of document.querySelectorAll("#cmp .row .meta .static")) {
      expect((s as HTMLElement).hidden).toBe(false);
    }
  });

  test("a touch pins the readout and a drag scrubs it", () => {
    const tip = document.getElementById("cmptip")!;
    point(sparks()[1], "pointerdown", VW * 0.2, "touch");
    expect(tip.hidden).toBe(false);
    const early = tip.textContent;
    // dragging across the strip reads day after day, without a second press
    point(sparks()[1], "pointermove", VW * 0.8, "touch");
    expect(tip.textContent).not.toBe(early);
    // a touch fires no pointerleave of its own, and a scroll away must not wipe a
    // reading the reader is still comparing rows against
    point(document.getElementById("cmp")!, "pointerleave", VW * 0.8, "touch");
    expect(tip.hidden).toBe(false);
    point(document.getElementById("cmp")!, "pointerdown", 5, "touch");
    expect(tip.hidden).toBe(true);
  });

  test("a mouse that never crosses a strip reads nothing", () => {
    point(document.getElementById("cmp")!, "pointermove", VW * 0.6, "mouse");
    expect(document.getElementById("cmptip")!.hidden).toBe(true);
  });

  test("crossing the seam between two rows keeps the reading up", () => {
    const tip = document.getElementById("cmptip")!;
    point(sparks()[1], "pointermove", VW * 0.6, "mouse");
    const read = tip.textContent;
    // the padding between two strips is not a reason to blink the column of values
    // the reader is running down
    point(document.querySelector("#cmp .row .pv")!, "pointermove", VW * 0.6, "mouse");
    expect(tip.hidden).toBe(false);
    expect(tip.textContent).toBe(read);
  });
});
