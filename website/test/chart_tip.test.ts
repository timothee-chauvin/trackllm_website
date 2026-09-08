/**
 * chart_tip.placeTip: the pointer sits at the tip's top-right corner, so a sweep
 * along a curve never enters the tip and a move down-left lands in it; the tip
 * flips to the other side only where that side has no room, and stays in the
 * viewport rather than the chart box.
 *
 * happy-dom has no layout engine: the tip's size and the box's rect are stubbed.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, describe, expect, test } from "bun:test";

const BOX = { left: 100, top: 300, width: 800, height: 400 };
const TIP = { w: 320, h: 250 };

let placeTip: typeof import("../src/chart_tip").placeTip;
let tip: HTMLElement;
let wrap: HTMLElement;

beforeAll(async () => {
  GlobalRegistrator.register();
  Object.defineProperty(window, "innerHeight", { value: 900, configurable: true });
  ({ placeTip } = await import("../src/chart_tip"));
  wrap = document.createElement("div");
  wrap.getBoundingClientRect = () =>
    ({ ...BOX, right: BOX.left + BOX.width, bottom: BOX.top + BOX.height }) as DOMRect;
  tip = document.createElement("div");
  Object.defineProperty(tip, "offsetWidth", { value: TIP.w });
  Object.defineProperty(tip, "offsetHeight", { value: TIP.h });
});
afterAll(() => GlobalRegistrator.unregister());

const at = (clientX: number, clientY: number): [number, number] => {
  placeTip(tip, wrap, { clientX, clientY } as PointerEvent);
  return [parseFloat(tip.style.left), parseFloat(tip.style.top)];
};

describe("placeTip", () => {
  test("hangs below and to the left of the pointer, one pixel clear of it", () => {
    const [left, top] = at(600, 400);
    const px = 600 - BOX.left, py = 400 - BOX.top;
    expect(left + TIP.w).toBe(px - 1);
    expect(top).toBe(py + 1);
  });
  test("flips to the right of the pointer near the box's left edge", () => {
    const [left] = at(BOX.left + 40, 400);
    expect(left).toBe(40 + 1);
  });
  test("rises above the pointer when it would run off the bottom of the viewport", () => {
    const [, top] = at(600, 850);
    expect(top + TIP.h).toBe(850 - BOX.top - 1);
  });
  test("never starts above the viewport", () => {
    // a viewport shorter than the tip: neither side fits, so it pins to the top
    Object.defineProperty(window, "innerHeight", { value: 200, configurable: true });
    const [, top] = at(600, 100);
    expect(top).toBe(-BOX.top + 4);
    Object.defineProperty(window, "innerHeight", { value: 900, configurable: true });
  });
});
