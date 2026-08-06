/**
 * The hero curve's changepoint rise used to be drawn straight through the lede on any
 * wide page: the rise sits at changeFrac of a full-bleed layer, which for the pinned
 * (symmetric) window is the middle of the screen -- and so is the text.
 *
 * happy-dom has no layout engine (see smoke.test.ts), so the widening is checked as the
 * pure geometry it is; the framing itself was measured in headless Chromium -- see the PR.
 */
import { describe, expect, test } from "bun:test";
import { HERO_CLEAR_GAP, heroDrawnTo, heroStretch } from "../src/hero_geom";

// A wide viewport: the text column is capped at --maxw, so the lede's right edge is a
// fixed distance right of centre however wide the window gets.
const ledeRight = (vw: number): number => vw / 2 + 60;

describe("heroStretch", () => {
  test("widens the layer until the rise clears the text", () => {
    for (const vw of [1440, 1920, 2560, 3840]) {
      const clearTo = ledeRight(vw) + HERO_CLEAR_GAP;
      const stretch = heroStretch(0.5, vw, clearTo);
      expect(stretch, `${vw}px: layer not widened`).toBeGreaterThan(1);
      expect(0.5 * stretch * vw, `${vw}px: rise still over the text`).toBeCloseTo(clearTo, 6);
    }
  });

  test("leaves the layer alone when the rise already clears the text", () => {
    expect(heroStretch(0.9, 1920, ledeRight(1920) + HERO_CLEAR_GAP)).toBe(1);
  });

  test("gives up rather than hide the curve's tail", () => {
    // an early changepoint would need a layer several screens wide
    expect(heroStretch(0.15, 1920, ledeRight(1920) + HERO_CLEAR_GAP)).toBe(1);
  });

  test("survives a degenerate hero", () => {
    expect(heroStretch(0, 1920, 1068)).toBe(1);
    expect(heroStretch(0.5, 0, 1068)).toBe(1);
  });
});

describe("heroDrawnTo", () => {
  // the pinned window as of this writing: 87 daily points, changepoint in the middle
  const START = "2025-11-29", END = "2026-02-23";

  test("names the last day the widened layer still leaves on screen", () => {
    expect(heroDrawnTo(START, END, 1.2)).toBe("2026-02-09"); // 86 days / 1.2, from the start
  });

  test("names the window's own end when nothing is clipped", () => {
    expect(heroDrawnTo(START, END, 1)).toBe(END);
  });
});
