import { describe, expect, test } from "bun:test";
import { sparkline } from "../src/components";

const ys = (svg: string): number[] =>
  [...svg.match(/<path d="M[^"]*" fill="none"/)![0].matchAll(/[ML][\d.]+ ([\d.]+)/g)].map((m) => +m[1]);

describe("sparkline y scale", () => {
  test("values under the cap use the shared cap axis", () => {
    const [lo, hi] = ys(sparkline([0, 0.75], 1.5, "c", null));
    expect(lo).toBe(31);
    expect(hi).toBe(17);
  });
  test("a series above the cap stretches its own axis instead of clipping", () => {
    const y = ys(sparkline([2.4, 5.9, 2.4], 1.5, "c", null));
    expect(y[1]).toBe(3);
    expect(y[0]).toBeGreaterThan(y[1]);
    expect(y[0]).toBe(y[2]);
  });
});
