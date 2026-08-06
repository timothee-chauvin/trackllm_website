/**
 * fmtCost is the browser twin of util.format_cost (Python), which formats the
 * server-rendered prices. Same cases as tests/test_util_format_cost.py: the two
 * must not drift, or the same number reads differently on two pages.
 */
import { describe, expect, test } from "bun:test";
import { fmtCost } from "../src/components";

describe("fmtCost", () => {
  test("shows two decimals for ordinary amounts", () => {
    expect(fmtCost(12.3456)).toBe("12.35");
    expect(fmtCost(0)).toBe("0.00");
    expect(fmtCost(1)).toBe("1.00");
    expect(fmtCost(1234.5)).toBe("1234.50");
    expect(fmtCost(0.005)).toBe("0.01");
    expect(fmtCost(0.01)).toBe("0.01");
    expect(fmtCost(0.019)).toBe("0.02");
  });

  test("falls back to two significant digits below the two-decimal floor", () => {
    expect(fmtCost(0.0049)).toBe("0.0049");
    expect(fmtCost(0.001)).toBe("0.0010");
    expect(fmtCost(0.0000123)).toBe("0.000012");
    expect(fmtCost(1e-9)).toBe("0.0000000010");
    expect(fmtCost(-0.0000123)).toBe("-0.000012");
  });

  test("never renders a nonzero cost as zero", () => {
    for (const v of [1e-12, 3e-7, 0.00449, 0.004999, 0.0051]) {
      expect(parseFloat(fmtCost(v))).not.toBe(0);
    }
  });
});
