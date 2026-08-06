/**
 * fmtCost is the browser twin of util.format_cost (Python), which formats the
 * server-rendered prices. Same cases as tests/test_util_format_cost.py: the two
 * must not drift, or the same number reads differently on two pages.
 */
import { describe, expect, test } from "bun:test";
import { fmtCost } from "../src/components";

/** Significant digits of a formatted amount, trailing zeros included. */
function sigDigits(s: string): string {
  return s.replace("-", "").replace(".", "").replace(/^0+/, "");
}

describe("fmtCost", () => {
  test("shows two decimals from $0.10 up", () => {
    expect(fmtCost(12.3456)).toBe("12.35");
    expect(fmtCost(0)).toBe("0.00");
    expect(fmtCost(1)).toBe("1.00");
    expect(fmtCost(1234.5)).toBe("1234.50");
    expect(fmtCost(0.1)).toBe("0.10");
    expect(fmtCost(0.567)).toBe("0.57");
  });

  test("grows the precision when two decimals would drop below two sig digits", () => {
    expect(fmtCost(0.0567)).toBe("0.057");
    expect(fmtCost(0.012)).toBe("0.012");
    expect(fmtCost(0.01)).toBe("0.010");
    expect(fmtCost(0.0049)).toBe("0.0049");
    expect(fmtCost(0.001)).toBe("0.0010");
    expect(fmtCost(0.0000123)).toBe("0.000012");
    expect(fmtCost(1e-9)).toBe("0.0000000010");
    expect(fmtCost(-0.0000123)).toBe("-0.000012");
  });

  test("keeps at least two significant digits for any nonzero cost", () => {
    for (const v of [1e-12, 3e-7, 0.00449, 0.0051, 0.0567, 0.099, 0.1, 7, 1e6]) {
      expect(parseFloat(fmtCost(v))).not.toBe(0);
      expect(sigDigits(fmtCost(v)).length).toBeGreaterThanOrEqual(2);
    }
  });
});
