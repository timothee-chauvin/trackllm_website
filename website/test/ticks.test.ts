/**
 * timeTicks picks the granularity; the caller only says how many labels fit.
 *
 * The bug it exists for: monthTicks returned nothing at all for an endpoint whose
 * whole observed span sits inside one month (2026-07-17 .. 2026-07-29), leaving the
 * chart with no gridlines and a blank axis.
 */
import { describe, expect, test } from "bun:test";
import { timeTicks } from "../src/components";

const DAY = 24 * 3600 * 1000;
const at = (s: string): number => Date.parse(s + "T00:00:00Z");

const DAY_LABEL = /^[A-Z][a-z]{2} \d{1,2}$/; // "Jul 18"
const MONTH_LABEL = /^[A-Z][a-z]{2} '\d{2}$/; // "Jul '26"

/** Every result, whatever the granularity, owes the caller these. */
function expectWellFormed(ticks: { t: number; label: string }[], d0: number, d1: number, max: number): void {
  expect(ticks.length).toBeGreaterThan(0);
  expect(ticks.length).toBeLessThanOrEqual(max);
  for (const { t } of ticks) {
    expect(t).toBeGreaterThanOrEqual(d0);
    expect(t).toBeLessThanOrEqual(d1);
  }
  const ts = ticks.map((k) => k.t);
  expect(ts).toEqual([...ts].sort((a, b) => a - b));
  expect(new Set(ts).size).toBe(ts.length);
}

describe("timeTicks", () => {
  test("dates a span too short to contain a first-of-month", () => {
    const d0 = at("2026-07-17"), d1 = at("2026-07-29");
    const ticks = timeTicks(d0, d1, 18);
    expectWellFormed(ticks, d0, d1, 18);
    expect(ticks.length).toBeGreaterThanOrEqual(3);
    for (const { label } of ticks) expect(label).toMatch(DAY_LABEL);
  });

  test("thins to whole days that still fit a phone's budget", () => {
    const d0 = at("2026-07-17"), d1 = at("2026-07-29");
    const ticks = timeTicks(d0, d1, 3);
    expectWellFormed(ticks, d0, d1, 3);
    for (const { label } of ticks) expect(label).toMatch(DAY_LABEL);
  });

  test("a quarter fits within the budget", () => {
    const d0 = at("2026-04-01"), d1 = at("2026-07-01");
    const ticks = timeTicks(d0, d1, 18);
    expectWellFormed(ticks, d0, d1, 18);
    expect(ticks.length).toBeGreaterThanOrEqual(3);
  });

  test("a span of months labels as months even when days would fit", () => {
    // 14-day ticks fit a 930px plot seven times over, but "Jul 3 .. Jan 15" spends
    // sixteen labels saying less than seven monthly ones -- and drops the year at
    // exactly the point the reader needs it.
    const d0 = at("2025-07-03"), d1 = at("2026-02-01");
    const ticks = timeTicks(d0, d1, 18);
    expectWellFormed(ticks, d0, d1, 18);
    for (const { label } of ticks) expect(label).toMatch(MONTH_LABEL);
  });

  test("two years label as months, never as bare days", () => {
    const d0 = at("2024-07-17"), d1 = at("2026-07-29");
    const ticks = timeTicks(d0, d1, 18);
    expectWellFormed(ticks, d0, d1, 18);
    for (const { label } of ticks) expect(label).toMatch(MONTH_LABEL);
  });

  test("a decade stays inside the budget", () => {
    const d0 = at("2016-01-01"), d1 = at("2026-01-01");
    const ticks = timeTicks(d0, d1, 6);
    expectWellFormed(ticks, d0, d1, 6);
  });

  test("a single day still gets a tick", () => {
    const d0 = at("2026-07-17");
    const ticks = timeTicks(d0, d0, 18);
    expectWellFormed(ticks, d0, d0, 18);
    expect(ticks[0].label).toMatch(DAY_LABEL);
  });

  test("two days get one tick each", () => {
    const d0 = at("2026-07-17"), d1 = d0 + DAY;
    expect(timeTicks(d0, d1, 18).length).toBe(2);
  });

  test("month labels carry the year, day labels the day of month", () => {
    expect(timeTicks(at("2024-01-01"), at("2026-07-29"), 18)[0].label).toMatch(/'2[456]$/);
    expect(timeTicks(at("2026-07-17"), at("2026-07-29"), 18)[0].label).not.toContain("'");
  });
});
