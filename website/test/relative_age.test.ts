/**
 * The front page is a static build, so the age of each method's last query is
 * computed in the browser from an absolute instant. These are the boundaries of
 * that formatter -- the only place the page can start lying about freshness.
 */
import { describe, expect, test } from "bun:test";
import { relativeAge } from "../src/components";

const NOW = Date.parse("2026-07-28T12:00:00Z");
const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/** The instant that is `age` ms before NOW. */
function ago(age: number): string {
  return new Date(NOW - age).toISOString();
}

describe("relativeAge", () => {
  test("shows whole minutes under an hour", () => {
    expect(relativeAge(ago(0), NOW)).toBe("0m ago");
    expect(relativeAge(ago(59_000), NOW)).toBe("0m ago");
    expect(relativeAge(ago(MINUTE), NOW)).toBe("1m ago");
    expect(relativeAge(ago(14 * MINUTE), NOW)).toBe("14m ago");
    expect(relativeAge(ago(HOUR - 1000), NOW)).toBe("59m ago");
  });

  test("shows padded hours and minutes under a day", () => {
    expect(relativeAge(ago(HOUR), NOW)).toBe("1h00m ago");
    expect(relativeAge(ago(3 * HOUR + 7 * MINUTE), NOW)).toBe("3h07m ago");
    expect(relativeAge(ago(DAY - MINUTE), NOW)).toBe("23h59m ago");
  });

  test("drops to days and hours past a day", () => {
    expect(relativeAge(ago(DAY), NOW)).toBe("1d 0h ago");
    expect(relativeAge(ago(2 * DAY + 4 * HOUR + 30 * MINUTE), NOW)).toBe("2d 4h ago");
    expect(relativeAge(ago(29 * DAY + 23 * HOUR), NOW)).toBe("29d 23h ago");
  });

  test("drops the hour past a month, where it is noise", () => {
    expect(relativeAge(ago(30 * DAY + 5 * HOUR), NOW)).toBe("30d ago");
    expect(relativeAge(ago(400 * DAY), NOW)).toBe("400d ago");
  });

  test("clamps a future timestamp to zero rather than counting up", () => {
    expect(relativeAge(ago(-5 * MINUTE), NOW)).toBe("0m ago");
    expect(relativeAge(ago(-3 * DAY), NOW)).toBe("0m ago");
  });

  test("reads both spellings of UTC identically", () => {
    expect(relativeAge("2026-07-28T11:46:00Z", NOW)).toBe("14m ago");
    expect(relativeAge("2026-07-28T11:46:00+00:00", NOW)).toBe("14m ago");
  });

  test("throws on an unparseable timestamp rather than rendering NaN", () => {
    expect(() => relativeAge("last tuesday", NOW)).toThrow();
  });
});
