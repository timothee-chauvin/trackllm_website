import { describe, expect, test } from "bun:test";
import { readingCaption } from "../src/caption";

describe("readingCaption", () => {
  test("names both methods when both lanes carry data", () => {
    const note = readingCaption(true, true);
    expect(note).toContain("both lanes share the time axis");
    expect(note).toContain("LT: σ");
    expect(note).toContain("B3IT: TV");
  });

  test("mentions only sigma on an LT-only endpoint", () => {
    const note = readingCaption(true, false);
    expect(note).not.toContain("both lanes");
    expect(note).not.toContain("B3IT");
    expect(note).not.toContain("TV");
    expect(note).toContain("σ");
  });

  test("mentions only total variation on a B3IT-only endpoint", () => {
    const note = readingCaption(false, true);
    expect(note).not.toContain("both lanes");
    expect(note).not.toContain("LT");
    expect(note).not.toContain("σ");
    expect(note).toContain("TV");
  });

  test("every variant keeps the shared reading advice", () => {
    for (const note of [
      readingCaption(true, true),
      readingCaption(true, false),
      readingCaption(false, true),
    ]) {
      expect(note).toStartWith("<b>Reading it:</b>");
      expect(note).toContain("step up that persists");
      expect(note).toContain("dashed lines mark detected changepoints");
    }
  });
});
