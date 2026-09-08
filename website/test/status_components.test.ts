import { describe, expect, test } from "bun:test";

import {
  HEADLINE_ORDER,
  headlineBadge,
  highlight,
  monitoredSpan,
  statusPill,
  statusRank,
  untrackedDirCells,
} from "../src/components";

describe("highlight", () => {
  test("wraps the matched substring in <mark>", () => {
    expect(highlight("gpt-5.2", "gpt-5")).toBe("<mark>gpt-5</mark>.2");
  });

  test("matches case-insensitively but keeps the original casing", () => {
    expect(highlight("Alibaba", "alibaba")).toBe("<mark>Alibaba</mark>");
  });

  test("escapes HTML in all three segments", () => {
    expect(highlight("a<b>&c", "<b>")).toBe("a<mark>&lt;b&gt;</mark>&amp;c");
  });

  test("no match and empty query just escape", () => {
    expect(highlight("a<b", "zzz")).toBe("a&lt;b");
    expect(highlight("a<b", "")).toBe("a&lt;b");
  });
});

describe("headlineBadge", () => {
  test("dashes the class, spaces the label", () => {
    expect(headlineBadge("too_expensive")).toBe(
      '<span class="badge st st-too-expensive" role="button" tabindex="0" ' +
        'data-tip="This endpoint costs more than our tracking budget allows.">too expensive</span>',
    );
  });

  test("tracked stays as is", () => {
    expect(headlineBadge("tracked")).toBe(
      '<span class="badge st st-tracked" role="button" tabindex="0" ' +
        'data-tip="This endpoint is actively tracked.">tracked</span>',
    );
  });

  test("every headline has a popover", () => {
    for (const h of HEADLINE_ORDER) expect(headlineBadge(h)).toContain("data-tip=");
  });
});

describe("statusRank", () => {
  test("tracked trace-statuses come before every headline group", () => {
    const tracked = statusRank({ methods: ["lt"], status: "retired", headline: "retired" });
    const untracked = statusRank({ methods: [], status: null, headline: "tracked" });
    expect(tracked).toBeLessThan(untracked);
  });

  test("untracked rows follow the headline priority order", () => {
    const ranks = HEADLINE_ORDER.map((h) =>
      statusRank({ methods: [], status: null, headline: h }),
    );
    expect(ranks).toEqual([...ranks].sort((a, b) => a - b));
  });
});

describe("untrackedDirCells", () => {
  test("wears one badge per headline, in the order given", () => {
    const html = untrackedDirCells(
      { slug: "s", headlines: ["retired", "too_expensive"], reason: "why" },
      "",
    );
    const badges = [...html.matchAll(/class="badge st st-([a-z-]+)"/g)].map((m) => m[1]);
    expect(badges).toEqual(["retired", "too-expensive"]);
    expect(html).toContain('title="why"');
  });
});

describe("statusPill", () => {
  test("a retired pill's popover carries the row's reason; a live one does not", () => {
    const retired = statusPill("retired", "Monitoring was retired: too pricey.");
    expect(retired).toContain("gone quiet");
    expect(retired).toContain("Monitoring was retired: too pricey.");
    const stable = statusPill("stable", "This endpoint is actively tracked.");
    expect(stable).not.toContain("actively tracked.");
  });
});

describe("monitoredSpan", () => {
  test("open while anything is still monitored, a closed range once nothing is", () => {
    expect(monitoredSpan("2026-01-15", "2026-06-30", true)).toBe("since Jan 2026");
    expect(monitoredSpan("2026-01-15", "2026-06-30", false)).toBe("Jan 2026 – Jun 2026");
  });
});
