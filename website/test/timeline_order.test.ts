/**
 * Group order on the shared timeline. The rows arrive already sorted by
 * timeline.py (freshest first); the banners over them are ordered here, and a
 * group is only as stale as its liveliest endpoint.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, expect, test } from "bun:test";

import type { TimelineEndpoint } from "../src/timeline";

beforeAll(() => GlobalRegistrator.register());
afterAll(() => GlobalRegistrator.unregister());

const ep = (base: string, name: string, lastQuery: string | null, nChanges: number): TimelineEndpoint => ({
  slug: `${base}-${name}`,
  provider: `${base}/${name}`,
  base,
  providerSlug: base,
  model: "m/a",
  modelSlug: "m2fa",
  methods: ["lt"],
  first: "2026-01-01",
  last: "2026-06-30",
  last_query: lastQuery,
  n_changes: nChanges,
  lt: { drift: [["2026-01-01", 0.1], ["2026-06-30", 0.2]], breaks: [], changes: [] },
  b3it: null,
  status: { lt: "tracked", bi: "pending", headline: "tracked", reason: "" },
});

async function groupOrder(endpoints: TimelineEndpoint[]): Promise<string[]> {
  const { renderTimeline } = await import("../src/timeline");
  document.body.innerHTML = `<div><div id="panel"></div></div>`;
  const panel = document.getElementById("panel")!;
  renderTimeline(
    panel,
    { date_min: "2026-01-01", date_max: "2026-06-30", changes: [], endpoints },
    {
      name: (e) => e.provider,
      changeName: (c) => c.provider,
      group: (e) => ({ key: e.base, label: e.base, href: "#", page: "provider page" }),
    }
  );
  return [...panel.querySelectorAll(".grp-h a")].map((a) => a.textContent ?? "");
}

test("the group with the most recently queried endpoint comes first", async () => {
  const order = await groupOrder([
    ep("stale", "a", "2026-01-20", 9),
    ep("stale", "b", "2026-01-20", 9),
    ep("fresh", "a", "2026-06-30", 0),
    ep("fresh", "b", "2026-06-30", 0),
  ]);
  expect(order).toEqual(["fresh", "stale"]);
});

test("one retired sibling does not sink a group that is otherwise alive", async () => {
  const order = await groupOrder([
    ep("mixed", "live", "2026-06-30", 0),
    ep("mixed", "dead", "2026-01-01", 0),
    ep("older", "a", "2026-05-01", 5),
    ep("older", "b", "2026-05-01", 5),
  ]);
  expect(order).toEqual(["mixed", "older"]);
});

test("groups tied on freshness keep the most-changed first", async () => {
  const order = await groupOrder([
    ep("quiet", "a", "2026-06-30", 0),
    ep("quiet", "b", "2026-06-30", 0),
    ep("moved", "a", "2026-06-30", 3),
    ep("moved", "b", "2026-06-30", 1),
  ]);
  expect(order).toEqual(["moved", "quiet"]);
});

test("a group that never answered sorts after every group that did", async () => {
  const order = await groupOrder([
    ep("never", "a", null, 0),
    ep("never", "b", null, 0),
    ep("ancient", "a", "2025-01-01", 0),
    ep("ancient", "b", "2025-01-01", 0),
  ]);
  expect(order).toEqual(["ancient", "never"]);
});
