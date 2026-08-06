/**
 * Missing days are read off the fill: the area under a curve covers the days the
 * endpoint was actually observed on and nothing else. Both plots that draw a daily
 * series -- the endpoint page's lanes and the shared timeline's strips -- have to
 * agree on that, so both are checked here.
 *
 * happy-dom has no layout engine, so these assert the path coordinates, not looks.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, describe, expect, test } from "bun:test";

beforeAll(() => GlobalRegistrator.register());
afterAll(() => GlobalRegistrator.unregister());

const days = (start: string, n: number): string[] =>
  Array.from({ length: n }, (_, i) =>
    new Date(Date.parse(start + "T00:00:00Z") + i * 86400000).toISOString().slice(0, 10)
  );
const series = (dates: string[]): [string, number][] => dates.map((d, i) => [d, i * 0.01]);

const parse = (markup: string): Element => {
  const host = document.createElement("div");
  host.innerHTML = markup;
  return host.firstElementChild!;
};
/** The x of every point of an SVG path's `d`, in draw order. */
const xs = (d: string): number[] =>
  [...d.matchAll(/[ML]\s*(-?[\d.]+)/g)].map((m) => +m[1]);

describe("downsampleRuns", () => {
  test("a contiguous series is one run", async () => {
    const { downsampleRuns } = await import("../src/chart_geom");
    const pairs = series(days("2026-01-01", 10));
    expect(downsampleRuns(pairs, 110)).toEqual({ series: pairs, breaks: [] });
  });

  test("a missing day breaks the series where the hole is", async () => {
    const { downsampleRuns } = await import("../src/chart_geom");
    const pairs = series([...days("2026-01-01", 3), ...days("2026-01-05", 3)]);
    expect(downsampleRuns(pairs, 110)).toEqual({ series: pairs, breaks: [3] });
  });

  test("thinning keeps both ends of every run", async () => {
    const { downsampleRuns } = await import("../src/chart_geom");
    const left = series(days("2026-01-01", 200));
    const right = series(days("2026-09-01", 200));
    const { series: kept, breaks } = downsampleRuns([...left, ...right], 110);
    expect(breaks.length).toBe(1);
    expect(kept.length).toBeLessThanOrEqual(110);
    expect(kept[0]).toEqual(left[0]);
    expect(kept[breaks[0] - 1]).toEqual(left[left.length - 1]);
    expect(kept[breaks[0]]).toEqual(right[0]);
    expect(kept[kept.length - 1]).toEqual(right[right.length - 1]);
  });

  test("an empty series has nothing to break", async () => {
    const { downsampleRuns } = await import("../src/chart_geom");
    expect(downsampleRuns([], 110)).toEqual({ series: [], breaks: [] });
  });
});

const GAPPED = {
  tv: series([...days("2026-07-01", 5), ...days("2026-07-11", 5)]),
  breaks: [5],
  changes: [],
  firstDate: "2026-07-01",
  lastDate: "2026-07-15",
};

/** The lane fills: one closed area subpath per run of observed days. */
const areaPaths = (svg: Element): Element[] =>
  [...svg.querySelectorAll("path")].filter((p) => (p.getAttribute("fill") ?? "none") !== "none");

describe("the endpoint chart's lane fill", () => {
  test("a gap in the series leaves a hole in the fill", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const d = areaPaths(parse(chartSvg(null, GAPPED, 1000)))[0].getAttribute("d")!;
    expect(d.match(/Z/g)?.length, "the fill was drawn straight across the gap").toBe(2);
  });

  test("no gap, no hole", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const flat = { ...GAPPED, tv: series(days("2026-07-01", 10)), breaks: [] };
    const d = areaPaths(parse(chartSvg(null, flat, 1000)))[0].getAttribute("d")!;
    expect(d.match(/Z/g)?.length).toBe(1);
  });
});

const TIMELINE = {
  date_min: "2026-06-01",
  date_max: "2026-08-31",
  changes: [],
  endpoints: [
    {
      slug: "m2fa23p1",
      provider: "p1",
      base: "p1",
      providerSlug: "p1",
      model: "m/a",
      modelSlug: "m2fa",
      methods: ["b3it"],
      first: "2026-07-01",
      last: "2026-07-15",
      n_changes: 0,
      lt: null,
      b3it: { tv: GAPPED.tv, breaks: GAPPED.breaks, changes: [] },
      status: { lt: "off", bi: "monitoring", headline: "monitored", reason: "" },
    },
  ],
};

const LABELS = {
  name: (ep: { provider: string }) => ep.provider,
  changeName: () => "",
  group: (ep: { providerSlug: string }) => ({
    key: ep.providerSlug,
    label: ep.providerSlug,
    href: "#",
    page: "provider page",
  }),
};

async function renderStrip(): Promise<Element> {
  const { renderTimeline } = await import("../src/timeline");
  document.body.innerHTML = `<div class="chartbox"><div id="cmp"></div></div><div id="cmptip"></div>`;
  const panel = document.getElementById("cmp")!;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  renderTimeline(panel, TIMELINE as any, LABELS as any);
  return panel.querySelector(".row[data-slug] .spark svg")!;
}

describe("the shared timeline's strip fill", () => {
  test("the fill stops at the endpoint's own span, not at the page's", async () => {
    const svg = await renderStrip();
    const W = 1000;
    // 2026-07-01..2026-07-15 sits inside a 2026-06-01..2026-08-31 axis
    const bounds = areaPaths(svg).flatMap((p) => xs(p.getAttribute("d")!));
    expect(Math.min(...bounds), "the fill runs back to the left edge of the page axis")
      .toBeGreaterThan(0);
    expect(Math.max(...bounds), "the fill runs on to the right edge of the page axis")
      .toBeLessThan(W);
  });

  test("a gap in the series leaves a hole in the fill", async () => {
    const svg = await renderStrip();
    const d = areaPaths(svg)
      .map((p) => p.getAttribute("d")!)
      .join(" ");
    expect(d.match(/Z/g)?.length, "the fill was drawn straight across the gap").toBe(2);
  });
});
