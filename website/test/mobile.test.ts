/**
 * The phone-sized failures the desktop-sized tests cannot see.
 *
 * happy-dom has no layout engine (see smoke.test.ts), so nothing here asserts a
 * pixel: the endpoint chart is checked through its markup at two container widths,
 * and the two CSS-only guarantees are guarded at their rule instead. The layout
 * itself was measured in headless Chromium while writing this -- see the PR.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, beforeEach, describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

const SITE = resolve(import.meta.dir, "..");
const DESIGN_VW = 1000; // endpoint.ts's fallback width, used wherever nothing measures
const PHONE_VW = 300; // a 375px phone leaves the endpoint chart ~264px of content box

function requireBuilt(path: string): string {
  const full = join(SITE, path);
  if (!existsSync(full)) {
    throw new Error(`${path} is missing -- run \`make build\` before these tests`);
  }
  return full;
}

/** 14 months of drift with one changepoint: enough months that the labels have to
 *  thin out on a phone, and few enough that they all fit at the design width. */
const LT = {
  drift: [
    ["2025-01-01", 0.02], ["2025-06-01", 0.05], ["2025-06-20", 0.62], ["2026-02-01", 0.58],
  ] as [string, number][],
  changes: [{ date: "2025-06-20", sigma: "42σ", drift: 0.62 }],
  firstDate: "2025-01-01",
  lastDate: "2026-02-01",
};

/** Three changepoints in one week: on a phone their labels all want the same stretch
 *  of the same baseline, which is where the third one used to print over the second. */
const CROWDED = {
  ...LT,
  changes: [
    { date: "2025-06-16", sigma: "23σ", drift: 0.6 },
    { date: "2025-06-20", sigma: "53σ", drift: 0.62 },
    { date: "2025-06-23", sigma: "128σ", drift: 0.61 },
  ],
};

/** The same three, months apart: at the design width nothing has to give. */
const SPREAD = {
  ...LT,
  changes: [
    { date: "2025-03-01", sigma: "23σ", drift: 0.6 },
    { date: "2025-08-01", sigma: "53σ", drift: 0.62 },
    { date: "2025-12-01", sigma: "128σ", drift: 0.61 },
  ],
};

/** Serve a page's fetches from disk, resolved against the page's own directory. */
function stubFetch(pageDir: string): void {
  globalThis.fetch = (async (input: string | URL) => {
    const file = resolve(SITE, pageDir, String(input));
    if (!existsSync(file)) return { ok: false, status: 404, json: async () => null } as Response;
    return {
      ok: true,
      status: 200,
      json: async () => JSON.parse(readFileSync(file, "utf8")),
    } as Response;
  }) as typeof fetch;
}

/** Load a generated page and run its script against it, the way smoke.test.ts does. */
async function renderPage(path: string, entry: string): Promise<void> {
  document.documentElement.innerHTML = readFileSync(requireBuilt(path), "utf8");
  stubFetch(dirname(path));
  await (await import(entry)).init();
}

const attr = (svg: Element, name: string): string => svg.getAttribute(name) ?? "";
const parse = (markup: string): Element => {
  const host = document.createElement("div");
  host.innerHTML = markup;
  return host.firstElementChild!;
};
/** The x-axis month labels: the only <text> anchored on the chart's bottom rule. */
const monthLabels = (svg: Element): Element[] =>
  [...svg.querySelectorAll("text")].filter((t) => /^\w{3} \d\d$/.test(t.textContent ?? ""));
/** The changepoint labels: bold, and the only bold text drawn at 10.5 (the lane
 *  titles, bold too, sit on the same baselines at 11.5). */
const cpLabels = (svg: Element): Element[] =>
  [...svg.querySelectorAll("text")].filter(
    (t) => attr(t, "font-weight") === "600" && attr(t, "font-size") === "10.5"
  );
/** The lane title, which shares row 0 of its lane's baseline with those labels. */
const laneTitles = (svg: Element): Element[] =>
  [...svg.querySelectorAll("text")].filter(
    (t) => attr(t, "font-weight") === "600" && attr(t, "font-size") === "11.5"
  );

beforeAll(() => {
  GlobalRegistrator.register();
  globalThis.ResizeObserver ??= class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  } as unknown as typeof ResizeObserver;
});
afterAll(() => GlobalRegistrator.unregister());

const EP_PAGE = "endpoints/deepseek2fdeepseek-chat-v3-032423fireworks.html";

describe("endpoint chart at a phone's width", () => {
  // the page is up before the first import, so endpoint.ts's own load-time init()
  // has the manifest it looks for
  beforeAll(() => renderPage(EP_PAGE, "../src/endpoint"));

  test("draws in the container's own pixels, so its text keeps its size", async () => {
    const { chartSvg } = await import("../src/endpoint");
    for (const w of [DESIGN_VW, PHONE_VW]) {
      const svg = parse(chartSvg(LT, null, w));
      expect(attr(svg, "viewBox"), `viewBox at ${w}`).toBe(`0 0 ${w} 324`);
    }
    // the same font-size over a narrower viewBox is what "bigger on screen" means
    // here: the SVG is scaled to the container either way, so relative size is size.
    const [wide, narrow] = [DESIGN_VW, PHONE_VW].map((w) => parse(chartSvg(LT, null, w)));
    const relSize = (svg: Element): number =>
      +attr(monthLabels(svg)[0], "font-size") / +attr(svg, "viewBox").split(" ")[2];
    expect(relSize(narrow)).toBeGreaterThan(relSize(wide) * 3);
  });

  test("thins the month labels rather than overprinting them", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const wide = monthLabels(parse(chartSvg(LT, null, DESIGN_VW)));
    const narrow = monthLabels(parse(chartSvg(LT, null, PHONE_VW)));
    expect(wide.length, "the design width should still label every month").toBe(14);
    expect(narrow.length).toBeLessThan(wide.length);
    expect(narrow.length, "a phone still needs some sense of the time axis").toBeGreaterThan(1);

    // every kept label has room for the ~41px "Jul 26" it draws
    const xs = narrow.map((t) => +attr(t, "x")).sort((a, b) => a - b);
    const pitch = Math.min(...xs.slice(1).map((x, i) => x - xs[i]));
    expect(pitch, `labels ${pitch.toFixed(1)}px apart`).toBeGreaterThan(41);
  });

  test("the design width is laid out exactly as before", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const svg = parse(chartSvg(LT, null, DESIGN_VW));
    // the bottom rule spans the plot area: PL=50 to VW-PR=980, as it always has
    const rule = [...svg.querySelectorAll("line")].pop()!;
    expect([attr(rule, "x1"), attr(rule, "x2")]).toEqual(["50", "980"]);
    expect(attr(svg, "viewBox")).toBe("0 0 1000 324");
  });

  test("never prints one changepoint label over another", async () => {
    const { chartSvg } = await import("../src/endpoint");
    const CHAR_W = 6.4; // endpoint.ts's own estimate, var(--mono) at font-size 10.5
    for (const w of [320, 375, PHONE_VW, 560, DESIGN_VW]) {
      const svg = parse(chartSvg(CROWDED, null, w));
      const rows = new Map<string, [number, number][]>();
      const put = (t: Element, span: [number, number]): void =>
        void rows.set(attr(t, "y"), [...(rows.get(attr(t, "y")) ?? []), span]);
      // the lane title is left-anchored on the same baseline; the labels are centred
      for (const t of laneTitles(svg)) {
        const x = +attr(t, "x");
        put(t, [x, x + (t.textContent ?? "").length * 7.2]);
      }
      for (const t of cpLabels(svg)) {
        const half = ((t.textContent ?? "").length * CHAR_W) / 2;
        const x = +attr(t, "x");
        put(t, [x - half, x + half]);
      }
      for (const [y, spans] of rows) {
        spans.sort((a, b) => a[0] - b[0]);
        for (let i = 1; i < spans.length; i++) {
          expect(spans[i][0], `labels overprint on row y=${y} at ${w}px`).toBeGreaterThanOrEqual(
            spans[i - 1][1]
          );
        }
      }
      // a label that has to go still leaves the day marked
      const rules = [...svg.querySelectorAll("line")].filter((l) => attr(l, "stroke-dasharray"));
      expect(rules.length, `changepoint rules at ${w}px`).toBe(CROWDED.changes.length);
      expect(cpLabels(svg).length, `every label dropped at ${w}px`).toBeGreaterThan(0);
    }
  });

  test("drops a label only when it has nowhere to go", async () => {
    const { chartSvg } = await import("../src/endpoint");
    expect(cpLabels(parse(chartSvg(SPREAD, null, DESIGN_VW))).length).toBe(SPREAD.changes.length);
  });

  test("a real endpoint page renders a chart", async () => {
    await renderPage(EP_PAGE, "../src/endpoint");
    const svg = document.querySelector("#mainchart svg");
    expect(svg, "the endpoint chart did not render").not.toBeNull();
    // no layout engine here, so the container measures 0 and the fallback applies
    expect(attr(svg!, "viewBox")).toBe(`0 0 ${DESIGN_VW} 324`);
    expect(monthLabels(svg!).length).toBeGreaterThan(1);
  });
});

describe("chart marks answer a tap", () => {
  // each test taps, so each starts from a page nothing has been tapped on yet
  beforeEach(() => renderPage("models/qwen2fqwen3-coder.html", "../src/model"));
  const tap = (el: Element): void => {
    el.dispatchEvent(new Event("click", { bubbles: true }));
  };

  test("a strip with changes is focusable and names them", () => {
    const strips = [...document.querySelectorAll("#cmp .spark svg, #cmp .allrow svg")];
    expect(strips.length, "no strips on the model page").toBeGreaterThan(1);
    const named = strips.filter((s) => s.hasAttribute("data-tip"));
    expect(named.length, "not one strip carries its changes as text").toBeGreaterThan(0);
    for (const s of named) {
      expect(s.getAttribute("tabindex")).toBe("0");
      expect(s.getAttribute("role")).toBe("img");
      // the accessible name and the tap text are the same words
      expect(s.getAttribute("aria-label")).toBe(s.getAttribute("data-tip"));
      expect(s.getAttribute("aria-label")).toMatch(/\d{4}-\d{2}-\d{2}/);
    }
    // a strip with nothing to say is decorative rather than a silent tab stop
    for (const s of strips.filter((x) => !x.hasAttribute("data-tip"))) {
      expect(s.getAttribute("aria-hidden")).toBe("true");
      expect(s.hasAttribute("tabindex")).toBe(false);
    }
  });

  test("tapping one captions it, right where it was tapped", () => {
    const strip = document.querySelector("#cmp [data-tip]")!;
    expect(document.querySelector("#cmp .tipline"), "the caption starts absent").toBeNull();
    tap(strip);
    const line = document.querySelector("#cmp .tipline")!;
    expect(line.textContent).toBe(strip.getAttribute("data-tip")!);
    // adjacency is the whole point: a caption at the foot of the panel is a screen
    // or more below the first strip on a phone
    expect(strip.nextElementSibling, "the caption is not next to its strip").toBe(line);
  });

  test("tapping again takes it away", () => {
    const strip = document.querySelector("#cmp [data-tip]")!;
    tap(strip);
    expect(document.querySelectorAll("#cmp .tipline").length, "captions piled up").toBe(1);
    tap(strip);
    expect(document.querySelector("#cmp .tipline")).toBeNull();
  });

  test("only ever one caption: a second strip moves it", () => {
    const [a, b] = [...document.querySelectorAll("#cmp [data-tip]")];
    tap(a);
    tap(b);
    const lines = [...document.querySelectorAll("#cmp .tipline")];
    expect(lines.length).toBe(1);
    expect(b.nextElementSibling).toBe(lines[0]);
  });

  test("the focus a tap brings does not undo the tap", () => {
    const strip = document.querySelector("#cmp [data-tip]")!;
    // the order a touch tap arrives in: focus first, then the click
    strip.dispatchEvent(new Event("focusin", { bubbles: true }));
    tap(strip);
    expect(document.querySelector("#cmp .tipline"), "the tap cancelled itself").not.toBeNull();
    strip.dispatchEvent(new Event("focusout", { bubbles: true }));
    expect(document.querySelector("#cmp .tipline"), "leaving the strip left the caption").toBeNull();
  });
});

/** Both fixes below are pure CSS, and both are invisible to a DOM with no layout:
 *  guard the rule that carries them, the way smoke.test.ts guards .feed. */
describe("sticky month headers clear the nav", () => {
  const css = readFileSync(join(SITE, "style.css"), "utf8");
  const rule = (selector: string): string => {
    const m = css.match(new RegExp(`^${selector.replace(/[.*]/g, "\\$&")} \\{[^}]*\\}`, "m"));
    expect(m, `${selector} rule not found in style.css`).not.toBeNull();
    return m![0];
  };

  test("the nav links never wrap to a second row", () => {
    // .mohead pins itself var(--nav-h) from the top; a wrapped nav is taller than
    // that and covers the banner, so the one row is what makes the constant true.
    const links = rule(".nav-links");
    expect(links, "the nav can still wrap").toContain("flex-wrap: nowrap");
    expect(links, "an unwrapped nav that cannot scroll hides its own links").toContain(
      "overflow-x: auto",
    );
    expect(links, "an unshrinkable flex item overflows the page instead").toContain("min-width: 0");
  });

  test("the banner still offsets by the nav height", () => {
    expect(rule(".mohead")).toContain("top: var(--nav-h)");
  });
});
