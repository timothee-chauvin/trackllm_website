/**
 * The provider drift-rate plot in its three modes (rate_plot.ts), and the Providers
 * page's method toggle with its URL hash (providers.ts), against the generated site.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { ProviderRate } from "../src/overview_data";
import { RateMode, rateStatLine, rateablePlotRows, ratePlot } from "../src/rate_plot";

const SITE = resolve(import.meta.dir, "..");

function requireBuilt(path: string): string {
  const full = join(SITE, path);
  if (!existsSync(full)) throw new Error(`${path} is missing -- run \`make build\` before these tests`);
  return full;
}

const BRAND = { name: "", logo: null, kind: "icon" as const, mono: false, dark: null };
type Block = { years: number; changes: number; rate: number | null; ci: [number, number] | null };
const thin: Block = { years: 0.2, changes: 1, rate: null, ci: null };
function prov(name: string, lt: Block, b3it: Block): ProviderRate {
  return {
    name, slug: name, brand: { ...BRAND, name }, n_endpoints: 5, n_models: 3, n_variants: 1, last_change: null,
    lt_endpoints: lt.rate === null ? 0 : 4, lt_years: lt.years, lt_changes: lt.changes, lt_rate: lt.rate, lt_ci: lt.ci,
    b3it_endpoints: 2, b3it_years: b3it.years, b3it_changes: b3it.changes, b3it_rate: b3it.rate, b3it_ci: b3it.ci,
  };
}
const PROVS = [
  prov("dual", { years: 4, changes: 4, rate: 1, ci: [0.02, 1.98] }, { years: 1, changes: 3, rate: 3, ci: [0, 6.4] }),
  prov("ltonly", { years: 10, changes: 20, rate: 2, ci: [1.12, 2.88] }, thin),
  prov("quiet", { years: 2, changes: 0, rate: 0, ci: [0, 1.5] }, thin),
  prov("b3only", thin, { years: 0.8, changes: 6, rate: 7.5, ci: [1.5, 13.5] }),
  prov("nothing", thin, thin),
];
const names = (rows: ProviderRate[]): string[] => rows.map(p => p.name);

describe("rateablePlotRows", () => {
  test("LT: rateable rows by rate, quiet ones by their ceiling", () => {
    expect(names(rateablePlotRows(PROVS, "lt"))).toEqual(["ltonly", "dual", "quiet"]);
  });
  test("B3IT: only rows with a B3IT rate", () => {
    expect(names(rateablePlotRows(PROVS, "b3it"))).toEqual(["b3only", "dual"]);
  });
  test("both: either rate qualifies, dual-method rows first, then by the higher rate", () => {
    expect(names(rateablePlotRows(PROVS, "both"))).toEqual(["dual", "b3only", "ltonly", "quiet"]);
  });
});

describe("ratePlot", () => {
  /** Right edges of the whiskers, in % of the axis: the widest must touch the end, none may pass it. */
  const whiskerEnds = (mode: RateMode): number[] => {
    document.body.innerHTML = ratePlot(rateablePlotRows(PROVS, mode), "", mode);
    return [...document.querySelectorAll<HTMLElement>(".rtrack .ci")].map(ci => parseFloat(ci.style.left) + parseFloat(ci.style.width));
  };
  test("the LT plot carries no B3IT marks or classes", () => {
    const html = ratePlot(rateablePlotRows(PROVS, "lt"), "", "lt");
    expect(html).not.toContain("b3it");
    expect(html).toContain('<i class="ci" style=');
    expect(html).toContain('<i class="dot" style=');
    expect(html).not.toContain("<span class=\"rval\"><span");
    expect(html).toContain("<small>5 ep · 4.0 ep-yr</small>");
  });
  test("the axis spans every whisker drawn in that mode, and no more", () => {
    for (const mode of ["lt", "b3it", "both"] as RateMode[]) {
      const ends = whiskerEnds(mode);
      expect(Math.max(...ends)).toBeCloseTo(100, 1);
      for (const e of ends) expect(e).toBeLessThanOrEqual(100.01);
    }
    // LT's widest whisker (2.88) sets a shorter axis than B3IT's (13.5): the dot at 1.0 sits further right
    document.body.innerHTML = ratePlot(rateablePlotRows(PROVS, "lt"), "", "lt");
    const ltDot = parseFloat(document.querySelector<HTMLElement>('.rrow[href$="dual.html"] .dot')!.style.left);
    document.body.innerHTML = ratePlot(rateablePlotRows(PROVS, "both"), "", "both");
    const bothDot = parseFloat(document.querySelector<HTMLElement>('.rrow[href$="dual.html"] .dot:not(.b3it)')!.style.left);
    expect(ltDot).toBeGreaterThan(bothDot * 3);
  });
  test("B3IT marks wear the b3it modifier", () => {
    document.body.innerHTML = ratePlot(rateablePlotRows(PROVS, "b3it"), "", "b3it");
    expect(document.querySelectorAll(".rtrack .dot.b3it").length).toBe(2);
    expect(document.querySelectorAll(".rtrack .dot:not(.b3it)").length).toBe(0);
  });
  test("both: two stacked marks where both rates exist, a dash where one is missing", () => {
    document.body.innerHTML = ratePlot(rateablePlotRows(PROVS, "both"), "", "both");
    const rows = [...document.querySelectorAll<HTMLElement>(".rrow")];
    const dual = rows[0], b3only = rows[1], ltonly = rows[2];
    expect(dual.querySelectorAll(".rtrack .dot").length).toBe(2);
    expect(dual.querySelectorAll(".rtrack .dot.b3it").length).toBe(1);
    expect([...dual.querySelectorAll(".rval span")].map(s => s.className)).toEqual(["lt", "b3it"]);
    expect(dual.title).toContain("LT: 1.00 changes per endpoint-year");
    expect(dual.title).toContain("B3IT: 3.00 changes per endpoint-year");
    expect(dual.querySelector("small")!.textContent).toBe("5 ep · 4.0 / 1.0 ep-yr");
    expect([...b3only.querySelectorAll(".rval span")].map(s => s.className)).toEqual(["none", "b3it"]);
    expect(b3only.querySelectorAll(".rtrack .dot").length).toBe(1);
    expect([...ltonly.querySelectorAll(".rval span")].map(s => s.className)).toEqual(["lt", "none"]);
  });
  test("the stat line counts the changes the floor keeps off the plot", () => {
    document.body.innerHTML = rateStatLine(PROVS, "b3it");
    expect(document.body.textContent).toBe("2 rateable providers · 9 of 12 B3IT changes over 1.8 ep-yr (3 more at 3 providers under the 0.5 ep-yr floor)");
    document.body.innerHTML = rateStatLine(PROVS, "both");
    expect(document.body.textContent).toContain("3 rateable providers · 24 of 26 LT changes over 16.0 ep-yr (2 more at 2 providers under the 0.5 ep-yr floor)");
    expect(document.body.textContent).toContain("|");
  });
});

describe("the Providers page toggle", () => {
  function stubFetch(): void {
    globalThis.fetch = (async (input: string | URL) => {
      const file = resolve(SITE, String(input));
      return { ok: true, status: 200, json: async () => JSON.parse(readFileSync(file, "utf8")) } as Response;
    }) as typeof fetch;
  }
  async function render(hash: string): Promise<void> {
    document.documentElement.innerHTML = readFileSync(requireBuilt("providers.html"), "utf8");
    location.hash = hash;
    stubFetch();
    await (await import("../src/providers")).init();
  }
  const onChip = (): string => document.querySelector<HTMLElement>("#rateMode .chip.on")!.dataset.m!;
  const full = (): ProviderRate[] => JSON.parse(readFileSync(requireBuilt("data/overview.json"), "utf8")).providers;

  test("opens on LT with no hash, and draws every LT-rateable provider", async () => {
    await render("");
    expect(onChip()).toBe("lt");
    expect(document.querySelectorAll("#rateMode .chip.on").length).toBe(1);
    expect(document.querySelectorAll("#provPlot .rrow").length).toBe(rateablePlotRows(full(), "lt").length);
    expect(document.querySelectorAll("#provPlot .b3it").length).toBe(0);
    expect(document.getElementById("provNote")!.textContent).toContain("LT covers only");
  });
  test("#b3it opens on B3IT", async () => {
    await render("#b3it");
    expect(onChip()).toBe("b3it");
    expect(document.querySelectorAll("#provPlot .rrow").length).toBe(rateablePlotRows(full(), "b3it").length);
    expect(document.querySelectorAll("#provPlot .dot.b3it").length).toBeGreaterThan(0);
    expect(document.getElementById("provNote")!.textContent).toMatch(/\d+ of \d+ B3IT changes sit at providers/);
  });
  test("clicking a chip switches the mode, the hash, and keeps exactly one chip on", async () => {
    await render("");
    document.querySelector<HTMLElement>('#rateMode .chip[data-m="both"]')!.click();
    expect(onChip()).toBe("both");
    expect(document.querySelectorAll("#rateMode .chip.on").length).toBe(1);
    expect(location.hash).toBe("#both");
    expect(document.getElementById("provPlot")!.classList.contains("both")).toBe(true);
    expect(document.querySelectorAll("#provPlot .rrow").length).toBe(rateablePlotRows(full(), "both").length);
    expect(document.querySelectorAll("#provLegend .k").length).toBe(3);
    document.querySelector<HTMLElement>('#rateMode .chip[data-m="lt"]')!.click();
    expect(location.hash).toBe("");
    expect(document.getElementById("provPlot")!.classList.contains("both")).toBe(false);
  });
  test("an unknown hash falls back to LT", async () => {
    await render("#nonsense");
    expect(onChip()).toBe("lt");
  });
});

beforeAll(() => GlobalRegistrator.register());
afterAll(() => GlobalRegistrator.unregister());
