/**
 * Status front end against the generated site: chips, search with highlight,
 * untracked endpoint/model pages. Same harness as smoke.test.ts -- real
 * generated HTML + JSON, so it also fails when the generator stops emitting
 * the status fields. Rows are discovered from the data, never hardcoded to a
 * catalog that churns daily.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

const SITE = resolve(import.meta.dir, "..");

interface FleetRow {
  slug: string;
  model: string;
  modelSlug: string;
  org: string;
  provider: string;
  providerSlug: string;
  methods: string[];
  headline: string;
  reason: string;
}

function requireBuilt(path: string): string {
  const full = join(SITE, path);
  if (!existsSync(full)) {
    throw new Error(`${path} is missing -- run \`make build\` before these tests`);
  }
  return full;
}

const OVERVIEW = JSON.parse(readFileSync(requireBuilt("data/overview.json"), "utf8"));
const ROWS: FleetRow[] = OVERVIEW.endpoints;
const untracked = (r: FleetRow): boolean => r.methods.length === 0;

function rowWith(pred: (r: FleetRow) => boolean, what: string): FleetRow {
  const row = ROWS.find(pred);
  if (!row) throw new Error(`no fleet row matches: ${what}`);
  return row;
}

function stubFetch(pageDir: string): void {
  globalThis.fetch = (async (input: string | URL) => {
    const file = resolve(SITE, pageDir, String(input));
    if (!existsSync(file)) {
      return { ok: false, status: 404, json: async () => null } as Response;
    }
    return {
      ok: true,
      status: 200,
      json: async () => JSON.parse(readFileSync(file, "utf8")),
    } as Response;
  }) as typeof fetch;
}

beforeAll(() => {
  GlobalRegistrator.register();
  globalThis.ResizeObserver ??= class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  } as unknown as typeof ResizeObserver;
});
afterAll(() => GlobalRegistrator.unregister());

async function renderOverview(): Promise<void> {
  document.documentElement.innerHTML = readFileSync(requireBuilt("index.html"), "utf8");
  stubFetch(".");
  await (await import("../src/overview")).init();
}

const search = (q: string): void => {
  const input = document.getElementById("q") as HTMLInputElement;
  input.value = q;
  input.dispatchEvent(new Event("input"));
};

const chip = (st: string): HTMLElement => {
  const el = document.querySelector<HTMLElement>(`#chips .chip[data-st="${st}"]`);
  if (!el) throw new Error(`no status chip for ${st}`);
  return el;
};

const shownCount = (): number =>
  Number(/^(\d+) of /.exec(document.getElementById("dirFoot")!.textContent ?? "")?.[1]);

describe("overview status chips", () => {
  test("default shows only tracked-headline rows, chips reveal the rest", async () => {
    await renderOverview();
    const byHeadline = (h: string): number => ROWS.filter((r) => r.headline === h).length;
    expect(chip("tracked").classList.contains("on")).toBe(true);
    expect(shownCount()).toBe(byHeadline("tracked"));

    chip("untrackable").dispatchEvent(new Event("click", { bubbles: true }));
    expect(shownCount()).toBe(byHeadline("tracked") + byHeadline("untrackable"));

    chip("tracked").dispatchEvent(new Event("click", { bubbles: true }));
    expect(shownCount()).toBe(byHeadline("untrackable"));

    // untracked rows show a status badge and their one-line reason
    const body = document.getElementById("dirBody")!;
    expect(body.querySelectorAll(".badge.st-untrackable").length).toBeGreaterThan(0);
    expect(body.textContent).toContain("no tracking method can work");
  });

  test("no status chip active means no status constraint", async () => {
    await renderOverview();
    chip("tracked").dispatchEvent(new Event("click", { bubbles: true }));
    expect(shownCount()).toBe(ROWS.length);
  });

  test("provider names without a provider page are not linked", async () => {
    await renderOverview();
    chip("tracked").dispatchEvent(new Event("click", { bubbles: true })); // show all
    const hrefs = document
      .getElementById("dirBody")!
      .querySelectorAll('a[href^="providers/"]');
    expect(hrefs.length).toBeGreaterThan(0);
    for (const a of hrefs) {
      const href = a.getAttribute("href")!;
      expect(existsSync(join(SITE, href)), `dead link: ${href}`).toBe(true);
    }
  });
});

describe("overview search", () => {
  test("finds gpt-5 by model despite the tracked chip, with <mark>", async () => {
    rowWith((r) => untracked(r) && r.model.includes("gpt-5"), "untracked gpt-5");
    await renderOverview();
    search("gpt-5");
    const matches = ROWS.filter((r) =>
      `${r.model} ${r.provider} ${r.org}`.toLowerCase().includes("gpt-5"),
    );
    expect(shownCount()).toBe(matches.length);
    const body = document.getElementById("dirBody")!;
    expect(body.innerHTML).toContain("<mark>gpt-5</mark>");
    expect(body.querySelectorAll(".badge.st").length).toBeGreaterThan(0);
  });

  test("finds alibaba by provider name, with <mark>", async () => {
    await renderOverview();
    search("alibaba");
    const matches = ROWS.filter((r) =>
      `${r.model} ${r.provider} ${r.org}`.toLowerCase().includes("alibaba"),
    );
    expect(matches.length).toBeGreaterThan(0);
    expect(shownCount()).toBe(matches.length);
    const prov = document.querySelector("#dirBody .prov-cell mark");
    expect(prov?.textContent?.toLowerCase()).toBe("alibaba");
  });

  test("clearing the search restores the chip filter", async () => {
    await renderOverview();
    search("gpt-5");
    search("");
    expect(shownCount()).toBe(ROWS.filter((r) => r.headline === "tracked").length);
  });
});

describe("untracked endpoint page", () => {
  const row = rowWith(
    (r) => untracked(r) && r.headline === "untrackable",
    "untrackable endpoint",
  );
  const html = readFileSync(requireBuilt(`endpoints/${row.slug}.html`), "utf8");

  test("renders the per-method status card and catalog metadata, no chart", () => {
    expect(html).toContain("status-methods");
    expect(html).toContain("meta-grid");
    expect(html).not.toContain('id="mainchart"');
    expect(html).not.toContain('id="statuscard"');
    expect(html).not.toContain("js/endpoint.js");
  });

  test("tracked endpoint pages keep the chart and gain the status card", () => {
    const tracked = rowWith((r) => r.methods.includes("lt"), "tracked endpoint");
    const trackedHtml = readFileSync(requireBuilt(`endpoints/${tracked.slug}.html`), "utf8");
    expect(trackedHtml).toContain("status-methods");
    expect(trackedHtml).toContain('id="mainchart"');
  });
});

describe("untracked model page", () => {
  // a model whose endpoints are all untracked: its page is badges, not strips
  const slug = [...new Set(ROWS.filter(untracked).map((r) => r.modelSlug))].find((s) => {
    const m = JSON.parse(readFileSync(requireBuilt(`data/models/${s}.json`), "utf8"));
    return m.n_endpoints === 0;
  });
  if (!slug) throw new Error("no fully-untracked model in the catalog");

  test("renders one badge row per endpoint and the status summary", async () => {
    const model = JSON.parse(readFileSync(requireBuilt(`data/models/${slug}.json`), "utf8"));
    document.documentElement.innerHTML = readFileSync(
      requireBuilt(`models/${slug}.html`),
      "utf8",
    );
    stubFetch("models");
    await (await import("../src/model")).init();

    const rows = document.querySelectorAll("#cmp .row");
    expect(rows.length).toBe(model.n_endpoints_total);
    expect(document.querySelectorAll("#cmp .badge.st").length).toBe(
      model.n_endpoints_total,
    );
    expect(document.getElementById("lede")!.textContent).toContain(model.status_summary);
    expect(document.querySelector("#cmp .allrow")).toBeNull();
  });
});

describe("provider page untracked rows", () => {
  test("show a badge and reason instead of a pill and sparkline", async () => {
    const row = rowWith(
      (r) => untracked(r) && existsSync(join(SITE, `providers/${r.providerSlug}.html`)),
      "untracked row on an existing provider page",
    );
    document.documentElement.innerHTML = readFileSync(
      requireBuilt(`providers/${row.providerSlug}.html`),
      "utf8",
    );
    stubFetch("providers");
    await (await import("../src/provider")).init();

    const body = document.getElementById("epBody")!;
    expect(body.querySelectorAll(".badge.st").length).toBeGreaterThan(0);
    expect(body.textContent).not.toContain("null");
  });
});

test("the org page badges its untracked models", () => {
  const html = readFileSync(requireBuilt("orgs/anthropic.html"), "utf8");
  expect(html).toContain("badge st st-untrackable");
});
