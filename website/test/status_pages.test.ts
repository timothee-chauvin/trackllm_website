/**
 * Status front end against the generated site: the Endpoints page's chips and
 * search with highlight,
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
  nChanges: number;
  lastChange: string | null;
  recent: boolean;
  headline: string;
  headlines: string[];
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

async function renderEndpoints(): Promise<void> {
  document.documentElement.innerHTML = readFileSync(requireBuilt("endpoints.html"), "utf8");
  stubFetch(".");
  await (await import("../src/endpoints")).init();
}

const click = (el: Element): void => {
  el.dispatchEvent(new Event("click", { bubbles: true }));
};
const chipF = (attr: string, value: string): HTMLElement => {
  const el = document.querySelector<HTMLElement>(`#chips .chip[data-${attr}="${value}"]`);
  if (!el) throw new Error(`no ${attr} chip for ${value}`);
  return el;
};
const matching = (q: string): FleetRow[] =>
  ROWS.filter((r) => `${r.model} ${r.provider} ${r.org}`.toLowerCase().includes(q));

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

const byHeadline = (h: string): FleetRow[] => ROWS.filter((r) => r.headlines.includes(h));
const STATUSES = [...new Set(ROWS.flatMap((r) => r.headlines))];

describe("endpoint directory chips", () => {
  // The status row is a radio: exactly one headline shows, and the one that is
  // on cannot be switched off.
  test("one status at a time, tracked by default", async () => {
    await renderEndpoints();
    expect(chip("tracked").classList.contains("on")).toBe(true);
    expect(shownCount()).toBe(byHeadline("tracked").length);

    click(chip("untrackable"));
    expect(chip("tracked").classList.contains("on")).toBe(false);
    expect(chip("untrackable").getAttribute("aria-pressed")).toBe("true");
    expect(shownCount()).toBe(byHeadline("untrackable").length);

    click(chip("untrackable")); // the active status stays active
    expect(chip("untrackable").classList.contains("on")).toBe(true);
    expect(shownCount()).toBe(byHeadline("untrackable").length);

    // untracked rows show a status badge and their one-line reason
    const body = document.getElementById("dirBody")!;
    expect(body.querySelectorAll(".badge.st-untrackable").length).toBeGreaterThan(0);
    expect(body.textContent).toContain("no tracking method can work");
  });

  // The change row is exclusive but optional, and combines with the status row.
  test("change-history chips are exclusive and combine with the status row", async () => {
    const retiredChanged = rowWith(
      (r) => r.headlines.includes("retired") && r.nChanges > 0,
      "retired-headline row with changes",
    );
    await renderEndpoints();
    click(chipF("c", "everchanged"));
    const changed = ROWS.filter((r) => r.nChanges > 0);
    expect(shownCount()).toBe(changed.filter((r) => r.headlines.includes("tracked")).length);
    expect(
      document.querySelector(`#dirBody a[href="endpoints/${retiredChanged.slug}.html"]`),
    ).toBeNull();

    click(chip("retired"));
    expect(shownCount()).toBe(changed.filter((r) => r.headlines.includes("retired")).length);
    expect(
      document.querySelector(`#dirBody a[href="endpoints/${retiredChanged.slug}.html"]`),
    ).not.toBeNull();

    click(chipF("c", "recent")); // switches, never adds
    expect(chipF("c", "everchanged").classList.contains("on")).toBe(false);
    expect(shownCount()).toBe(ROWS.filter((r) => r.headlines.includes("retired") && r.recent).length);

    click(chipF("c", "recent")); // off again: no change constraint
    expect(shownCount()).toBe(byHeadline("retired").length);
  });

  // A row carries every headline that happened to it (retired and too expensive
  // both, say), and each status chip lists it. The data churns, so the case is
  // discovered; a fleet with no such row skips rather than fakes one.
  const multi = ROWS.find((r) => r.headlines.length > 1);
  test.skipIf(!multi)("a row shows under every headline it carries", async () => {
    await renderEndpoints();
    for (const h of multi!.headlines) {
      click(chip(h));
      expect(shownCount()).toBe(byHeadline(h).length);
      expect(
        document.querySelector(`#dirBody a.model-cell[href="endpoints/${multi!.slug}.html"]`),
      ).not.toBeNull();
    }
    expect(new Set(multi!.headlines).size).toBe(multi!.headlines.length);
  });

  test("method chips conjoin: both on means tracked by both", async () => {
    await renderEndpoints();
    const tracked = byHeadline("tracked");
    const has = (...ms: string[]): number =>
      tracked.filter((r) => ms.every((m) => r.methods.includes(m))).length;
    click(chipF("f", "lt"));
    expect(shownCount()).toBe(has("lt"));
    click(chipF("f", "b3it"));
    expect(shownCount()).toBe(has("lt", "b3it"));
    expect(has("lt", "b3it")).toBeLessThan(has("lt"));
  });

  test("model names and drift strips link to the endpoint page", async () => {
    await renderEndpoints();
    const row = byHeadline("tracked")[0];
    const body = document.getElementById("dirBody")!;
    const links = body.querySelectorAll(`a[href="endpoints/${row.slug}.html"]`);
    // model name, status pill's stretched link, and the strip
    expect(links.length).toBe(3);
    expect(body.querySelector(".model-cell")!.getAttribute("href")).toMatch(/^endpoints\//);
    expect(body.querySelector(".spark-cell a svg")).not.toBeNull();
  });

  test("the last-change column is a date, or a dash when never changed", async () => {
    const never = rowWith((r) => r.headlines.includes("tracked") && r.nChanges === 0, "never-changed row");
    const once = rowWith((r) => r.headlines.includes("tracked") && r.lastChange !== null, "changed row");
    await renderEndpoints();
    const cell = (slug: string): string =>
      document
        .querySelector(`#dirBody a.model-cell[href="endpoints/${slug}.html"]`)!
        .closest("tr")!
        .querySelectorAll("td")[4].textContent!;
    expect(cell(never.slug)).toBe("—");
    expect(cell(once.slug)).toMatch(/^\d{1,2} [A-Z][a-z]{2}/);
  });

  test("provider names without a provider page are not linked", async () => {
    await renderEndpoints();
    const hrefs = new Set<string>();
    for (const st of STATUSES) {
      click(chip(st));
      for (const a of document.getElementById("dirBody")!.querySelectorAll('a[href^="providers/"]')) {
        hrefs.add(a.getAttribute("href")!);
      }
    }
    expect(hrefs.size).toBeGreaterThan(0);
    for (const href of hrefs) {
      expect(existsSync(join(SITE, href)), `dead link: ${href}`).toBe(true);
    }
  });
});

describe("endpoint directory search", () => {
  test("narrows within the chips, and highlights the hit with <mark>", async () => {
    rowWith((r) => untracked(r) && r.model.includes("gpt-5"), "untracked gpt-5");
    await renderEndpoints();
    search("gpt-5");
    const matches = matching("gpt-5");
    expect(shownCount()).toBe(matches.filter((r) => r.headlines.includes("tracked")).length);
    click(chip("untrackable")); // the untracked matches, with their badges
    expect(shownCount()).toBe(matches.filter((r) => r.headlines.includes("untrackable")).length);
    const body = document.getElementById("dirBody")!;
    expect(body.innerHTML).toContain("<mark>gpt-5</mark>");
    expect(body.querySelectorAll(".badge.st").length).toBeGreaterThan(0);
  });

  test("finds alibaba by provider name, with <mark>", async () => {
    await renderEndpoints();
    search("alibaba");
    const matches = matching("alibaba").filter((r) => r.headlines.includes("tracked"));
    expect(matches.length).toBeGreaterThan(0);
    expect(shownCount()).toBe(matches.length);
    const prov = document.querySelector("#dirBody .prov-cell mark");
    expect(prov?.textContent?.toLowerCase()).toBe("alibaba");
  });

  test("clearing the search restores the chip filter", async () => {
    await renderEndpoints();
    search("gpt-5");
    search("");
    expect(shownCount()).toBe(ROWS.filter((r) => r.headlines.includes("tracked")).length);
  });
});

describe("untracked endpoint page", () => {
  const row = rowWith(
    (r) => untracked(r) && r.headlines.includes("untrackable"),
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
    // no drift to show: the section must not promise "Drift by provider"
    expect(document.getElementById("cmpTitle")!.textContent).toBe("Endpoints");
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
