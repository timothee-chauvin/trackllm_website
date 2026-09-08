/**
 * The model and provider pages' monitoring span: "since <month>" while any of
 * the page's endpoints is still monitored, the closed "<month> – <month>" range
 * only once every one of them is retired. Rendered from the generated pages,
 * with the page JSON's own `n_active` (model.py / provider.py) flipped, so the
 * copy is checked in both states whatever the live fleet happens to be in.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

const SITE = resolve(import.meta.dir, "..");

function requireBuilt(path: string): string {
  const full = join(SITE, path);
  if (!existsSync(full)) {
    throw new Error(`${path} is missing -- run \`make build\` before these tests`);
  }
  return full;
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

/** Serve the page's fetches from disk, with `patch` applied to the page's own JSON. */
function stubFetch(pageDir: string, ownJson: string, patch: Record<string, unknown>): void {
  globalThis.fetch = (async (input: string | URL) => {
    const file = resolve(SITE, pageDir, String(input));
    if (!existsSync(file)) {
      return { ok: false, status: 404, json: async () => null } as Response;
    }
    const data = JSON.parse(readFileSync(file, "utf8"));
    return {
      ok: true,
      status: 200,
      json: async () => (file === resolve(SITE, ownJson) ? { ...data, ...patch } : data),
    } as Response;
  }) as typeof fetch;
}

const MONTH = "[A-Z][a-z]{2} \\d{4}";
const RANGE = new RegExp(`${MONTH} – ${MONTH}`);

/** One tracked model with a series, and the provider its first endpoint is on. */
function pickPages(): { model: string; provider: string } {
  const rows = JSON.parse(readFileSync(requireBuilt("data/overview.json"), "utf8")).endpoints;
  const row = rows.find(
    (r: { methods: string[]; providerSlug: string; modelSlug: string }) =>
      r.methods.length && existsSync(join(SITE, `providers/${r.providerSlug}.html`)),
  );
  if (!row) throw new Error("no tracked endpoint with a provider page in overview.json");
  return { model: row.modelSlug, provider: row.providerSlug };
}
const PAGES = pickPages();

describe("model page", () => {
  const render = async (n_active: number): Promise<string> => {
    document.documentElement.innerHTML = readFileSync(
      requireBuilt(`models/${PAGES.model}.html`),
      "utf8",
    );
    stubFetch("models", `data/models/${PAGES.model}.json`, { n_active });
    await (await import("../src/model")).init();
    const card = [...document.querySelectorAll("#summary .s")].find(
      (s) => s.querySelector(".k")?.textContent === "Monitored",
    );
    if (!card) throw new Error("no Monitored card on the model page");
    return card.querySelector(".v")!.textContent!;
  };

  test("reads Since <month> while an endpoint is still tracked", async () => {
    expect(await render(1)).toMatch(new RegExp(`^Since ${MONTH}$`));
  });

  test("reads the closed range once every endpoint is retired", async () => {
    expect(await render(0)).toMatch(new RegExp(`^${MONTH} – ${MONTH}$`));
  });
});

describe("provider page", () => {
  const render = async (n_active: number): Promise<{ lede: string; cards: string }> => {
    document.documentElement.innerHTML = readFileSync(
      requireBuilt(`providers/${PAGES.provider}.html`),
      "utf8",
    );
    stubFetch("providers", `data/providers/${PAGES.provider}.json`, { n_active });
    await (await import("../src/provider")).init();
    return {
      lede: document.getElementById("lede")!.textContent!,
      cards: document.getElementById("ratecards")!.textContent!,
    };
  };

  test("reads monitored since <month> while an endpoint is still active", async () => {
    const { lede, cards } = await render(1);
    expect(lede).toMatch(new RegExp(`monitored since ${MONTH}\\.`));
    expect(lede).not.toMatch(RANGE);
    expect(cards).not.toMatch(RANGE);
    const active = [...document.querySelectorAll("#summary .s")].find(
      (s) => s.querySelector(".k")?.textContent === "Still active",
    );
    expect(active?.querySelector(".v")?.textContent).toBe("1"); // the JSON's n_active, not a recount
  });

  test("reads the closed range once every endpoint is retired", async () => {
    const { lede, cards } = await render(0);
    expect(lede).toMatch(new RegExp(`monitored ${MONTH} – ${MONTH}\\.`));
    expect(lede).not.toContain("monitored since");
    expect(cards).not.toContain("monitored since");
  });
});
