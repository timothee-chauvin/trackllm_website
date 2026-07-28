/**
 * Renders each data-driven page the way a browser does -- real generated HTML,
 * real generated JSON -- and checks the output is not obviously broken.
 *
 * `tsc` cannot catch this class of bug: an interface that simply lies about its
 * JSON typechecks fine and renders `undefined/yr`. That is what shipped when the
 * Overview's provider rows changed shape, and it stayed broken across several
 * commits with a green typecheck. These assertions are the cheap net for it:
 * every mount point fills, nothing renders a placeholder token, every internal
 * link points at a file that exists.
 *
 * Requires a build first (`make build`) -- it reads the generated site, not
 * fixtures, so it also fails when the generator stops emitting something.
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

const SITE = resolve(import.meta.dir, "..");

/** Values that reach the page only when data is missing or mis-shaped. */
const PLACEHOLDERS = ["undefined", "NaN", "Invalid Date", "[object Object]"];

/** The two entrypoints not covered: both render through Plotly, which needs a
 *  real canvas. They are static-shell pages whose data is passed by the
 *  template rather than fetched, so they cannot fail in the way above. */
interface Page {
  name: string;
  html: string;
  entry: string;
  ready: string;
  mounts: string[];
}

const PAGES: Page[] = [
  {
    name: "overview",
    html: "index.html",
    entry: "../src/overview.ts",
    ready: "#dirBody tr",
    mounts: ["telemetry", "feed", "provBoards", "provBody", "dirBody"],
  },
  {
    name: "changes",
    html: "changes.html",
    entry: "../src/changes.ts",
    ready: "#log .event",
    mounts: ["lede", "summary", "hist", "topEndpoints", "log"],
  },
  {
    name: "provider",
    html: "providers/chutes.html",
    entry: "../src/provider.ts",
    ready: "#epBody tr",
    mounts: ["lede", "summary", "ratecards", "timeline", "variantBody", "epBody"],
  },
  {
    name: "model",
    html: "models/deepseek2fdeepseek-chat-v3-0324.html",
    entry: "../src/model.ts",
    ready: "#cmp .row",
    mounts: ["lede", "summary", "cmp"],
  },
];

function requireBuilt(path: string): string {
  const full = join(SITE, path);
  if (!existsSync(full)) {
    throw new Error(`${path} is missing -- run \`make build\` before these tests`);
  }
  return full;
}

/** Serve the page's fetches from disk, resolved against the page's own directory
 *  exactly as a browser would resolve them. */
function stubFetch(pageDir: string): void {
  globalThis.fetch = (async (input: string | URL) => {
    const url = String(input);
    const file = resolve(SITE, pageDir, url);
    if (!existsSync(file)) {
      return { ok: false, status: 404, json: async () => null } as Response;
    }
    const body = readFileSync(file, "utf8");
    return { ok: true, status: 200, json: async () => JSON.parse(body) } as Response;
  }) as typeof fetch;
}

async function waitFor(selector: string, timeoutMs = 10_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (document.querySelector(selector)) return;
    await new Promise((r) => setTimeout(r, 20));
  }
  throw new Error(`nothing rendered into ${selector} within ${timeoutMs}ms`);
}

beforeAll(() => {
  GlobalRegistrator.register();
  // happy-dom ships no Web Animations API. The Overview animates the hero
  // trace's stroke, and an unhandled TypeError there aborts init() before
  // anything renders -- a gap in the environment, not in the page.
  const proto = globalThis.Element.prototype as unknown as Record<string, unknown>;
  proto.animate ??= () => ({ finished: Promise.resolve(), cancel: () => {} });
});
afterAll(() => GlobalRegistrator.unregister());

describe.each(PAGES)("$name page", (page) => {
  test("renders against the generated data", async () => {
    const html = readFileSync(requireBuilt(page.html), "utf8");
    document.documentElement.innerHTML = html;
    stubFetch(dirname(page.html));

    await import(page.entry);
    await waitFor(page.ready);
  });

  test("fills every mount point", () => {
    for (const id of page.mounts) {
      const el = document.getElementById(id);
      expect(el, `#${id} is not in ${page.html}`).not.toBeNull();
      expect(el!.innerHTML.trim(), `#${id} rendered empty`).not.toBe("");
    }
  });

  test("renders no placeholder values", () => {
    const text = document.body.textContent ?? "";
    for (const bad of PLACEHOLDERS) {
      expect(text, `"${bad}" reached the page`).not.toContain(bad);
    }
  });

  test("every internal link resolves to a generated file", () => {
    const pageDir = dirname(page.html);
    const missing: string[] = [];
    for (const a of document.querySelectorAll("a[href]")) {
      const href = a.getAttribute("href") ?? "";
      if (!href || href.startsWith("http") || href.startsWith("#")) continue;
      const target = resolve(SITE, pageDir, href.split("#")[0]);
      if (!existsSync(target)) missing.push(href);
    }
    expect(missing, `dead links: ${missing.slice(0, 5).join(", ")}`).toHaveLength(0);
  });
});
