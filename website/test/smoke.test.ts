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
    mounts: ["telemetry", "freshness", "feed", "provBoards", "provBody", "dirBody"],
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
    mounts: ["lede", "summary", "ratecards", "cmp", "timeline", "variantBody", "epBody"],
  },
  {
    // Chosen for its provider mix: one company serving two variants, nine serving
    // one each -- both shapes the group banner has to tell apart.
    name: "model",
    html: "models/qwen2fqwen3-coder.html",
    entry: "../src/model.ts",
    ready: "#cmp .row",
    mounts: ["lede", "summary", "cmp"],
  },
  {
    // Discovered, not hardcoded: which endpoints exist churns daily. It needs a
    // change of its own, or the changes table below the chart has no rows.
    name: "endpoint",
    html: `endpoints/${changedEndpointSlug()}.html`,
    entry: "../src/endpoint.ts",
    ready: "#changerows tr",
    mounts: ["statuscard", "mainchart", "changerows"],
  },
];

/** The slug of one tracked endpoint with at least one detected change. */
function changedEndpointSlug(): string {
  const rows = JSON.parse(
    readFileSync(requireBuilt("data/overview.json"), "utf8"),
  ).endpoints;
  const row = rows.find(
    (r: { methods: string[]; nChanges: number }) => r.methods.length && r.nChanges > 0,
  );
  if (!row) throw new Error("no tracked endpoint with a change in overview.json");
  return row.slug;
}

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

/** Relative hrefs that point at a file the generator did not emit. */
function deadLinks(root: ParentNode, pageDir: string): string[] {
  const missing: string[] = [];
  for (const a of root.querySelectorAll("a[href]")) {
    const href = a.getAttribute("href") ?? "";
    if (!href || href.startsWith("http") || href.startsWith("#")) continue;
    if (!existsSync(resolve(SITE, pageDir, href.split("#")[0]))) missing.push(href);
  }
  return missing;
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
  // happy-dom ships neither the Web Animations API nor ResizeObserver. Nothing on
  // the Overview animates any more, so only the observer needs a stand-in -- and
  // leaving `animate` undefined is what keeps the hero from growing one back.
  globalThis.ResizeObserver ??= class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  } as unknown as typeof ResizeObserver;
});
afterAll(() => GlobalRegistrator.unregister());

describe.each(PAGES)("$name page", (page) => {
  test("renders against the generated data", async () => {
    const html = readFileSync(requireBuilt(page.html), "utf8");
    document.documentElement.innerHTML = html;
    stubFetch(dirname(page.html));

    // the import side effect renders only on the first import; when another test
    // file imported this module first, re-render through the exported init
    const mod = await import(page.entry);
    await mod.init?.();
    await waitFor(page.ready);
  });

  // A sticky element only unpins at the bottom of its containing block. With one
  // flat run of siblings every month's banner stayed pinned under the nav, stacked
  // on top of each other and over the newest change -- which is what shipped.
  test.if(page.name === "changes")("gives each month its own containing block", () => {
    const heads = document.querySelectorAll("#log .mohead");
    expect(heads.length, "no month banners rendered").toBeGreaterThan(1);
    for (const h of heads) {
      expect(h.parentElement?.className, `${h.textContent?.trim()} is not in a group`).toBe(
        "mogroup",
      );
      expect(h.parentElement?.firstElementChild, "banner is not first in its group").toBe(h);
    }
  });

  // A group banner exists to say "these rows are the same company" (the same
  // model, on the provider page); over a lone row it says nothing and just costs
  // a full-width band of surface-3.
  test.if(["model", "provider"].includes(page.name))("banners only over groups of more than one row", () => {
    const banners = document.querySelectorAll("#cmp .grp-h");
    expect(
      banners.length,
      "no banner at all -- the page above no longer has a multi-variant group",
    ).toBeGreaterThan(0);
    for (const b of banners) {
      let rows = 0;
      for (let el = b.nextElementSibling; el; el = el.nextElementSibling) {
        if (!el.classList.contains("row")) break;
        rows++;
      }
      expect(rows, `${b.textContent?.trim()} banners a single row`).toBeGreaterThan(1);
    }
  });

  /** The hero curve is one real endpoint's series around one real changepoint. It
   *  shipped once as six unrelated traces concatenated and autoscaled, which drew a
   *  flat line pinned to the bottom edge with a few spikes -- and said so nowhere. */
  test.if(page.name === "overview")("draws the hero from one named endpoint", () => {
    const line = document.querySelector("#heroTrace path:last-of-type");
    expect(line, "no hero curve rendered").not.toBeNull();
    const xs = [...line!.getAttribute("d")!.matchAll(/[ML]([\d.]+) /g)].map((m) => +m[1]);
    expect(Math.min(...xs), "curve starts short of the left edge").toBe(0);
    expect(Math.max(...xs), "curve stops short of the right edge").toBe(1200);

    // hovering has to say whose data this is: the endpoint, the method, the date
    const href = document.querySelector(".hero-hit")?.getAttribute("href");
    expect(href, "the curve is not attributed to an endpoint").toMatch(/^endpoints\/.+\.html$/);
    const tip = document.getElementById("heroTip")!.textContent ?? "";
    expect(tip, "the hover card names no endpoint").toMatch(/\S+\s*@\s*\S+/);
    expect(tip, "the hover card gives no change date").toMatch(/\d{4}-\d{2}-\d{2}/);
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
    const missing = deadLinks(document, dirname(page.html));
    expect(missing, `dead links: ${missing.slice(0, 5).join(", ")}`).toHaveLength(0);
  });
});

/** Pages the generator renders whole: nothing is fetched, so the only ways they
 *  can break are a missing file and a dead link. */
const STATIC_PAGES = ["methodology.html", "orgs/deepseek.html"];

/** The favicon href is depth-relative, so a page nested one level down needs its
 *  own prefix -- exactly the mistake a root-only check would miss. */
test("the favicon link resolves from every page depth", () => {
  for (const path of [...PAGES.map((p) => p.html), ...STATIC_PAGES]) {
    const html = readFileSync(requireBuilt(path), "utf8");
    const href = html.match(/<link rel="icon" href="([^"]+)"/)?.[1];
    expect(href, `${path} has no favicon link`).toBeDefined();
    expect(existsSync(resolve(SITE, dirname(path), href!)), `${path}: ${href}`).toBe(
      true,
    );
  }
});

describe.each(STATIC_PAGES)("%s", (path) => {
  test("every internal link resolves to a generated file", () => {
    document.documentElement.innerHTML = readFileSync(requireBuilt(path), "utf8");
    const missing = deadLinks(document, dirname(path));
    expect(missing, `dead links: ${missing.slice(0, 5).join(", ")}`).toHaveLength(0);
  });
});

test("the methodology page links out to the blog post and both papers", () => {
  const html = readFileSync(requireBuilt("methodology.html"), "utf8");
  for (const url of [
    "tchauvin.com/change-detection-llm-apis",
    "arxiv.org/abs/2512.03816",
    "arxiv.org/abs/2602.11083",
  ]) {
    expect(html, `${url} is not linked`).toContain(url);
  }
});

test("the org page lists models and links each one", () => {
  document.documentElement.innerHTML = readFileSync(
    requireBuilt("orgs/deepseek.html"),
    "utf8",
  );
  const links = document.querySelectorAll('#modelBody a[href^="../models/"]');
  expect(links.length, "no model rows on the org page").toBeGreaterThan(1);
});

/** The other half of the same bug, and the half no DOM assertion can see: happy-dom
 *  has no layout engine, so guard the CSS precondition instead. `overflow: hidden`
 *  makes .feed the scrollport for the sticky .mohead inside it, which then parks
 *  var(--nav-h) below the feed's own top -- on top of the newest change. */
test("the change feed is not a scroll container", () => {
  const css = readFileSync(join(SITE, "style.css"), "utf8");
  const rule = css.match(/^\.feed \{[^}]*\}/m)?.[0];
  expect(rule, ".feed rule not found in style.css").toBeDefined();
  expect(rule!).not.toContain("overflow");
});

/** Same blind spot, same remedy: the footer was full-bleed while every page above
 *  it sits in a var(--maxw) column, so on the narrow methodology page it ran far
 *  past the text on both sides. Nothing in a layout-less DOM can see that. */
test("the footer is constrained to the content column", () => {
  const css = readFileSync(join(SITE, "style.css"), "utf8");
  const rule = css.match(/^footer\.site \{[^}]*\}/m)?.[0];
  expect(rule, "footer.site rule not found in style.css").toBeDefined();
  expect(rule!, "footer runs full-bleed").toContain("max-width: var(--maxw)");
  expect(rule!, "footer is not centred in the column").toContain("auto");
});
