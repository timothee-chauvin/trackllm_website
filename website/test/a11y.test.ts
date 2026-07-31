/**
 * Keyboard operability of the interactive controls, against the generated site --
 * same harness as status_pages.test.ts, so these also fail when a template stops
 * emitting the ARIA the scripts keep in sync.
 *
 * Note on the month bars: they are native <button>s, which is what earns them
 * Enter/Space activation in a real browser. happy-dom does not synthesize a click
 * from a key event, so those tests assert the element type (the guarantee) and
 * drive the filter through click (what a browser dispatches on Enter).
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

/** Press a key on an element the way a keyboard user would reach it. */
function press(el: Element, key: string): KeyboardEvent {
  const ev = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true });
  el.dispatchEvent(ev);
  return ev;
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

const shownCount = (): number =>
  Number(/^(\d+) of /.exec(document.getElementById("dirFoot")!.textContent ?? "")?.[1]);

describe("filter chips", () => {
  test("every chip is a button to the keyboard, with its state exposed", async () => {
    await renderOverview();
    const chips = document.querySelectorAll<HTMLElement>(".chip");
    expect(chips.length).toBeGreaterThan(0);
    for (const c of chips) {
      expect(c.getAttribute("role"), c.textContent!).toBe("button");
      expect(c.getAttribute("tabindex"), c.textContent!).toBe("0");
      // the one chip that starts on ("Tracked") must say so from the markup
      expect(c.getAttribute("aria-pressed"), c.textContent!).toBe(
        String(c.classList.contains("on")),
      );
    }
  });

  test("Enter and Space toggle a chip, and aria-pressed follows", async () => {
    await renderOverview();
    const tracked = document.querySelector<HTMLElement>('#chips .chip[data-st="tracked"]')!;
    const all = shownCount();

    press(tracked, "Enter"); // tracked off: no status constraint left
    expect(tracked.getAttribute("aria-pressed")).toBe("false");
    expect(tracked.classList.contains("on")).toBe(false);
    expect(shownCount()).toBeGreaterThan(all);

    press(tracked, " ");
    expect(tracked.getAttribute("aria-pressed")).toBe("true");
    expect(shownCount()).toBe(all);
  });

  test("Space activates rather than scrolling the page", async () => {
    await renderOverview();
    const chip = document.querySelector<HTMLElement>('#chips .chip[data-f="lt"]')!;
    expect(press(chip, " ").defaultPrevented).toBe(true);
    // an unrelated key is left alone
    expect(press(chip, "a").defaultPrevented).toBe(false);
    expect(chip.getAttribute("aria-pressed")).toBe("true");
  });

  test("the second chip group on the Overview is keyboard-operable too", async () => {
    await renderOverview();
    const chip = document.querySelector<HTMLElement>('#provChips .chip[data-f="changed"]')!;
    const foot = document.getElementById("provFoot")!;
    const before = foot.textContent;
    press(chip, "Enter");
    expect(chip.getAttribute("aria-pressed")).toBe("true");
    expect(foot.textContent).not.toBe(before);
  });
});

describe("sortable column headers", () => {
  const sortable = (): HTMLElement[] => [
    ...document.querySelectorAll<HTMLElement>("#dirBody")[0].closest("table")!
      .querySelectorAll<HTMLElement>("th[data-sort]"),
  ];

  test("are focusable and announce the current sort", async () => {
    await renderOverview();
    const headers = sortable();
    expect(headers.length).toBeGreaterThan(1);
    for (const th of headers) {
      expect(th.getAttribute("tabindex")).toBe("0");
      expect(th.getAttribute("scope")).toBe("col");
    }
    // the directory opens sorted by nChanges, high to low
    const sorted = headers.filter((th) => th.getAttribute("aria-sort") !== "none");
    expect(sorted.map((th) => th.dataset.sort)).toEqual(["nChanges"]);
    expect(sorted[0].getAttribute("aria-sort")).toBe("descending");
  });

  test("Enter sorts, and aria-sort moves to the column that now owns it", async () => {
    await renderOverview();
    const model = sortable().find((th) => th.dataset.sort === "model")!;
    const changes = sortable().find((th) => th.dataset.sort === "nChanges")!;

    press(model, "Enter");
    expect(model.getAttribute("aria-sort")).toBe("ascending");
    expect(changes.getAttribute("aria-sort")).toBe("none");
    const firstAsc = document.querySelector("#dirBody .model-cell")!.textContent;

    press(model, " "); // same column again just flips the direction
    expect(model.getAttribute("aria-sort")).toBe("descending");
    expect(document.querySelector("#dirBody .model-cell")!.textContent).not.toBe(firstAsc);
  });

  test("the provider table keeps its own sort state", async () => {
    await renderOverview();
    const psorted = [...document.querySelectorAll<HTMLElement>("th[data-psort]")].filter(
      (th) => th.getAttribute("aria-sort") !== "none",
    );
    expect(psorted.map((th) => th.dataset.psort)).toEqual(["lt_rate"]);
  });
});

describe("changes-page month histogram", () => {
  async function renderChanges(): Promise<void> {
    document.documentElement.innerHTML = readFileSync(
      requireBuilt("changes.html"),
      "utf8",
    );
    stubFetch(".");
    await (await import("../src/changes")).init();
  }

  /** The month's two counts, from its accessible name. */
  const counts = (b: Element): number[] =>
    [...(b.getAttribute("aria-label") ?? "").matchAll(/(\d+) (?:LT|B3IT)/g)].map((m) => +m[1]);

  test("each month is a named button carrying its counts", async () => {
    await renderChanges();
    const bars = document.querySelectorAll<HTMLElement>("#hist .mo");
    expect(bars.length).toBeGreaterThan(1);
    for (const b of bars) {
      expect(b.tagName).toBe("BUTTON"); // hence tabbable and Enter/Space-activatable
      expect(b.getAttribute("type")).toBe("button");
      const label = b.getAttribute("aria-label") ?? "";
      expect(label, `${b.dataset.m} has no counts in its name`).toMatch(
        /^\w{3} '\d\d: \d+ LT, \d+ B3IT/,
      );
      expect(b.getAttribute("aria-pressed")).toBe("false");
    }
  });

  test("activating a bar filters the log and marks itself pressed", async () => {
    await renderChanges();
    const bars = [...document.querySelectorAll<HTMLElement>("#hist .mo")];
    const bar = bars.find((b) => counts(b).some((n) => n > 0))!;
    expect(bar, "no month with any change to filter to").toBeDefined();
    const count = document.getElementById("logCount")!;
    const before = count.textContent;

    bar.click(); // what Enter/Space dispatches on a native button
    expect(bar.getAttribute("aria-pressed")).toBe("true");
    expect(count.textContent).not.toBe(before);
    expect(bars.filter((b) => b.getAttribute("aria-pressed") === "true")).toEqual([bar]);

    bar.click(); // pressing the same month again clears the filter
    expect(bar.getAttribute("aria-pressed")).toBe("false");
    expect(count.textContent).toBe(before);
  });
});

describe("accessible names", () => {
  test("the hero curve link names the endpoint it opens", async () => {
    await renderOverview();
    const hit = document.querySelector(".hero-hit")!;
    const href = hit.getAttribute("href")!;
    const label = hit.getAttribute("aria-label") ?? "";
    expect(label, "the hero link has no name of its own").toContain("@");
    expect(hit.textContent!.trim(), "the link's own content is an invisible path").toBe("");
    expect(href).toMatch(/^endpoints\/.+\.html$/);
  });

  test("every search input is named", () => {
    for (const path of ["index.html", "changes.html", "providers/chutes.html"]) {
      const html = readFileSync(requireBuilt(path), "utf8");
      const inputs = [...html.matchAll(/<input [^>]*>/g)].map((m) => m[0]);
      expect(inputs.length, `${path} has no search input`).toBeGreaterThan(0);
      for (const i of inputs) expect(i, path).toContain("aria-label=");
    }
  });
});

/** No layout engine here either (see smoke.test.ts): guard the token instead.
 *  --text-dim carries 0.6-0.82rem body text, so it owes 4.5:1 against the
 *  surfaces it is drawn on -- the lightest one in dark mode, the darkest in light. */
describe("--text-dim contrast", () => {
  const css = readFileSync(join(SITE, "style.css"), "utf8");

  function luminance(hex: string): number {
    const lin = (c: number): number => {
      const s = c / 255;
      return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
    };
    const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  }
  function ratio(a: string, b: string): number {
    const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return (hi + 0.05) / (lo + 0.05);
  }
  /** The `n`th palette block's value for `token` (dark first, then both light copies). */
  function tokenValues(token: string): string[] {
    return [...css.matchAll(new RegExp(`--${token}: (#[0-9A-Fa-f]{6});`, "g"))].map(
      (m) => m[1],
    );
  }

  test("clears AA on the worst background of each theme", () => {
    const dims = tokenValues("text-dim");
    expect(dims.length, "expected one dark and two light palette copies").toBe(3);
    const [dark, light1, light2] = dims;
    expect(light1, "the two light palette copies have drifted apart").toBe(light2);

    // dark theme: the lightest surface a dim label sits on is --surface-3
    for (const bg of ["#0B0F14", "#12181F", "#171F28", "#1E2731"]) {
      expect(ratio(dark, bg), `dark --text-dim on ${bg}`).toBeGreaterThanOrEqual(4.5);
    }
    // light theme: the darkest is --surface
    for (const bg of ["#FAFCFD", "#F1F5F8", "#FFFFFF"]) {
      expect(ratio(light1, bg), `light --text-dim on ${bg}`).toBeGreaterThanOrEqual(4.5);
    }
  });
});

describe("navigation", () => {
  test("the current page's nav link is marked, and only it", () => {
    const cases: [string, string][] = [
      ["changes.html", "changes.html"],
      ["spend.html", "spend.html"],
      ["methodology.html", "methodology.html"],
    ];
    for (const [page, href] of cases) {
      document.documentElement.innerHTML = readFileSync(requireBuilt(page), "utf8");
      const current = document.querySelectorAll(".nav-links a[aria-current]");
      expect(current.length, `${page} marks ${current.length} nav links`).toBe(1);
      expect(current[0].getAttribute("href")).toBe(href);
      expect(current[0].classList.contains("active")).toBe(true);
    }
  });

  test("a page with no nav entry of its own marks none", () => {
    for (const page of ["index.html", "orgs/deepseek.html"]) {
      document.documentElement.innerHTML = readFileSync(requireBuilt(page), "utf8");
      expect(document.querySelectorAll(".nav-links a[aria-current]").length, page).toBe(0);
    }
  });
});

/** .table-wrap clips (overflow: hidden), so a table wide enough to overflow it
 *  disappears at the edge unless it sits in the .table-scroll port. */
test("every directory table sits in a scroll port", () => {
  for (const path of ["index.html", "providers/chutes.html", "orgs/deepseek.html"]) {
    document.documentElement.innerHTML = readFileSync(requireBuilt(path), "utf8");
    const tables = document.querySelectorAll("table.dir");
    expect(tables.length, `${path} has no directory table`).toBeGreaterThan(0);
    for (const t of tables) {
      expect(t.closest(".table-scroll"), `${path}: a table.dir is not in .table-scroll`)
        .not.toBeNull();
    }
  }
});
