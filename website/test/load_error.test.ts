/**
 * A failed data fetch must render an explicit "failed to load" card — never a
 * silent skeleton of empty mounts, and never a factual "no data" claim (PR #48
 * rule). Same harness as smoke.test.ts: real generated HTML, so these also need
 * a build first (`make build`).
 */
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { showLoadError } from "../src/components";

const SITE = resolve(import.meta.dir, "..");

function requireBuilt(path: string): string {
  const full = join(SITE, path);
  if (!existsSync(full)) {
    throw new Error(`${path} is missing -- run \`make build\` before these tests`);
  }
  return full;
}

/** Every fetch fails the way a dropped connection does. */
function stubNetworkFailure(): void {
  // Promise<never> does not overlap typeof fetch, hence the unknown hop
  globalThis.fetch = (async (_input: string | URL) => {
    throw new TypeError("Failed to fetch");
  }) as unknown as typeof fetch;
}

/** Every fetch reaches a server that answers 500. */
function stubServerError(): void {
  globalThis.fetch = (async (_input: string | URL) =>
    ({ ok: false, status: 500, json: async () => null }) as Response) as typeof fetch;
}

/** Serve fetches from disk, as smoke.test.ts does. */
function stubDisk(pageDir: string): void {
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

beforeAll(async () => {
  GlobalRegistrator.register();
  globalThis.ResizeObserver ??= class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  } as unknown as typeof ResizeObserver;
  // The first import of each entrypoint runs its top-level init(); give that run
  // a real page and working data so it settles cleanly (an import under a failing
  // fetch would surface as an unhandled rejection outside any test).
  document.documentElement.innerHTML = readFileSync(requireBuilt("index.html"), "utf8");
  stubDisk(".");
  await import("../src/overview");
  await import("../src/provider"); // no #providerData on index.html: returns early
  await import("../src/model"); // no #cmp: returns early
  await new Promise((r) => setTimeout(r, 30));
});
afterAll(() => GlobalRegistrator.unregister());

describe("showLoadError", () => {
  test("renders an explicit failure card into the mount", () => {
    document.body.innerHTML = '<div id="m">old</div>';
    showLoadError("m", "test data");
    const card = document.querySelector("#m .empty.load-error");
    expect(card).not.toBeNull();
    expect(card!.textContent).toContain("Failed to load test data");
    expect(card!.textContent).toContain("not an absence of data");
  });

  test("escapes the description and is a no-op without the mount", () => {
    document.body.innerHTML = '<div id="m"></div>';
    showLoadError("m", "<script>");
    expect(document.getElementById("m")!.innerHTML).not.toContain("<script>");
    expect(() => showLoadError("gone", "x")).not.toThrow();
  });
});

describe("pages under a failed fetch", () => {
  test("provider page shows the card, not a skeleton", async () => {
    document.documentElement.innerHTML = readFileSync(
      requireBuilt("providers/chutes.html"),
      "utf8",
    );
    stubNetworkFailure();
    const { init } = await import("../src/provider");
    await expect(init()).rejects.toThrow();
    expect(document.querySelector("#lede .load-error")).not.toBeNull();
  });

  test("model page shows the card, never the 'no monitoring data' claim", async () => {
    document.documentElement.innerHTML = readFileSync(
      requireBuilt("models/qwen2fqwen3-coder.html"),
      "utf8",
    );
    stubServerError();
    const { init } = await import("../src/model");
    await expect(init()).rejects.toThrow("HTTP 500");
    expect(document.querySelector("#cmp .load-error")).not.toBeNull();
    expect(document.body.textContent).not.toContain("No monitoring data available");
  });

  test("overview shows the card and drops the hero shell", async () => {
    document.documentElement.innerHTML = readFileSync(requireBuilt("index.html"), "utf8");
    stubNetworkFailure();
    const { init } = await import("../src/overview");
    await expect(init()).rejects.toThrow();
    expect(document.querySelector("#telemetry .load-error")).not.toBeNull();
    // no live dot and no empty trace layers above the error card
    expect(document.getElementById("eyebrow")).toBeNull();
    expect(document.querySelector(".hero-trace")).toBeNull();
  });
});
