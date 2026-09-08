// Prerender the front page: run its own script (src/overview.ts) in happy-dom over
// the generated index.html and write the rendered sections back into it, so the
// page is filled at first paint and readable with scripts off. In the browser
// the same script runs again over this markup -- every section is an innerHTML
// write, so the second pass is idempotent -- for the hover cards, the popovers and
// the live "last update" line. Only the front page: its data is inlined (home.py);
// the other pages fetch theirs.
//
// Run after the site generator: `bun run prerender` (package.json), as the
// Makefile and deploy-pages.yml do.
import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const path = resolve(import.meta.dir, "..", "index.html");

GlobalRegistrator.register();
document.documentElement.innerHTML = readFileSync(path, "utf8");
const { init } = await import("../src/overview");
await init();

// Nothing that ages, and nothing a zero-size layout computed: the freshness line
// is the browser's to write, and the hero layers are sized against a real viewport.
document.getElementById("freshness")!.textContent = "";
for (const id of ["heroTrace", "heroHit"]) document.getElementById(id)?.removeAttribute("style");

for (const sel of ["#telemetry .stat", "#feed .event", "#provPlot .rrow", "#dirBody tr"]) {
  if (!document.querySelector(sel)) {
    console.error(`prerender: ${sel} rendered nothing`);
    process.exit(1);
  }
}
writeFileSync(path, "<!DOCTYPE html>\n" + document.documentElement.outerHTML + "\n");
console.log("Prerendered index.html");
await GlobalRegistrator.unregister();
process.exit(0);
