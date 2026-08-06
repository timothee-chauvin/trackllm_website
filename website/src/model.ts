// The `init` export both makes this a module (so its top-level names don't collide
// with other bundler entrypoints when type-checked as one tsc program) and lets the
// smoke tests re-render a fresh document without busting the module cache.
import { esc, headlineBadge, bindTips, plural, prettyDate, showLoadError } from "./components";
import { type TimelineData, hasTimeline, renderTimeline } from "./timeline";

interface ModelData extends TimelineData {
  model: string;
  org: string;
  n_endpoints: number;
  n_providers: number;
  n_endpoints_total: number;
  n_changed: number;
  max_drift: number;
  headline: string;
  status_summary: string;
}

export async function init(): Promise<void> {
  const cmpEl = document.getElementById("cmp");
  const slugEl = document.getElementById("model-slug");
  if (!cmpEl || !slugEl) return;

  const slug: string = JSON.parse(slugEl.textContent || '""');
  let D: ModelData;
  try {
    const res = await fetch(`../data/models/${slug}.json`);
    if (!res.ok) throw new Error(`models/${slug}.json: HTTP ${res.status}`);
    D = await res.json();
  } catch (err) {
    // a fetch failure must not read as the factual claim "no data yet"
    showLoadError("cmp", "this model's data");
    throw err;
  }

  if (!D.endpoints.length) {
    cmpEl.innerHTML = `<div style="padding:2rem 1rem;color:var(--text-dim);font-size:0.85rem">No monitoring data available yet for this model.</div>`;
    return;
  }

  const tracked = D.endpoints.filter((e) => e.methods.length);
  // a model can be all catalog, no series: its page is badge rows, not a timeline
  if (!hasTimeline(D)) {
    document.getElementById("cmpDesc")?.remove();
    document.getElementById("cmpLegend")?.remove();
    // there is no drift to show, only per-endpoint status rows
    const title = document.getElementById("cmpTitle");
    if (title) title.textContent = "Endpoints";
  }

  const ledeEl = document.getElementById("lede");
  if (ledeEl) {
    // n_endpoints counts serving endpoints, n_providers the companies behind them:
    // saying "providers" for the larger number would contradict the groups below.
    ledeEl.innerHTML = tracked.length
      ? `Served by <b>${D.n_providers}</b> ${D.n_providers === 1 ? "provider" : "providers"}` +
        ` on ${plural(D.n_endpoints, "tracked endpoint")}. ` +
        `<span class="hl">${D.n_changed}</span> of those ${D.n_changed === 1 ? "shows" : "show"}` +
        ` at least one detected change since launch.` +
        ` ${esc(D.status_summary)} across the catalog.`
      : `This model is not tracked: ${esc(D.status_summary)}. Each endpoint below says why.`;
  }
  const summaryEl = document.getElementById("summary");
  if (summaryEl) {
    summaryEl.innerHTML = tracked.length
      ? `
      <div class="s"><div class="v">${D.n_endpoints}</div><div class="k">Endpoints</div></div>
      <div class="s"><div class="v" style="color:var(--changed)">${D.n_changed}</div><div class="k">With changes</div></div>
      <div class="s"><div class="v">${D.changes.length}</div><div class="k">Changes total</div></div>
      <div class="s"><div class="v">${prettyDate(D.date_min)} – ${prettyDate(D.date_max)}</div><div class="k">Monitored</div></div>`
      : `
      <div class="s"><div class="v">${D.n_endpoints_total}</div><div class="k">Catalog endpoints</div></div>
      <div class="s"><div class="v">${headlineBadge(D.headline)}</div><div class="k">Status</div></div>`;
  }

  renderTimeline(cmpEl, D, {
    name: (ep) => ep.provider,
    changeName: (c) => c.provider,
    group: (ep) => ({
      key: ep.base,
      label: ep.base,
      href: `../providers/${ep.providerSlug}.html`,
      page: "provider page",
    }),
  });
  // document.body, not cmpEl: the status badge up in #summary (headlineBadge, above)
  // wants the same popover as the ones in #cmp, and one binding covers both.
  bindTips(document.body);
}

init();
