// The `init` export both makes this a module (so its top-level names don't collide
// with the same names in other bundler entrypoints when type-checked as one tsc
// program) and lets the tests re-render a fresh document without busting the
// module cache -- exactly as overview.ts does.
import {
  Brand,
  FeedItem,
  bindFilterChips,
  esc,
  eventRow,
  monthLabel,
  plural,
  prettyDate,
  providerLabel,
  showLoadError,
} from "./components";

interface ChangeStats {
  total: number;
  lt: number;
  b3it: number;
  endpoints_affected: number;
  providers_involved: number;
  changes_30d: number;
  largest_lt_drift: number | null;
  since: string | null;
  now: string | null;
}

interface TopEndpoint {
  slug: string;
  model: string;
  provider: string;
  brand: Brand;
  variant: string;
  providerSlug: string;
  modelSlug: string;
  n: number;
  last: string;
}

interface ChangesData {
  stats: ChangeStats;
  items: FeedItem[];
  top_endpoints: TopEndpoint[];
}

const RECENT_DAYS = 90;

export async function init(): Promise<void> {
  let D: ChangesData;
  try {
    const res = await fetch("data/changes_page.json");
    if (!res.ok) throw new Error(`changes_page.json: HTTP ${res.status}`);
    D = await res.json();
  } catch (err) {
    showLoadError("lede", "the change log");
    throw err;
  }
  const S = D.stats;

  const ledeEl = document.getElementById("lede");
  if (ledeEl) {
    ledeEl.innerHTML =
      `The complete log: <b>${plural(S.total, "detected change")}</b> across ` +
      `${plural(S.endpoints_affected, "endpoint")}` +
      (S.since ? ` since ${prettyDate(S.since)}` : "") +
      ` — ${S.lt} from logprob tracking (LT), ${S.b3it} from border inputs (B3IT). ` +
      `Each entry records that an endpoint's outputs moved away from its own baseline on that date.`;
  }

  const summaryEl = document.getElementById("summary");
  if (summaryEl) {
    const drift =
      S.largest_lt_drift === null
        ? "—"
        : `${S.largest_lt_drift.toFixed(2)}<small> nats</small>`;
    summaryEl.innerHTML = `
      <div class="s"><div class="v">${S.total}</div><div class="k">Changes</div></div>
      <div class="s"><div class="v">${S.endpoints_affected}</div><div class="k">Endpoints affected</div></div>
      <div class="s"><div class="v">${S.providers_involved}</div><div class="k">Providers involved</div></div>
      <div class="s"><div class="v">${S.changes_30d}</div><div class="k">Last 30 days</div></div>
      <div class="s"><div class="v">${drift}</div><div class="k">Largest LT drift</div></div>`;
  }

  const topEl = document.getElementById("topEndpoints");
  if (topEl) {
    const most = D.top_endpoints[0]?.n || 1;
    topEl.innerHTML = D.top_endpoints
      .map((e, i) => {
        const cells = `<span class="rk">${i + 1}</span>
        <span class="pv">${esc(e.model)}<small>@ ${providerLabel(e.brand, e.variant, "")}</small></span>
        <span class="rbar"><span style="width:${((e.n / most) * 100).toFixed(0)}%"></span><b>${e.n}</b></span>
        <span class="meta">last<br><b>${esc(e.last)}</b></span>`;
        // no modelSlug: the endpoint has left the fleet and has no model page
        // (feed.py leaves its slugs empty), so the row is text, not a link to a 404
        return e.modelSlug
          ? `<a class="brow" href="models/${esc(e.modelSlug)}.html">${cells}</a>`
          : `<div class="brow">${cells}</div>`;
      })
      .join("");
  }

  const items = D.items;
  const filters = new Set<string>();
  const qEl = document.getElementById("q") as HTMLInputElement | null;
  const logEl = document.getElementById("log");
  if (!qEl || !logEl) return;

  // "Large" is per method — nats and total variation are different units — so the
  // verdict comes from feed.py's own alert thresholds via sevKey, never re-derived here.
  const isLarge = (e: FeedItem): boolean => e.sevKey === "alert";

  function render(): void {
    const q = qEl!.value.trim().toLowerCase();
    // LT and B3IT are the whole population between them, so filtering on both — or
    // on neither — is the same as no method filter at all.
    const oneMethod = filters.has("lt") !== filters.has("b3it");
    const rows = items.filter((e) => {
      if (q && !`${e.model} ${e.provider} ${e.org}`.toLowerCase().includes(q)) return false;
      if (oneMethod && !filters.has(e.method)) return false;
      if (filters.has("recent") && e.daysAgo > RECENT_DAYS) return false;
      if (filters.has("big") && !isLarge(e)) return false;
      return true;
    });

    // Each month gets its own element rather than a flat run of siblings: a sticky
    // .mohead only unpins at the bottom of its containing block, so sharing one
    // would pile every month's banner up under the nav over the same change row.
    const groups: { month: string; events: FeedItem[] }[] = [];
    for (const e of rows) {
      const m = e.date.slice(0, 7);
      let last = groups[groups.length - 1];
      // items arrive sorted by date desc (feed.py), so a month is always contiguous
      if (!last || last.month !== m) {
        last = { month: m, events: [] };
        groups.push(last);
      }
      last.events.push(e);
    }
    const html = groups
      .map(
        (g) => `<section class="mogroup"><div class="mohead"><span>${monthLabel(g.month)}</span>
          <span class="c">${plural(g.events.length, "change")}</span></div>${g.events.map(eventRow).join("")}</section>`,
      )
      .join("");
    logEl!.innerHTML = html || '<div class="empty">No changes match these filters.</div>';

    const countEl = document.getElementById("logCount");
    if (countEl) {
      countEl.textContent = `${rows.length} of ${items.length} changes`;
    }
  }

  qEl.addEventListener("input", render);
  const chipsEl = document.getElementById("chips");
  if (chipsEl) bindFilterChips(chipsEl, filters, render);
  render();
}

init();
