// `export {}` makes this a module so its top-level names (DATA, init, ...) don't
// collide with the same names in other bundler-entrypoint scripts (endpoint.ts)
// when type-checked together as one tsc program.
export {};

import {
  FeedItem,
  esc,
  eventRow,
  monthLabel,
  plural,
  prettyDate,
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

interface MonthBucket {
  month: string;
  lt: number;
  b3it: number;
}

interface TopEndpoint {
  slug: string;
  model: string;
  provider: string;
  providerSlug: string;
  modelSlug: string;
  n: number;
  last: string;
}

interface ChangesData {
  stats: ChangeStats;
  items: FeedItem[];
  months: MonthBucket[];
  top_endpoints: TopEndpoint[];
}

const BARS_H = 118; // px of the .hist .bars box the tallest month fills
const RECENT_DAYS = 90;
// "Large" is per method: the two magnitudes are in different units (nats vs total
// variation). Mirrors feed.py's alert thresholds.
const LARGE_LT = 0.8;
const LARGE_B3IT = 0.6;

async function init(): Promise<void> {
  const res = await fetch("data/changes_page.json").catch(() => null);
  const D: ChangesData | null = res && res.ok ? await res.json() : null;
  if (!D) return;
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

  // month histogram — B3IT stacked over LT, scaled to the tallest month
  const maxMonth = Math.max(1, ...D.months.map((m) => m.lt + m.b3it));
  const histEl = document.getElementById("hist");
  if (histEl) {
    const bars = D.months
      .map((m) => {
        const total = m.lt + m.b3it;
        const h = (n: number): string => ((n / maxMonth) * BARS_H).toFixed(1);
        return `<div class="mo" data-m="${esc(m.month)}" title="${monthLabel(m.month)}: ${m.lt} LT, ${m.b3it} B3IT">
          ${total ? `<span class="n">${total}</span>` : ""}
          ${m.b3it ? `<div class="b3" style="height:${h(m.b3it)}px"></div>` : ""}
          ${m.lt ? `<div class="lt ${m.b3it ? "" : "bottom"}" style="height:${h(m.lt)}px"></div>` : ""}
        </div>`;
      })
      .join("");
    // Januaries and Julys are the readable anchors on a month-per-column axis
    const axis = D.months
      .map((m) => {
        const anchor = m.month.endsWith("-01") || m.month.endsWith("-07");
        return `<span class="${anchor ? "q" : ""}">${esc(m.month.slice(5))}</span>`;
      })
      .join("");
    histEl.innerHTML = `<div class="bars">${bars}</div><div class="xaxis">${axis}</div>`;
  }

  const topEl = document.getElementById("topEndpoints");
  if (topEl) {
    const most = D.top_endpoints[0]?.n ?? 1;
    topEl.innerHTML = D.top_endpoints
      .map(
        (e, i) => `<a class="brow" href="models/${esc(e.modelSlug)}.html">
        <span class="rk">${i + 1}</span>
        <span class="pv">${esc(e.model)}<small>@ ${esc(e.provider)}</small></span>
        <span class="rbar"><span style="width:${((e.n / most) * 100).toFixed(0)}%"></span><b>${e.n}</b></span>
        <span class="meta">last<br><b>${esc(e.last)}</b></span></a>`
      )
      .join("");
  }

  const items = D.items;
  const filters = new Set<string>();
  let month: string | null = null;
  const qEl = document.getElementById("q") as HTMLInputElement | null;
  const logEl = document.getElementById("log");
  if (!qEl || !logEl) return;

  function isLarge(e: FeedItem): boolean {
    if (e.magnitude === null) return false;
    return e.magnitude >= (e.method === "lt" ? LARGE_LT : LARGE_B3IT);
  }

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
      if (month !== null && e.date.slice(0, 7) !== month) return false;
      return true;
    });

    const perMonth = new Map<string, number>();
    for (const e of rows) {
      const m = e.date.slice(0, 7);
      perMonth.set(m, (perMonth.get(m) ?? 0) + 1);
    }
    let html = "";
    let current: string | null = null;
    for (const e of rows) {
      const m = e.date.slice(0, 7);
      if (m !== current) {
        current = m;
        html += `<div class="mohead"><span>${monthLabel(m)}</span>
          <span class="c">${plural(perMonth.get(m)!, "change")}</span></div>`;
      }
      html += eventRow(e);
    }
    logEl!.innerHTML = html || '<div class="empty">No changes match these filters.</div>';

    const countEl = document.getElementById("logCount");
    if (countEl) {
      countEl.textContent =
        `${rows.length} of ${items.length} changes` +
        (month !== null ? ` · ${monthLabel(month)}` : "");
    }
    const selEl = document.getElementById("monthSel");
    if (selEl) {
      selEl.textContent =
        month !== null ? `filtered to ${monthLabel(month)} — click again to clear` : "";
    }
    document.querySelectorAll<HTMLElement>("#hist .mo").forEach((el) => {
      el.classList.toggle("dim", month !== null && el.dataset.m !== month);
    });
  }

  qEl.addEventListener("input", render);
  document.getElementById("chips")?.addEventListener("click", (e) => {
    const chip = (e.target as HTMLElement).closest(".chip") as HTMLElement | null;
    if (!chip) return;
    const f = chip.dataset.f!;
    if (filters.has(f)) { filters.delete(f); chip.classList.remove("on"); }
    else { filters.add(f); chip.classList.add("on"); }
    render();
  });
  histEl?.addEventListener("click", (e) => {
    const col = (e.target as HTMLElement).closest(".mo") as HTMLElement | null;
    if (!col) return;
    const m = col.dataset.m!;
    month = month === m ? null : m;
    render();
  });
  render();
}

init();
