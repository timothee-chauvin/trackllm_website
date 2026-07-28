// `export {}` makes this a module so its top-level names (DATA, fmtInt, init, ...)
// don't collide with the same names in other bundler-entrypoint scripts (endpoint.ts)
// when type-checked together as one tsc program.
export {};

import {
  B3IT_CAP,
  FeedItem,
  LT_CAP,
  MIN_ENDPOINT_YEARS,
  esc,
  eventRow,
  methodBadges,
  plural,
  rateBar,
  sparkline,
  statusPill,
  volGrid,
} from "./components";

interface Stats {
  endpoints: number;
  providers: number;
  provider_companies: number;
  models: number;
  orgs: number;
  changes_total: number;
  changes_lt: number;
  changes_b3it: number;
  active: number;
  changed_endpoints: number;
  changes_30d: number;
  lt_endpoints: number;
  b3it_endpoints: number;
  b3it_monitoring: number;
  b3it_since: string | null;
  queries: number;
  since: string | null;
  spend_cumulative: number;
  now: string | null;
}

/** One provider *company*, with its serving variants pooled (provider.py::overview_rows).
 *  `lt_rate` and `lt_ci` are null together, and that null is the "not enough
 *  monitoring" state — never a rate of zero, and never recomputed here. */
interface ProviderRate {
  name: string;
  slug: string;
  n_endpoints: number;
  n_models: number;
  n_variants: number;
  lt_years: number;
  lt_changes: number;
  lt_rate: number | null;
  lt_ci: [number, number] | null;
  b3it_endpoints: number;
  b3it_years: number;
  last_change: string | null;
}

type EndpointStatus = "stable" | "changed" | "retired";

interface EndpointRow {
  slug: string;
  model: string;
  modelSlug: string;
  org: string;
  provider: string;
  providerSlug: string;
  methods: string[];
  status: EndpointStatus;
  stableDays: number | null;
  nChanges: number;
  trace: number[];
}

type SortKey = "model" | "provider" | "status" | "nChanges" | "stableDays";
type ProviderSortKey = "name" | "n_endpoints" | "lt_rate" | "last_change";

const BOARD_SIZE = 5;
const QUIET_MIN_YEARS = 1; // a "nothing detected yet" board entry needs real exposure

(async function (): Promise<void> {
  const DATA: {
    stats: Stats;
    feed: FeedItem[];
    providers: ProviderRate[];
    endpoints: EndpointRow[];
  } = await (await fetch("data/overview.json")).json();
  const S = DATA.stats;
  const fmtInt = (n: number): string => n.toLocaleString("en-US");
  const fmtM = (n: number): string =>
    n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "k" : "" + n;

  // hero trace: real drift series concatenated, autoscaled — the only place that
  // wants a min/max fit rather than the fixed LT_CAP/B3IT_CAP sparkline scale.
  function sparkPath(vals: number[], w: number, h: number, pad: number): string {
    const min = Math.min(...vals), max = Math.max(...vals);
    const rng = max - min || 1;
    const step = (w - pad * 2) / (vals.length - 1);
    return vals.map((v, i) => {
      const x = pad + i * step, y = pad + (h - pad * 2) * (1 - (v - min) / rng);
      return (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
    }).join(" ");
  }

  (function () {
    const svg = document.getElementById("heroTrace") as unknown as SVGSVGElement;
    let vals: number[] = [];
    DATA.endpoints.filter(e => e.trace && e.trace.length > 10).slice(0, 6).forEach(e => vals.push(...e.trace));
    if (vals.length < 20) vals = [0.1, 0.3, 0.2, 0.4, 0.25];
    const d = sparkPath(vals, 1200, 300, 8);
    svg.innerHTML = `<defs><linearGradient id="hg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="var(--accent)" stop-opacity="0"/><stop offset="0.5" stop-color="var(--accent)" stop-opacity="0.85"/>
      <stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></linearGradient></defs>
      <path d="${d}" fill="none" stroke="url(#hg)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`;
    const path = svg.querySelector("path") as SVGPathElement, len = path.getTotalLength();
    path.style.strokeDasharray = String(len); path.style.strokeDashoffset = String(len);
    path.animate([{ strokeDashoffset: len }, { strokeDashoffset: 0 }], { duration: 2400, easing: "ease-out", fill: "forwards" });
  })();

  document.getElementById("eyebrow")!.innerHTML = `<span class="dot"></span> Continuously monitoring ${S.active} active endpoints`;
  const stats = [
    { label: "Endpoints", value: fmtInt(S.endpoints), sub: `${S.active} active · ${S.endpoints - S.active} retired` },
    { label: "Models tracked", value: fmtInt(S.models), sub: `across ${S.orgs} orgs` },
    { label: "Providers", value: fmtInt(S.provider_companies), sub: `${S.providers} serving variants` },
    { label: "Queries logged", value: fmtM(S.queries), sub: `since ${S.since}` },
    { label: "Changes detected", value: fmtInt(S.changes_total), sub: `${S.changes_lt} LT · ${S.changes_b3it} B3IT` },
  ];
  document.getElementById("telemetry")!.innerHTML = stats.map(s =>
    `<div class="stat"><div class="label">${s.label}</div><div class="value">${s.value}</div><div class="sub">${s.sub}</div></div>`).join("");
  const perM = S.spend_cumulative / (S.queries / 1e6);
  document.getElementById("cap")!.innerHTML =
    `Cheap enough to run continuously — <b>${fmtM(S.queries)}</b> logprob queries for <b>$${S.spend_cumulative.toFixed(2)}</b> total (~$${perM.toFixed(2)}/M). ` +
    `LT on ${S.lt_endpoints} endpoints since ${S.since}; B3IT on ${S.b3it_endpoints} since ${S.b3it_since} (${S.b3it_monitoring} still active).`;

  document.getElementById("feed")!.innerHTML = DATA.feed.map(eventRow).join("");

  // ---- providers: two ranked boards, then the full sortable table ----
  const provs = DATA.providers;
  // One scale for every bar drawn on this page. Spans only what is drawn — the
  // rates and their interval upper bounds — so no band is clipped; zero-rate rows
  // render as text, so their (rule-of-three) ceilings must not set the scale.
  const drawn = provs.filter(p => p.lt_rate !== null && p.lt_rate > 0);
  const maxRate = Math.max(
    1,
    ...drawn.map(p => p.lt_rate!),
    ...drawn.map(p => p.lt_ci?.[1] ?? 0)
  );

  function boardRow(p: ProviderRate, i: number): string {
    return `<a class="brow" href="providers/${esc(p.slug)}.html">
      <span class="rk">${i + 1}</span>
      <span class="pv">${esc(p.name)}<small>${plural(p.n_endpoints, "endpoint")} · ${plural(p.n_variants, "variant")}</small></span>
      ${rateBar(p.lt_years, p.lt_rate, p.lt_ci, maxRate)}
      <span class="meta"><b>${p.lt_changes}</b> chg<br>${p.lt_years.toFixed(1)} ep-yr</span></a>`;
  }
  function board(title: string, tag: string, desc: string, rows: ProviderRate[]): string {
    return `<div class="board"><h3>${title} <em>${tag}</em></h3><p>${desc}</p>
      ${rows.map(boardRow).join("") || '<div class="empty">Nothing to rank yet.</div>'}</div>`;
  }

  const drifty = provs
    .filter(p => p.lt_rate !== null && p.lt_changes > 0)
    .sort((a, b) => b.lt_rate! - a.lt_rate!)
    .slice(0, BOARD_SIZE);
  const quiet = provs
    .filter(p => p.lt_changes === 0 && p.lt_years >= QUIET_MIN_YEARS)
    .sort((a, b) => b.lt_years - a.lt_years)
    .slice(0, BOARD_SIZE);
  document.getElementById("provBoards")!.innerHTML =
    board("Most drift-prone", "LT · rate", "Bar is the point estimate, the faint band its 95% interval.", drifty) +
    board("Nothing detected yet", "LT · upper bound",
      "Ranked by monitoring volume: the more we have watched, the tighter the ceiling on their true rate.", quiet);

  const provQ = document.getElementById("provQ") as HTMLInputElement;
  const provFilters = new Set<string>();
  let provSort: ProviderSortKey = "lt_rate";
  let provDir = -1;

  function renderProviders(): void {
    const q = provQ.value.trim().toLowerCase();
    const list = provs.filter(p => {
      if (q && !p.name.toLowerCase().includes(q)) return false;
      if (provFilters.has("changed") && p.lt_changes === 0) return false;
      if (provFilters.has("rateable") && p.lt_rate === null) return false;
      if (provFilters.has("b3it") && p.b3it_endpoints === 0) return false;
      return true;
    });
    list.sort((a, b) => {
      let av: string | number, bv: string | number;
      // an unmeasurable rate is not a low rate: park those rows at the bottom
      if (provSort === "lt_rate") { av = a.lt_rate ?? -1; bv = b.lt_rate ?? -1; }
      else if (provSort === "n_endpoints") { av = a.n_endpoints; bv = b.n_endpoints; }
      else if (provSort === "last_change") { av = a.last_change ?? ""; bv = b.last_change ?? ""; }
      else { av = a.name.toLowerCase(); bv = b.name.toLowerCase(); }
      if (av < bv) return -provDir;
      if (av > bv) return provDir;
      return a.name.localeCompare(b.name);
    });
    document.getElementById("provBody")!.innerHTML =
      list.map(p => `<tr>
        <td><a class="model-cell" href="providers/${esc(p.slug)}.html">${esc(p.name)}</a>
          <div class="org-cell">${p.n_variants > 1 ? plural(p.n_variants, "serving variant") : "single variant"} · ${plural(p.n_models, "model")}</div></td>
        <td class="r"><span class="cc">${p.n_endpoints}</span></td>
        <td style="min-width:190px">${rateBar(p.lt_years, p.lt_rate, p.lt_ci, maxRate)}</td>
        <td class="col-hide">${volGrid(p.lt_years)}</td>
        <td class="col-hide">${p.b3it_endpoints
          ? `<span class="vol"><span class="lbl">${p.b3it_endpoints} ep · ${p.b3it_years.toFixed(1)} ep-yr</span></span>`
          : '<span class="org-cell">—</span>'}</td>
        <td class="r col-hide"><span class="cc ${p.last_change ? "some" : "zero"}">${p.last_change ? esc(p.last_change) : "—"}</span></td>
      </tr>`).join("") || '<tr><td colspan="6"><div class="empty">No providers match.</div></td></tr>';
    const unrated = provs.filter(p => p.lt_rate === null).length;
    document.getElementById("provFoot")!.textContent =
      `${list.length} of ${provs.length} providers · ${unrated} under ${MIN_ENDPOINT_YEARS} endpoint-years, so not yet rateable`;
    document.querySelectorAll<HTMLElement>("th[data-psort]").forEach(th => {
      const arr = th.querySelector(".arr");
      if (arr) arr.textContent = th.dataset.psort === provSort ? (provDir < 0 ? "▼" : "▲") : "";
    });
  }
  provQ.addEventListener("input", renderProviders);
  document.getElementById("provChips")!.addEventListener("click", e => {
    const chip = (e.target as HTMLElement).closest(".chip") as HTMLElement | null;
    if (!chip) return;
    const f = chip.dataset.f!;
    if (provFilters.has(f)) { provFilters.delete(f); chip.classList.remove("on"); }
    else { provFilters.add(f); chip.classList.add("on"); }
    renderProviders();
  });
  document.querySelectorAll<HTMLElement>("th[data-psort]").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.psort as ProviderSortKey;
    if (provSort === k) provDir *= -1;
    else { provSort = k; provDir = k === "name" ? 1 : -1; }
    renderProviders();
  }));
  renderProviders();

  // ---- endpoint directory ----
  const rows = DATA.endpoints;
  const active = new Set<string>();
  let sortKey: SortKey = "nChanges";
  let sortDir = -1;
  const STATUS_ORDER: Record<EndpointStatus, number> = { changed: 0, stable: 1, retired: 2 };
  function stableCell(r: EndpointRow): string {
    if (r.status === "retired" || r.stableDays === null) return `<span class="org-cell">—</span>`;
    const d = r.stableDays;
    return `<span class="cc">${d >= 365 ? (d / 365).toFixed(1) + "y" : d + "d"}</span>`;
  }
  function render(): void {
    const q = (document.getElementById("q") as HTMLInputElement).value.trim().toLowerCase();
    const mf = [...active].filter(f => f === "lt" || f === "b3it");
    const list = rows.filter(r => {
      if (q && !(`${r.model} ${r.provider} ${r.org}`.toLowerCase().includes(q))) return false;
      if (mf.length && !mf.every(m => r.methods.includes(m))) return false;
      if (active.has("everchanged") && r.nChanges === 0) return false;
      if (active.has("recent") && r.status !== "changed") return false;
      if (active.has("retired") && r.status !== "retired") return false;
      return true;
    });
    list.sort((a, b) => {
      let av: string | number, bv: string | number;
      if (sortKey === "status") { av = STATUS_ORDER[a.status]; bv = STATUS_ORDER[b.status]; }
      else if (sortKey === "stableDays") { av = a.stableDays ?? -1; bv = b.stableDays ?? -1; }
      else if (sortKey === "nChanges") { av = a.nChanges; bv = b.nChanges; }
      else { av = String(a[sortKey]).toLowerCase(); bv = String(b[sortKey]).toLowerCase(); }
      if (av < bv) return -sortDir; if (av > bv) return sortDir; return a.model.localeCompare(b.model);
    });
    document.getElementById("dirBody")!.innerHTML = list.map(r => {
      const isLT = r.methods.includes("lt");
      return `
      <tr>
        <td><a class="model-cell" href="models/${esc(r.modelSlug)}.html">${esc(r.model)}</a><div class="org-cell">${esc(r.org)}</div></td>
        <td class="col-hide"><a class="prov-cell" href="providers/${esc(r.providerSlug)}.html">${esc(r.provider)}</a></td>
        <td><a href="endpoints/${esc(r.slug)}.html">${statusPill(r.status)}</a></td>
        <td class="r"><span class="cc ${r.nChanges ? "some" : "zero"}">${r.nChanges}</span></td>
        <td class="col-hide"><span class="methods">${methodBadges(r.methods)}</span></td>
        <td class="r col-hide">${stableCell(r)}</td>
        <td class="col-hide" style="width:130px">${sparkline(r.trace, isLT ? LT_CAP : B3IT_CAP, isLT ? "var(--accent)" : "var(--b3it)", null)}</td>
      </tr>`;
    }).join("") || '<tr><td colspan="7"><div class="empty">No endpoints match.</div></td></tr>';
    document.getElementById("dirFoot")!.textContent = `${list.length} of ${rows.length} endpoints`;
    document.querySelectorAll("thead th[data-sort] .arr").forEach(a => a.textContent = "");
    const th = document.querySelector(`thead th[data-sort="${sortKey}"] .arr`); if (th) th.textContent = sortDir > 0 ? "▲" : "▼";
  }
  document.getElementById("dirCount")!.innerHTML = `${fmtInt(rows.length)} endpoints · <b style="color:var(--changed)">${S.changes_total} changes</b> across ${S.changed_endpoints} of them`;
  document.getElementById("q")!.addEventListener("input", render);
  document.getElementById("chips")!.addEventListener("click", e => {
    const chip = (e.target as HTMLElement).closest(".chip") as HTMLElement | null; if (!chip) return;
    const f = chip.dataset.f!;
    if (active.has(f)) { active.delete(f); chip.classList.remove("on"); }
    else { active.add(f); chip.classList.add("on"); }
    render();
  });
  document.querySelectorAll<HTMLElement>("thead th[data-sort]").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.sort as SortKey; if (sortKey === k) sortDir *= -1; else { sortKey = k; sortDir = (k === "stableDays" || k === "nChanges") ? -1 : 1; } render();
  }));
  render();
})();
