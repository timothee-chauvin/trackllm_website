// The `init` export both makes this a module (so its top-level names don't collide
// with other bundler entrypoints when type-checked as one tsc program) and lets the
// smoke tests re-render a fresh document without busting the module cache.
import {
  FeedItem,
  MIN_ENDPOINT_YEARS,
  bindFilterChips,
  esc,
  eventRow,
  highlight,
  magnitudeLabel,
  methodBadges,
  plural,
  rateBar,
  relDays,
  relativeAge,
  showLoadError,
  toggleChip,
  volGrid,
} from "./components";
import { EndpointRow, initDirectory, initSortHeaders } from "./directory";

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
  // absolute instants (overview.py), turned into an age here at page load
  last_query_lt: string | null;
  last_query_b3it: string | null;
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

/** The one real change event the hero draws, chosen at build time (hero.py). */
interface Hero {
  slug: string;
  model: string;
  org: string;
  provider: string;
  method: "lt" | "b3it";
  date: string;
  daysAgo: number;
  magnitude: number;
  baseline: number;
  start: string;
  end: string;
  values: number[];
  changeFrac: number;
  yMax: number;
}

type ProviderSortKey = "name" | "n_endpoints" | "lt_rate" | "last_change";

const BOARD_SIZE = 5;
const QUIET_MIN_YEARS = 1; // a "nothing detected yet" board entry needs real exposure
const FRESH_TICK_MS = 60_000; // the line's own resolution, so no point ticking faster

// Hero trace geometry, in the SVG's own box. Both hero layers are sized to the clear
// space above the stat cards, so the curve is never half-hidden behind them however
// the hero reflows: zero drift sits on the cards' top edge and the fill between them
// is the area under the curve.
const HERO_W = 1200;
const HERO_VB_H = 200; // also the zero line
const HERO_TOP = 14;
const HERO_HIT_WIDTH = 18; // invisible fat stroke: the curve is 1.6 units thin
const HERO_TIP_DX = 16;
const HERO_TIP_DY = 18;

export async function init(): Promise<void> {
  let DATA: {
    stats: Stats;
    hero: Hero | null;
    feed: FeedItem[];
    providers: ProviderRate[];
    endpoints: EndpointRow[];
  };
  try {
    const res = await fetch("data/overview.json");
    if (!res.ok) throw new Error(`overview.json: HTTP ${res.status}`);
    DATA = await res.json();
  } catch (err) {
    showLoadError("telemetry", "the overview data");
    // no half-broken hero above the error card: drop its layers and the live dot
    document.getElementById("eyebrow")?.remove();
    document.querySelectorAll(".hero-trace, .hero-hit-layer, .hero-tip").forEach(el => el.remove());
    throw err;
  }
  const S = DATA.stats;
  const fmtInt = (n: number): string => n.toLocaleString("en-US");
  const fmtM = (n: number): string =>
    n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "k" : "" + n;

  // ---- hero trace: one real change event, picked at build time (hero.py) ----
  // Full-bleed by construction -- x runs 0..HERO_W with no padding and no end fade,
  // so the line never appears to stop short of either edge.
  function renderHero(h: Hero): void {
    const svg = document.getElementById("heroTrace") as unknown as SVGSVGElement;
    const hitLayer = document.getElementById("heroHit") as unknown as SVGSVGElement;
    const tip = document.getElementById("heroTip")!;
    const hero = svg.parentElement!;

    const color = h.method === "lt" ? "var(--accent)" : "var(--b3it)";
    const x = (i: number): number => (i / (h.values.length - 1)) * HERO_W;
    const y = (v: number): number => HERO_VB_H - (v / h.yMax) * (HERO_VB_H - HERO_TOP);
    const line = h.values
      .map((v, i) => (i ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1))
      .join(" ");
    const cut = (h.changeFrac * HERO_W).toFixed(1);
    svg.innerHTML = `
      <line x1="${cut}" y1="0" x2="${cut}" y2="${HERO_VB_H}" stroke="${color}"
        stroke-width="1" stroke-dasharray="5 5" opacity="0.55" vector-effect="non-scaling-stroke"/>
      <path d="${line}" fill="none" stroke="${color}" stroke-width="1.6"
        stroke-linejoin="round" vector-effect="non-scaling-stroke"/>`;

    // The hover target has to sit *above* .hero-inner, which covers the whole hero and
    // would otherwise swallow every pointer event before the trace behind it sees one.
    hitLayer.innerHTML = `<a href="endpoints/${esc(h.slug)}.html" class="hero-hit">
      <path d="${line}" fill="none" stroke="transparent" stroke-width="${HERO_HIT_WIDTH}"/></a>`;

    const method = h.method === "lt" ? "LT" : "B3IT";
    tip.innerHTML = `<div class="who">${methodBadges([h.method])}
        <b>${esc(h.model)}</b><span class="at">@ ${esc(h.provider)}</span></div>
      <div class="what">Live data from this endpoint — ${method} detected a change on
        ${esc(h.date)} (${relDays(h.daysAgo)}), moving from
        ${magnitudeLabel(h.method, h.baseline)} to ${magnitudeLabel(h.method, h.magnitude)}.
        Showing ${esc(h.start)} to ${esc(h.end)}, one point per day.</div>
      <div class="go">Open the endpoint →</div>`;

    // Both layers stop where the stat cards begin, so no part of the curve is drawn
    // behind them and no pointer event over a card reaches the hit stroke.
    const cards = document.getElementById("telemetry")!;
    function fitLayers(): void {
      const top = cards.getBoundingClientRect().top - hero.getBoundingClientRect().top;
      for (const el of [svg, hitLayer]) el.style.height = Math.max(0, top) + "px";
    }
    for (const el of [svg, hitLayer]) {
      el.setAttribute("viewBox", `0 0 ${HERO_W} ${HERO_VB_H}`);
    }
    fitLayers();
    new ResizeObserver(fitLayers).observe(hero);

    const hit = hitLayer.querySelector(".hero-hit")!;
    hit.addEventListener("pointermove", ev => {
      const e = ev as PointerEvent;
      const box = hero.getBoundingClientRect();
      tip.hidden = false;
      const left = e.clientX - box.left + HERO_TIP_DX;
      tip.style.left = Math.max(0, Math.min(left, box.width - tip.offsetWidth)) + "px";
      tip.style.top = e.clientY - box.top + HERO_TIP_DY + "px";
    });
    hit.addEventListener("pointerleave", () => { tip.hidden = true; });
  }

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
  // after the cards: the hero layers are sized against where they land
  if (DATA.hero && DATA.hero.values.length > 1) renderHero(DATA.hero);
  else document.querySelectorAll(".hero-trace, .hero-hit-layer, .hero-tip").forEach(el => el.remove());

  // How fresh each method's data is. The build emits absolute instants, so the age
  // is computed here and re-computed as the tab stays open; a method with no data
  // at all is left out rather than shown as an age since the epoch.
  const lastQueries: [string, string | null][] = [
    ["LT", S.last_query_lt],
    ["B3IT", S.last_query_b3it],
  ];
  const freshEl = document.getElementById("freshness")!;
  function paintFreshness(): void {
    const now = Date.now();
    const parts: string[] = [];
    for (const [method, iso] of lastQueries) {
      if (iso === null) continue;
      parts.push(`<span class="num" title="${esc(iso)}">${relativeAge(iso, now)}</span>` +
        ` <span class="${method.toLowerCase()}">(${method})</span>`);
    }
    freshEl.innerHTML = parts.length
      ? `Last update ${parts.join('<span class="sep">·</span>')}`
      : "";
  }
  paintFreshness();
  setInterval(paintFreshness, FRESH_TICK_MS);

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
  const provSort = initSortHeaders<ProviderSortKey>(
    "psort", "lt_rate", -1, ["n_endpoints", "lt_rate", "last_change"], renderProviders);

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
      if (provSort.key === "lt_rate") {
        // an unmeasurable rate is not a low rate: park those rows at the bottom in
        // both directions, so reversing the sort never promotes them to the top
        if ((a.lt_rate === null) !== (b.lt_rate === null)) return a.lt_rate === null ? 1 : -1;
        av = a.lt_rate ?? 0; bv = b.lt_rate ?? 0;
      }
      else if (provSort.key === "n_endpoints") { av = a.n_endpoints; bv = b.n_endpoints; }
      else if (provSort.key === "last_change") { av = a.last_change ?? ""; bv = b.last_change ?? ""; }
      else { av = a.name.toLowerCase(); bv = b.name.toLowerCase(); }
      if (av < bv) return -provSort.dir;
      if (av > bv) return provSort.dir;
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
    provSort.paintArrows();
  }
  provQ.addEventListener("input", renderProviders);
  bindFilterChips(document.getElementById("provChips")!, provFilters, renderProviders);
  renderProviders();

  // ---- endpoint directory ----
  const rows = DATA.endpoints;
  // provider pages only exist for providers with tracked endpoints; a row whose
  // provider has none must name it without linking it
  const providerPages = new Set(provs.map(p => p.slug));
  const active = new Set<string>();
  // headline chips are OR within the group; an empty set means no status constraint
  const statusFilters = new Set<string>(["tracked"]);
  const render = initDirectory({
    rows,
    root: "",
    q: document.getElementById("q") as HTMLInputElement,
    body: document.getElementById("dirBody")!,
    foot: document.getElementById("dirFoot")!,
    descending: ["stableDays", "nChanges"],
    providerValue: r => r.provider.toLowerCase(),
    list: q => {
      const ql = q.toLowerCase();
      const mf = [...active].filter(f => f === "lt" || f === "b3it");
      // a change-history chip bypasses the status group: its result set is defined
      // by the change criterion alone (only observed endpoints can have changes),
      // exactly as before the status chips existed
      const changeChip = active.has("everchanged") || active.has("recent");
      // grey out whatever the current mode ignores, so chips never look
      // toggleable while having no effect
      const chipsEl = document.getElementById("chips")!;
      chipsEl.classList.toggle("bypass-all", !!q);
      chipsEl.classList.toggle("bypass-status", changeChip);
      // a search spans every row: chips must never hide a hit
      return q
        ? rows.filter(r => `${r.model} ${r.provider} ${r.org}`.toLowerCase().includes(ql))
        : rows.filter(r => {
            if (!changeChip && statusFilters.size && !statusFilters.has(r.headline)) return false;
            if (mf.length && !mf.every(m => r.methods.includes(m))) return false;
            if (active.has("everchanged") && r.nChanges === 0) return false;
            if (active.has("recent") && r.status !== "changed") return false;
            return true;
          });
    },
    leadCells: (r, q) => {
      const provCell = providerPages.has(r.providerSlug)
        ? `<a class="prov-cell" href="providers/${esc(r.providerSlug)}.html">${highlight(r.provider, q)}</a>`
        : `<span class="prov-cell">${highlight(r.provider, q)}</span>`;
      return `
        <td><a class="model-cell" href="models/${esc(r.modelSlug)}.html">${highlight(r.model, q)}</a><div class="org-cell">${highlight(r.org, q)}</div></td>
        <td class="col-hide">${provCell}</td>`;
    },
  });
  document.getElementById("dirCount")!.innerHTML = `${fmtInt(rows.length)} endpoints · <b style="color:var(--changed)">${S.changes_total} changes</b> across ${S.changed_endpoints} of them`;
  // two chip groups share the toolbar: data-st chips toggle the status set,
  // data-f chips the method/change set
  document.getElementById("chips")!.addEventListener("click", e => {
    const chip = (e.target as HTMLElement).closest(".chip") as HTMLElement | null; if (!chip) return;
    const st = chip.dataset.st;
    toggleChip(chip, st ? statusFilters : active, st ?? chip.dataset.f!);
    render();
  });
}

init();
