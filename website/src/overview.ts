// The `init` export both makes this a module (so its top-level names don't collide
// with other bundler entrypoints when type-checked as one tsc program) and lets the
// smoke tests re-render a fresh document without busting the module cache.
import {
  bindTips,
  esc,
  eventRow,
  magnitudeLabel,
  methodBadges,
  relDays,
  relativeAge,
} from "./components";
import { dirRowsHtml, overviewLeadCells, sortEndpointRows } from "./directory";
import {
  HERO_CLEAR_GAP,
  HERO_HIT_WIDTH,
  HERO_TIP_DX,
  HERO_TIP_DY,
  HERO_TOP,
  HERO_VB_H,
  HERO_W,
  HERO_WIDE,
  heroDrawnTo,
  heroStretch,
} from "./hero_geom";
import { Hero, OverviewData, fmtInt, loadOverview } from "./overview_data";
import { rateablePlotRows, ratePlot } from "./rate_plot";

const PLOT_SIZE = 10; // the front page shows the most drift-prone slice
const DIR_SIZE = 10; // ... and the most-changed endpoints
const FRESH_TICK_MS = 60_000; // the line's own resolution, so no point ticking faster

export async function init(): Promise<void> {
  let DATA: OverviewData;
  try {
    DATA = await loadOverview("telemetry");
  } catch (err) {
    // no half-broken hero above the error card: drop its layers and the live dot
    document.getElementById("eyebrow")?.remove();
    document.querySelectorAll(".hero-trace, .hero-hit-layer, .hero-tip").forEach(el => el.remove());
    throw err;
  }
  const S = DATA.stats;
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
    // Its only content is an invisible path, so the link needs a name of its own --
    // the same thing the hover card says, for anyone who never sees the hover card.
    hitLayer.innerHTML = `<a href="endpoints/${esc(h.slug)}.html" class="hero-hit"
      aria-label="Open ${esc(h.model)} @ ${esc(h.provider)} — the endpoint this curve is drawn from">
      <path d="${line}" fill="none" stroke="transparent" stroke-width="${HERO_HIT_WIDTH}"/></a>`;

    const method = h.method === "lt" ? "LT" : "B3IT";
    tip.innerHTML = `<div class="who">${methodBadges([h.method])}
        <b>${esc(h.model)}</b><span class="at">@ ${esc(h.provider)}</span></div>
      <div class="what">Live data from this endpoint — ${method} detected a change on
        ${esc(h.date)} (${relDays(h.daysAgo)}), moving from
        ${magnitudeLabel(h.method, h.baseline)} to ${magnitudeLabel(h.method, h.magnitude)}.
        Showing ${esc(h.start)} to <span class="drawn-to">${esc(h.end)}</span>, one point per day.</div>
      <div class="go">Open the endpoint →</div>`;

    // Both layers stop where the stat cards begin, so no part of the curve is drawn
    // behind them and no pointer event over a card reaches the hit stroke.
    const cards = document.getElementById("telemetry")!;
    const lede = document.querySelector(".lede")!; // the hero's widest line
    const wide = matchMedia(HERO_WIDE);
    function fitLayers(): void {
      const box = hero.getBoundingClientRect();
      const top = cards.getBoundingClientRect().top - box.top;
      const clearTo = lede.getBoundingClientRect().right - box.left + HERO_CLEAR_GAP;
      const stretch = wide.matches ? heroStretch(h.changeFrac, box.width, clearTo) : 1;
      for (const el of [svg, hitLayer]) {
        el.style.height = Math.max(0, top) + "px";
        el.style.width = (stretch * 100).toFixed(2) + "%";
      }
      tip.querySelector(".drawn-to")!.textContent = heroDrawnTo(h.start, h.end, stretch);
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

  const now = Date.now();
  document.getElementById("feed")!.innerHTML = DATA.feed.map(e => eventRow(e, now)).join("");
  document.getElementById("allChanges")!.textContent = `All ${S.changes_total} changes →`;

  // ---- providers: the most drift-prone slice of the rate plot ----
  const provs = DATA.providers;
  document.getElementById("provPlot")!.innerHTML =
    ratePlot(rateablePlotRows(provs).slice(0, PLOT_SIZE), "");

  // ---- endpoints: the most-changed actively tracked rows ----
  const rows = DATA.endpoints;
  const providerPages = new Set(provs.map(p => p.slug));
  const top = rows.filter(r => r.headline === "tracked");
  sortEndpointRows(top, "nChanges", -1, r => r.provider.toLowerCase());
  document.getElementById("dirBody")!.innerHTML =
    dirRowsHtml(top.slice(0, DIR_SIZE), "", overviewLeadCells(providerPages), "");
  document.getElementById("dirCount")!.innerHTML = `${fmtInt(rows.length)} endpoints · <b style="color:var(--changed)">${S.changes_total} changes</b> across ${S.changed_endpoints} of them`;
  // every directory badge/pill above carries a popover
  bindTips(document.body);
}

init();
