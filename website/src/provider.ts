// `export {}` makes this a module so its top-level names (MONTH_NAMES, init, ...)
// don't collide with the same names in other bundler-entrypoint scripts (endpoint.ts)
// when type-checked together as one tsc program.
export {};

import {
  B3IT_CAP,
  LT_CAP,
  MIN_ENDPOINT_YEARS,
  esc,
  methodBadges,
  rateBar,
  sparkline,
  statusPill,
  volGrid,
} from "./components";

interface MethodBlock {
  endpoints: number;
  years: number;
  changes: number;
  rate: number | null;
  ci: [number, number] | null;
}

interface Variant {
  name: string;
  n_endpoints: number;
  lt: MethodBlock;
  b3it: MethodBlock;
  monitoring: number[];
}

interface ProviderChange {
  date: string;
  method: "lt" | "b3it";
  magnitude: number | null;
  model: string;
  provider: string;
  slug: string;
}

interface EndpointRow {
  slug: string;
  model: string;
  modelSlug: string;
  org: string;
  provider: string;
  methods: string[];
  status: string;
  stableDays: number | null;
  nChanges: number;
  trace: number[];
}

interface ProviderData {
  name: string;
  slug: string;
  n_endpoints: number;
  n_models: number;
  n_variants: number;
  first: string | null;
  last: string | null;
  months: string[];
  lt: MethodBlock;
  b3it: MethodBlock;
  variants: Variant[];
  changes: ProviderChange[];
  endpoints: EndpointRow[];
}

type SortKey = "model" | "provider" | "status" | "nChanges" | "stableDays";

const MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const monthLabel = (m: string): string => MONTH_NAMES[+m.slice(5, 7) - 1] + " '" + m.slice(2, 4);
const prettyDate = (d: string | null): string =>
  d ? MONTH_NAMES[+d.slice(5, 7) - 1] + " " + d.slice(0, 4) : "—";
const plural = (n: number, w: string): string => `${n} ${w}${n === 1 ? "" : "s"}`;

async function init(): Promise<void> {
  const slugEl = document.getElementById("providerData");
  if (!slugEl) return;
  const slug: string = JSON.parse(slugEl.textContent || '""');
  const res = await fetch(`../data/providers/${slug}.json`).catch(() => null);
  const D: ProviderData | null = res && res.ok ? await res.json() : null;
  if (!D) return;

  const NAME = D.name;
  const FIRST = D.first;
  // the endpoint/change `provider` field is "<base>" for the default stack and
  // "<base>/<variant>" otherwise; variant names come straight from the JSON.
  const variantOf = (provider: string): string =>
    provider === NAME ? "" : provider.slice(NAME.length + 1);
  const variantLabel = (v: string): string => (v ? esc(NAME) + "/" + esc(v) : esc(NAME));

  const variants = D.variants.slice().sort((a, b) => b.n_endpoints - a.n_endpoints);
  // one scale for every bar on the page so variants are visually comparable
  const maxRate = Math.max(
    1,
    D.lt.rate ?? 0,
    ...variants.map((v) => v.lt.rate ?? 0),
    ...variants.map((v) => v.lt.ci?.[1] ?? 0)
  );

  // lede
  const rated = variants.filter((v) => v.lt.rate !== null);
  let spread = "";
  if (rated.length > 1) {
    const worst = rated.reduce((a, b) => (a.lt.rate! > b.lt.rate! ? a : b));
    const best = rated.reduce((a, b) => (a.lt.rate! < b.lt.rate! ? a : b));
    if (worst.lt.rate! > best.lt.rate! * 1.8) {
      spread =
        ` Its serving variants do not behave alike: <span class="hl">${variantLabel(worst.name)}</span>` +
        ` shows ${worst.lt.rate!.toFixed(2)} LT changes per endpoint-year against` +
        ` ${best.lt.rate!.toFixed(2)} for <b>${variantLabel(best.name)}</b>.`;
    }
  }
  const ledeEl = document.getElementById("lede");
  if (ledeEl) {
    ledeEl.innerHTML =
      `<b>${D.n_endpoints}</b> ${D.n_endpoints === 1 ? "endpoint" : "endpoints"} across ` +
      `${plural(D.n_models, "model")} and ${plural(D.n_variants, "serving variant")}, ` +
      (D.first && D.last
        ? `monitored ${prettyDate(D.first)} – ${prettyDate(D.last)}.`
        : "with no monitoring recorded yet.") +
      spread;
  }

  const summaryEl = document.getElementById("summary");
  if (summaryEl) {
    const active = D.endpoints.filter((e) => e.status !== "retired").length;
    const affected = D.endpoints.filter((e) => e.nChanges > 0).length;
    summaryEl.innerHTML = `
      <div class="s"><div class="v">${D.n_endpoints}</div><div class="k">Endpoints</div></div>
      <div class="s"><div class="v">${active}</div><div class="k">Still active</div></div>
      <div class="s"><div class="v"${D.changes.length ? ' style="color:var(--changed)"' : ""}>${D.changes.length}</div><div class="k">Changes detected</div></div>
      <div class="s"><div class="v">${affected}</div><div class="k">Endpoints affected</div></div>`;
  }

  // rate cards — rate and ci come precomputed; rate === null *is* the gate
  function rateCard(label: string, badge: string, m: MethodBlock): string {
    const top = `<div class="top"><span class="k">${label}</span>${badge}</div>`;
    if (m.rate === null || m.ci === null) {
      return `<div class="ratecard nd">${top}
        <div class="big nd">Not enough monitoring</div>
        <div class="ci">${m.years.toFixed(2)} endpoint-years accumulated — a rate needs ${MIN_ENDPOINT_YEARS.toFixed(1)}.</div>
        <div class="exposure">${
          m.endpoints === 0
            ? "No endpoints run this method here yet."
            : `${plural(m.changes, "change")} seen so far across ${plural(m.endpoints, "endpoint")}${FIRST ? ", monitored since " + prettyDate(FIRST) : ""}.`
        } The rate appears once the exposure does.</div>
      </div>`;
    }
    return `<div class="ratecard">${top}
      <div class="big">${m.rate.toFixed(2)} <small>changes / endpoint-year</small></div>
      <div class="ci">95% interval ${m.ci[0].toFixed(2)} – ${m.ci[1].toFixed(2)}${m.changes === 0 ? " (rule of three)" : ""}</div>
      <div class="exposure"><b>${m.changes}</b> change${m.changes === 1 ? "" : "s"} over <b>${m.years.toFixed(1)}</b> endpoint-years of monitoring on ${plural(m.endpoints, "endpoint")}.</div>
    </div>`;
  }
  const cardsEl = document.getElementById("ratecards");
  if (cardsEl) {
    cardsEl.innerHTML =
      rateCard("Logprob tracking", '<span class="badge lt">LT</span>', D.lt) +
      rateCard("Border input tracking", '<span class="badge b3it">B3IT</span>', D.b3it);
  }

  // monitoring lanes: grey area = endpoints under monitoring per month, dots = changes
  const MONTHS = D.months;
  const NM = Math.max(1, MONTHS.length - 1);
  const laneX = (i: number): number => (i / NM) * 600;

  function laneSvg(v: Variant, changes: ProviderChange[]): string {
    const W = 600, H = 46, base = H - 9;
    const max = Math.max(...v.monitoring, 1);
    const pts = v.monitoring.map((c, i): [number, number] => [laneX(i), base - (c / max) * (base - 6)]);
    const line = pts
      .map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1))
      .join(" ");
    const area = pts.length
      ? `<path d="${line} L${W} ${base} L0 ${base} Z" fill="var(--text-dim)" opacity="0.16"/>
         <path d="${line}" fill="none" stroke="var(--text-dim)" stroke-width="1" opacity="0.5" vector-effect="non-scaling-stroke"/>`
      : "";
    const dots = changes
      .map((c) => {
        const isLT = c.method === "lt";
        const col = isLT ? "var(--accent)" : "var(--b3it)";
        const r = 2.6 + Math.min(1, (c.magnitude ?? 0) / (isLT ? LT_CAP : B3IT_CAP)) * 3.2;
        const idx = MONTHS.indexOf(c.date.slice(0, 7));
        const cx = laneX(idx < 0 ? MONTHS.length - 1 : idx).toFixed(1);
        const mag = c.magnitude === null ? "" : isLT ? ` · ${c.magnitude} nats` : ` · TV ${c.magnitude}`;
        return `<line x1="${cx}" y1="${base}" x2="${cx}" y2="6" stroke="${col}" stroke-width="1" opacity="0.22"/>
          <circle cx="${cx}" cy="${base - 14}" r="${r.toFixed(1)}" fill="${col}" fill-opacity="0.75" stroke="${col}" stroke-width="1">
          <title>${esc(c.date)} · ${esc(c.model)} · ${isLT ? "LT" : "B3IT"}${esc(mag)}</title></circle>`;
      })
      .join("");
    return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <line x1="0" y1="${base}" x2="${W}" y2="${base}" stroke="var(--border)" stroke-width="1"/>
      ${area}${dots}</svg>`;
  }

  const timelineEl = document.getElementById("timeline");
  if (timelineEl) {
    if (!MONTHS.length) {
      timelineEl.innerHTML = `<div class="empty">No monitoring recorded yet for this provider.</div>`;
    } else {
      const lanes = variants
        .map((v) => {
          const vc = D.changes.filter((c) => variantOf(c.provider) === v.name);
          return `<div class="tlrow">
            <div class="lane-k">${variantLabel(v.name)}<small>${plural(v.n_endpoints, "endpoint")}</small></div>
            <div class="lane">${laneSvg(v, vc)}</div>
            <div class="lane-m"><b class="${vc.length ? "" : "zero"}">${vc.length}</b>${vc.length === 1 ? "change" : "changes"}</div>
          </div>`;
        })
        .join("");
      const ticks = MONTHS.map((m, i) =>
        i % 3 === 0 || i === MONTHS.length - 1
          ? `<span style="left:${((i / NM) * 100).toFixed(1)}%">${monthLabel(m)}</span>`
          : ""
      ).join("");
      timelineEl.innerHTML = `<div class="tl">${lanes}</div>
        <div class="tlaxis"><div class="ticks">${ticks}</div></div>`;
    }
  }

  const variantBody = document.getElementById("variantBody");
  if (variantBody) {
    variantBody.innerHTML = variants
      .map(
        (v) => `<tr>
        <td><span class="prov-cell">${variantLabel(v.name)}</span>${v.name ? "" : ' <span class="org-cell">(default)</span>'}</td>
        <td class="r"><span class="cc">${v.n_endpoints}</span></td>
        <td style="min-width:190px">${rateBar(v.lt.years, v.lt.rate, v.lt.ci, maxRate)}</td>
        <td class="col-hide">${volGrid(v.lt.years)}</td>
        <td class="r col-hide"><span class="cc ${v.lt.changes ? "some" : "zero"}">${v.lt.changes}</span></td>
        <td class="r col-hide"><span class="cc ${v.b3it.endpoints ? "" : "zero"}">${v.b3it.endpoints ? v.b3it.endpoints + " ep" : "—"}</span></td>
      </tr>`
      )
      .join("");
  }

  // endpoint directory
  const rows = D.endpoints;
  const filters = new Set<string>();
  let sortKey: SortKey = "nChanges";
  let sortDir = -1;
  const STATUS_ORDER: Record<string, number> = { changed: 0, stable: 1, retired: 2 };
  const qEl = document.getElementById("epq") as HTMLInputElement | null;
  const bodyEl = document.getElementById("epBody");
  const footEl = document.getElementById("epFoot");
  if (!qEl || !bodyEl || !footEl) return;

  function stableCell(r: EndpointRow): string {
    if (r.status === "retired" || r.stableDays === null) return `<span class="org-cell">—</span>`;
    return `<span class="cc">${r.stableDays >= 365 ? (r.stableDays / 365).toFixed(1) + "y" : r.stableDays + "d"}</span>`;
  }

  function render(): void {
    const q = qEl!.value.trim().toLowerCase();
    const list = rows.filter((r) => {
      if (q && !`${r.model} ${r.org} ${r.provider}`.toLowerCase().includes(q)) return false;
      if (filters.has("changed") && r.nChanges === 0) return false;
      if (filters.has("retired") && r.status !== "retired") return false;
      if (filters.has("b3it") && !r.methods.includes("b3it")) return false;
      return true;
    });
    list.sort((a, b) => {
      let av: string | number, bv: string | number;
      if (sortKey === "status") { av = STATUS_ORDER[a.status] ?? 3; bv = STATUS_ORDER[b.status] ?? 3; }
      else if (sortKey === "stableDays") { av = a.stableDays ?? -1; bv = b.stableDays ?? -1; }
      else if (sortKey === "nChanges") { av = a.nChanges; bv = b.nChanges; }
      else if (sortKey === "provider") { av = variantOf(a.provider); bv = variantOf(b.provider); }
      else { av = a.model.toLowerCase(); bv = b.model.toLowerCase(); }
      if (av < bv) return -sortDir;
      if (av > bv) return sortDir;
      return a.model.localeCompare(b.model);
    });
    bodyEl!.innerHTML =
      list
        .map((r) => {
          const isLT = r.methods.includes("lt");
          return `<tr>
        <td><a class="model-cell" href="../models/${esc(r.modelSlug)}.html">${esc(r.model)}</a><div class="org-cell">${esc(r.org)}</div></td>
        <td class="col-hide"><span class="prov-cell">${variantOf(r.provider) ? esc(variantOf(r.provider)) : "—"}</span></td>
        <td><a href="../endpoints/${esc(r.slug)}.html">${statusPill(r.status)}</a></td>
        <td class="r"><span class="cc ${r.nChanges ? "some" : "zero"}">${r.nChanges}</span></td>
        <td class="col-hide"><span class="methods">${methodBadges(r.methods)}</span></td>
        <td class="r col-hide">${stableCell(r)}</td>
        <td class="col-hide" style="width:130px">${sparkline(r.trace, isLT ? LT_CAP : B3IT_CAP, isLT ? "var(--accent)" : "var(--b3it)", null)}</td>
      </tr>`;
        })
        .join("") || '<tr><td colspan="7"><div class="empty">No endpoints match.</div></td></tr>';
    footEl!.textContent = `${list.length} of ${rows.length} endpoints`;
    document.querySelectorAll<HTMLElement>("th[data-sort]").forEach((th) => {
      const arr = th.querySelector(".arr");
      if (arr) arr.textContent = th.dataset.sort === sortKey ? (sortDir < 0 ? "▼" : "▲") : "";
    });
  }

  const countEl = document.getElementById("epCount");
  if (countEl) {
    countEl.textContent = `${rows.length} endpoints from ${NAME} · model names link to the model page, the status pill to the endpoint page`;
  }
  qEl.addEventListener("input", render);
  document.getElementById("epChips")?.addEventListener("click", (e) => {
    const chip = (e.target as HTMLElement).closest(".chip") as HTMLElement | null;
    if (!chip) return;
    const f = chip.dataset.f!;
    if (filters.has(f)) { filters.delete(f); chip.classList.remove("on"); }
    else { filters.add(f); chip.classList.add("on"); }
    render();
  });
  document.querySelectorAll<HTMLElement>("th[data-sort]").forEach((th) =>
    th.addEventListener("click", () => {
      const k = th.dataset.sort as SortKey;
      if (sortKey === k) sortDir *= -1;
      else { sortKey = k; sortDir = k === "model" || k === "provider" ? 1 : -1; }
      render();
    })
  );
  render();
}

init();
