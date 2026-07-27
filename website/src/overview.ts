interface Stats {
  endpoints: number;
  providers: number;
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

interface FeedEvent {
  date: string;
  iso: string;
  daysAgo: number;
  model: string;
  provider: string;
  method: "lt" | "b3it";
  desc: string;
  primary: string;
  secondary: string;
  sevKey: "alert" | "changed" | "stable";
  trace: number[];
  changeFrac: number;
}

interface ProviderRate {
  name: string;
  n_endpoints: number;
  endpoint_years: number;
  months: number;
  n_changes: number;
  rate: number;
  conf: number;
}

type EndpointStatus = "stable" | "changed" | "retired";

interface EndpointRow {
  slug: string;
  model: string;
  org: string;
  provider: string;
  methods: string[];
  status: EndpointStatus;
  stableDays: number | null;
  nChanges: number;
  trace: number[];
}

type SortKey = "model" | "provider" | "status" | "nChanges" | "stableDays";

interface OverviewData {
  stats: Stats;
  feed: FeedEvent[];
  providers: ProviderRate[];
  endpoints: EndpointRow[];
}

(async function (): Promise<void> {
  const DATA: OverviewData = await (await fetch("data/overview.json")).json();
  const S = DATA.stats;
  const fmtInt = (n: number): string => n.toLocaleString("en-US");
  const fmtM = (n: number): string =>
    n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(0) + "k" : "" + n;
  const fmtMag = (v: number | null): string =>
    v == null ? "—" : v >= 100 ? v.toFixed(0) : v.toFixed(2);

  function sparkPath(
    vals: number[],
    w: number,
    h: number,
    pad: number,
    dom: [number, number] | null
  ): string {
    let min: number, max: number;
    if (dom) { min = dom[0]; max = dom[1]; } else { min = Math.min(...vals); max = Math.max(...vals); }
    const rng = (max - min) || 1;
    const step = (w - pad * 2) / (vals.length - 1);
    return vals.map((v, i) => {
      const cv = Math.max(min, Math.min(max, v));
      const x = pad + i * step, y = pad + (h - pad * 2) * (1 - (cv - min) / rng);
      return (i ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
    }).join(" ");
  }

  // hero trace: real entropy signals concatenated
  (function () {
    const svg = document.getElementById("heroTrace") as unknown as SVGSVGElement;
    let vals: number[] = [];
    DATA.endpoints.filter(e => e.trace && e.trace.length > 10).slice(0, 6).forEach(e => vals.push(...e.trace));
    if (vals.length < 20) vals = [0.1, 0.3, 0.2, 0.4, 0.25];
    const d = sparkPath(vals, 1200, 300, 8, null);
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
    { label: "Providers", value: fmtInt(S.providers), sub: `OpenRouter-routed` },
    { label: "Queries logged", value: fmtM(S.queries), sub: `since ${S.since}` },
    { label: "Changes detected", value: fmtInt(S.changes_total), sub: `${S.changes_lt} LT · ${S.changes_b3it} B3IT` },
  ];
  document.getElementById("telemetry")!.innerHTML = stats.map(s =>
    `<div class="stat"><div class="label">${s.label}</div><div class="value">${s.value}</div><div class="sub">${s.sub}</div></div>`).join("");
  const perM = S.spend_cumulative / (S.queries / 1e6);
  document.getElementById("cap")!.innerHTML =
    `Cheap enough to run continuously — <b>${fmtM(S.queries)}</b> logprob queries for <b>$${S.spend_cumulative.toFixed(2)}</b> total (~$${perM.toFixed(2)}/M). ` +
    `LT on ${S.lt_endpoints} endpoints since ${S.since}; B3IT on ${S.b3it_endpoints} since ${S.b3it_since} (${S.b3it_monitoring} still active).`;

  // feed (LT = entropy signal + Δ̄/σ · B3IT = TV signal + peak TV)
  const SEVC: Record<FeedEvent["sevKey"], string> = { alert: "var(--alert)", changed: "var(--changed)", stable: "var(--stable)" };
  document.getElementById("feed")!.innerHTML = DATA.feed.map(e => {
    const sev = SEVC[e.sevKey] || "var(--changed)";
    let spark = "";
    if (e.trace && e.trace.length > 4) {
      const d = sparkPath(e.trace, 132, 34, 3, null);
      const cx = 3 + (132 - 6) * e.changeFrac;
      spark = `<svg class="spark" viewBox="0 0 132 34" preserveAspectRatio="none">
        <line x1="${cx.toFixed(1)}" y1="2" x2="${cx.toFixed(1)}" y2="32" stroke="${sev}" stroke-width="1" stroke-dasharray="2 2" opacity="0.6"/>
        <path d="${d}" fill="none" stroke="${sev}" stroke-width="1.5"/></svg>`;
    }
    return `<div class="event" style="--sev:${sev}">
      <div class="when">${e.date.slice(5)}<span class="rel">${e.daysAgo}d ago</span></div>
      <div class="what"><div><a class="model" href="#">${e.model}</a> <span class="at">@ ${e.provider}</span></div><div class="desc">${e.desc}</div></div>
      ${spark}
      <div class="mag"><span class="badge ${e.method}">${e.method}</span>
        <div class="delta">${e.primary}</div>
        <div class="conf">${e.secondary}</div>
      </div>
    </div>`;
  }).join("");

  // monitoring-duration grid: 1 square = one endpoint-month, 12 per year-row, years stacked, wrapping into column-groups of YRS_PER_COL
  function volGrid(months: number): string {
    const YRS_PER_COL = 6, rows = Math.ceil(months / 12), groups = Math.max(1, Math.ceil(rows / YRS_PER_COL));
    let html = "";
    for (let g = 0; g < groups; g++) {
      let rowsHtml = "";
      for (let y = g * YRS_PER_COL; y < Math.min(rows, (g + 1) * YRS_PER_COL); y++) {
        const filled = Math.min(12, months - y * 12);
        let sq = "";
        for (let i = 0; i < 12; i++) sq += `<span class="sq${i < filled ? "" : " off"}"></span>`;
        rowsHtml += `<span class="yr">${sq}</span>`;
      }
      html += `<span class="grp">${rowsHtml}</span>`;
    }
    return `<span class="grid">${html}</span>`;
  }

  // provider drift rate (incl. stable providers; duration grid = confidence in the rate)
  const provs = DATA.providers;
  const maxRate = Math.max(...provs.map(p => p.rate), 1);
  document.getElementById("rates")!.innerHTML = provs.map((p, i) => {
    const lowConf = p.conf < 0.5 && p.rate > 0;
    const rateBar = p.rate === 0
      ? `<div class="rate-bar zero"><b>stable — 0 changes</b></div>`
      : `<div class="rate-bar"><span style="width:${Math.max(6, p.rate / maxRate * 100)}%;opacity:${(0.4 + 0.6 * p.conf).toFixed(2)}"></span><b>${p.rate.toFixed(1)}/yr</b></div>`;
    return `<div class="rate-row">
      <span class="rk">${i + 1}</span>
      <a class="pv" href="#">${p.name}</a>
      ${rateBar}
      <div class="vol">${volGrid(p.months)}<span class="lbl ${lowConf ? "low" : ""}">${p.endpoint_years}&thinsp;yr${lowConf ? " · low" : ""}</span></div>
      <span class="meta">${p.n_changes} chg</span>
    </div>`;
  }).join("");

  // directory
  const rows = DATA.endpoints;
  const active = new Set<string>();
  let sortKey: SortKey = "nChanges";
  let sortDir = -1;
  const STATUS_ORDER: Record<EndpointStatus, number> = { changed: 0, stable: 1, retired: 2 };
  const LABEL: Record<EndpointStatus, string> = { stable: "Stable", changed: "Changed", retired: "Retired" };
  const pill = (s: EndpointStatus): string => `<span class="pill ${s}"><span class="led"></span>${LABEL[s]}</span>`;
  const badges = (ms: string[]): string => `<div class="methods">${ms.map(m => `<span class="badge ${m}">${m}</span>`).join("")}</div>`;
  function stableCell(r: EndpointRow): string {
    if (r.status === "retired" || r.stableDays == null) return `<span class="prov-cell" style="color:var(--text-dim)">—</span>`;
    const d = r.stableDays; return `<span class="num">${d >= 365 ? (d / 365).toFixed(1) + "y" : d + "d"}</span>`;
  }
  function rowSpark(r: EndpointRow): string {
    if (!r.trace || r.trace.length < 4) return "";
    const d = sparkPath(r.trace, 130, 26, 3, [0, 1.5]);
    const col = r.status === "changed" ? "var(--changed)" : r.status === "retired" ? "var(--retired)" : "var(--stable)";
    return `<svg width="130" height="26" viewBox="0 0 130 26" preserveAspectRatio="none"><path d="${d}" fill="none" stroke="${col}" stroke-width="1.4" opacity="${r.status === 'retired' ? 0.4 : 0.95}"/></svg>`;
  }
  function render(): void {
    const q = (document.getElementById("q") as HTMLInputElement).value.trim().toLowerCase();
    const mf = [...active].filter(f => f === "lt" || f === "b3it");
    let list = rows.filter(r => {
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
      else if (sortKey === "stableDays") { av = a.stableDays == null ? -1 : a.stableDays; bv = b.stableDays == null ? -1 : b.stableDays; }
      else if (sortKey === "nChanges") { av = a.nChanges; bv = b.nChanges; }
      else { av = String(a[sortKey]).toLowerCase(); bv = String(b[sortKey]).toLowerCase(); }
      if (av < bv) return -sortDir; if (av > bv) return sortDir; return a.model.localeCompare(b.model);
    });
    document.getElementById("dirBody")!.innerHTML = list.map(r => `
      <tr>
        <td><a class="model-cell" href="#">${r.model}</a><div class="org-cell">${r.org}</div></td>
        <td class="col-hide"><span class="prov-cell">${r.provider}</span></td>
        <td>${pill(r.status)}</td>
        <td class="r"><span class="cc ${r.nChanges ? 'some' : 'zero'}">${r.nChanges}</span></td>
        <td class="col-hide">${badges(r.methods)}</td>
        <td class="r col-hide">${stableCell(r)}</td>
        <td class="col-hide">${rowSpark(r)}</td>
      </tr>`).join("");
    document.getElementById("dirFoot")!.textContent = `${list.length} of ${rows.length} endpoints`;
    document.querySelectorAll("thead th[data-sort] .arr").forEach(a => a.textContent = "");
    const th = document.querySelector(`thead th[data-sort="${sortKey}"] .arr`); if (th) th.textContent = sortDir > 0 ? "▲" : "▼";
  }
  document.getElementById("dirCount")!.innerHTML = `${fmtInt(rows.length)} endpoints · <b style="color:var(--changed)">${S.changes_total} changes</b> across ${S.changed_endpoints} of them`;
  document.getElementById("q")!.addEventListener("input", render);
  document.getElementById("chips")!.addEventListener("click", e => {
    const chip = (e.target as HTMLElement).closest(".chip") as HTMLElement | null; if (!chip) return;
    const f = chip.dataset.f!; active.has(f) ? (active.delete(f), chip.classList.remove("on")) : (active.add(f), chip.classList.add("on"));
    render();
  });
  document.querySelectorAll<HTMLElement>("thead th[data-sort]").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.sort as SortKey; if (sortKey === k) sortDir *= -1; else { sortKey = k; sortDir = (k === "stableDays" || k === "nChanges") ? -1 : 1; } render();
  }));
  render();
})();
