// The `init` export both makes this a module (so its top-level names don't collide
// with other bundler entrypoints when type-checked as one tsc program) and lets the
// smoke tests re-render a fresh document without busting the module cache.
import {
  B3IT_CAP,
  MONTH_NAMES,
  esc,
  headlineBadge,
  methodBadges,
  monthTicks,
  plural,
  prettyDate,
  showLoadError,
  td,
} from "./components";

interface LTChange {
  date: string;
  sigma: string;
  drift: number;
}

interface B3ITChange {
  date: string;
  peakTV: number;
}

interface EndpointLT {
  drift: [string, number][];
  changes: LTChange[];
}

interface EndpointB3IT {
  tv: [string, number][];
  changes: B3ITChange[];
}

interface EndpointStatusJSON {
  lt: string;
  bi: string;
  headline: string;
  reason: string;
}

interface ModelEndpoint {
  slug: string;
  provider: string;
  base: string;
  providerSlug: string;
  methods: string[];
  first: string | null;
  last: string | null;
  n_changes: number;
  lt: EndpointLT | null;
  b3it: EndpointB3IT | null;
  status: EndpointStatusJSON;
}

interface ModelChange {
  date: string;
  method: "lt" | "b3it";
  provider: string;
}

interface ModelData {
  model: string;
  org: string;
  date_min: string | null;
  date_max: string | null;
  n_endpoints: number;
  n_providers: number;
  n_endpoints_total: number;
  n_changed: number;
  max_drift: number;
  headline: string;
  status_summary: string;
  changes: ModelChange[];
  endpoints: ModelEndpoint[];
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
  const hasTimeline = tracked.length > 0 && !!D.date_min && !!D.date_max;
  if (!hasTimeline) {
    document.getElementById("cmpDesc")?.remove();
    document.getElementById("cmpLegend")?.remove();
    // there is no drift to show, only per-endpoint status rows
    const title = document.getElementById("cmpTitle");
    if (title) title.textContent = "Endpoints";
  }

  const changes = D.changes; // hoisted so the nested renderers keep the non-null narrowing
  const D0 = hasTimeline ? td(D.date_min!) : 0;
  const D1 = hasTimeline ? td(D.date_max!) : 1;
  const W = 1000;
  const xpos = (s: string): number => ((td(s) - D0) / (D1 - D0 || 1)) * W;

  const ledeEl = document.getElementById("lede");
  if (ledeEl) {
    // n_endpoints counts serving endpoints, n_providers the companies behind them:
    // saying "providers" for the larger number would contradict the groups below.
    ledeEl.innerHTML = tracked.length
      ? `Served by <b>${D.n_providers}</b> ${D.n_providers === 1 ? "provider" : "providers"}` +
        ` on ${plural(D.n_endpoints, "tracked endpoint")}. ` +
        `<span class="hl">${D.n_changed}</span> of those ${D.n_changed === 1 ? "shows" : "show"}` +
        ` at least one detected change since launch — evidence the served behaviour drifts even when the model version doesn't.` +
        ` ${esc(D.status_summary)} across the catalog.`
      : `This model is not tracked: ${esc(D.status_summary)}. Each endpoint below says why.`;
  }
  const summaryEl = document.getElementById("summary");
  if (summaryEl) {
    summaryEl.innerHTML = tracked.length
      ? `
      <div class="s"><div class="v">${D.n_endpoints}</div><div class="k">Endpoints</div></div>
      <div class="s"><div class="v" style="color:var(--changed)">${D.n_changed}</div><div class="k">With changes</div></div>
      <div class="s"><div class="v">${changes.length}</div><div class="k">Changes total</div></div>
      <div class="s"><div class="v">${prettyDate(D.date_min)} – ${prettyDate(D.date_max)}</div><div class="k">Monitored</div></div>`
      : `
      <div class="s"><div class="v">${D.n_endpoints_total}</div><div class="k">Catalog endpoints</div></div>
      <div class="s"><div class="v">${headlineBadge(D.headline)}</div><div class="k">Status</div></div>`;
  }

  function gridLines(H: number): string {
    return monthTicks(D0, D1)
      .map((d) => {
        const x = xpos(d.toISOString().slice(0, 10)).toFixed(1);
        return `<line x1="${x}" y1="2" x2="${x}" y2="${H - 2}" stroke="var(--border-soft)" stroke-width="1"/>`;
      })
      .join("");
  }

  // shared per-method scales so strip heights are directly comparable. The changes
  // are unioned in because a change's peak drift can exceed the downsampled series.
  const LT_MAX = Math.max(
    0.5,
    ...D.endpoints
      .filter((e) => e.lt)
      .flatMap((e) => e.lt!.drift.map((p) => p[1]).concat(e.lt!.changes.map((c) => c.drift)))
  );

  interface Mark {
    date: string;
    y: number;
    color: string;
    title: string;
  }

  /** One provider's drift line, with each change dot at the level that change reached.
   *  An LT dot and a B3IT dot never share a scale: nats go through the model-wide LT
   *  scale, total variation through its own 0..B3IT_CAP one. */
  function strip(ep: ModelEndpoint): string {
    const sig = ep.lt ? ep.lt.drift : ep.b3it ? ep.b3it.tv : [];
    const H = 40, pad = 5;
    const y = (v: number, dmax: number): number =>
      H - pad - (Math.min(v, dmax) / dmax) * (H - 2 * pad);
    const path =
      sig.length > 1
        ? sig
            .map(
              (p, i) =>
                `${i ? "L" : "M"}${xpos(p[0]).toFixed(1)} ${y(p[1], ep.lt ? LT_MAX : B3IT_CAP).toFixed(1)}`
            )
            .join(" ")
        : "";
    const col = ep.lt ? "var(--accent)" : "var(--b3it)";
    const marks: Mark[] = [
      ...(ep.lt
        ? ep.lt.changes.map(
            (c): Mark => ({
              date: c.date,
              y: y(c.drift, LT_MAX),
              color: "var(--accent)",
              title: `LT ${c.date} · ${c.sigma}, drift ${c.drift} nats`,
            })
          )
        : []),
      ...(ep.b3it
        ? ep.b3it.changes.map(
            (c): Mark => ({
              date: c.date,
              y: y(c.peakTV, B3IT_CAP),
              color: "var(--b3it)",
              title: `B3IT ${c.date} · peak TV ${c.peakTV}`,
            })
          )
        : []),
    ];
    const dots = marks
      .map((m) => {
        const x = xpos(m.date).toFixed(1);
        return `<line x1="${x}" y1="${H - 3}" x2="${x}" y2="${m.y.toFixed(1)}" stroke="${m.color}" stroke-width="1" opacity="0.35"/>
      <circle cx="${x}" cy="${m.y.toFixed(1)}" r="3.4" fill="${m.color}"><title>${esc(m.title)}</title></circle>`;
      })
      .join("");
    return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${gridLines(H)}
      ${path ? `<path d="${path} L${W} ${H} L0 ${H} Z" fill="${col}" opacity="0.10"/>
      <path d="${path}" fill="none" stroke="${col}" stroke-width="1.3" opacity="0.65" vector-effect="non-scaling-stroke"/>` : ""}
      ${dots}</svg>`;
  }

  /** Every change for the model on the shared axis: when did this model move at all. */
  function allStrip(): string {
    const H = 34;
    const marks = changes
      .map((c) => {
        const x = xpos(c.date).toFixed(1);
        const col = c.method === "lt" ? "var(--accent)" : "var(--b3it)";
        return `<line x1="${x}" y1="6" x2="${x}" y2="${H - 6}" stroke="${col}" stroke-width="2" opacity="0.85"><title>${esc(`${c.date} · ${c.provider} · ${c.method.toUpperCase()}`)}</title></line>`;
      })
      .join("");
    return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${gridLines(H)}${marks}</svg>`;
  }

  /** A catalog endpoint with no series: headline badge + reason where the strip would be. */
  function untrackedRow(ep: ModelEndpoint): string {
    const epHref = `../endpoints/${esc(ep.slug)}.html`;
    return `<div class="row untracked">
      <div class="pv"><a href="${epHref}">${esc(ep.provider)}</a>
        <div class="mm">${headlineBadge(ep.status.headline)}</div></div>
      <div class="reason">${esc(ep.status.reason)}</div>
      <div class="meta"><a class="golink" href="${epHref}">endpoint →</a></div>
    </div>`;
  }

  function row(ep: ModelEndpoint): string {
    if (!ep.methods.length) return untrackedRow(ep);
    const peak = ep.lt
      ? `${Math.max(...ep.lt.drift.map((p) => p[1])).toFixed(2)} nats`
      : ep.b3it
        ? `TV ${Math.max(...ep.b3it.tv.map((p) => p[1])).toFixed(2)}`
        : "—";
    const last = [
      ...(ep.lt ? ep.lt.changes.map((c) => c.date) : []),
      ...(ep.b3it ? ep.b3it.changes.map((c) => c.date) : []),
    ]
      .sort()
      .pop();
    // the row names a provider but identifies an endpoint: the endpoint page is where
    // both this model and that provider are one click away.
    const epHref = `../endpoints/${esc(ep.slug)}.html`;
    return `<div class="row">
      <div class="pv"><a href="${epHref}">${esc(ep.provider)}</a>
        <div class="mm">${methodBadges(ep.methods)}${headlineBadge(ep.status.headline)}<a href="${epHref}">endpoint →</a></div></div>
      <div class="spark">${strip(ep)}</div>
      <div class="meta"><span class="${ep.n_changes ? "some" : "zero"}">${ep.n_changes || "—"} chg</span>
        <div class="peak">${ep.n_changes && last ? "last " + esc(last) : peak}</div></div>
    </div>`;
  }

  // one group per provider company, most-changed first
  const byBase = new Map<string, ModelEndpoint[]>();
  for (const e of D.endpoints) {
    byBase.set(e.base, (byBase.get(e.base) || []).concat(e));
  }
  const total = (list: ModelEndpoint[]): number =>
    list.reduce((s, e) => s + e.n_changes, 0);
  const groups = [...byBase.entries()].sort((a, b) => total(b[1]) - total(a[1]));

  cmpEl.innerHTML =
    (hasTimeline
      ? `<div class="allrow">
      <div class="k">All endpoints<small>${plural(changes.length, "change")}</small></div>
      <div>${allStrip()}</div>
      <div class="meta"><span class="${D.n_changed ? "some" : "zero"}">${D.n_changed}/${D.n_endpoints}</span><div class="peak">affected</div></div>
    </div>`
      : "") +
    groups
      .map(([base, list]) => {
        const n = total(list);
        // The banner exists to tie sibling rows to one company. Over a lone row it
        // says nothing the row's own name doesn't -- `venice/fp8` already reads as
        // venice -- so it is a band of surface for no information.
        if (list.length <= 1) return list.map(row).join("");
        // provider pages only exist for providers with tracked endpoints; an
        // all-untracked group names its company without linking it
        const header = list.some((e) => e.methods.length)
          ? `<div class="grp-h"><a href="../providers/${esc(list[0].providerSlug)}.html">${esc(base)}</a>
               <span>${plural(list.length, "variant")} · ${plural(n, "change")} · provider page →</span></div>`
          : `<div class="grp-h"><span class="base">${esc(base)}</span>
               <span>${plural(list.length, "variant")}</span></div>`;
        return header + list.map(row).join("");
      })
      .join("") +
    (hasTimeline ? `<div class="axis"><div class="ticks" id="ticks"></div></div>` : "");

  const ticksEl = document.getElementById("ticks");
  if (ticksEl) {
    ticksEl.innerHTML = monthTicks(D0, D1)
      .filter((_, i) => i % 2 === 0)
      .map(
        (d) =>
          `<span style="left:${((xpos(d.toISOString().slice(0, 10)) / W) * 100).toFixed(2)}%">${MONTH_NAMES[d.getUTCMonth()]} ${String(d.getUTCFullYear()).slice(2)}</span>`
      )
      .join("");
  }
}

init();
