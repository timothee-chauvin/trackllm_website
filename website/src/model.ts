// The `init` export both makes this a module (so its top-level names don't collide
// with other bundler entrypoints when type-checked as one tsc program) and lets the
// smoke tests re-render a fresh document without busting the module cache.
import {
  B3IT_CAP,
  MONTH_NAMES,
  bindTips,
  esc,
  headlineBadge,
  methodBadges,
  monthTicks,
  plural,
  prettyDate,
  showLoadError,
  stripTip,
  td,
} from "./components";
import { sampleAt } from "./chart_geom";
import { type StripRow, STRIP_VW, bindSharedHover, dayAt, readCells } from "./model_hover";

// drift/peakTV are null when the level the change reached is unknown: the series
// has no point on or after it (model.py, mirroring feed.py and endpoint.ts).
interface LTChange {
  date: string;
  sigma: string;
  drift: number | null;
}

interface B3ITChange {
  date: string;
  peakTV: number | null;
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

// Strip geometry, in the STRIP_VW-wide units every strip on the shared timeline is
// drawn in. The all-endpoints strip is shorter: it marks days, not levels.
const STRIP_H = 40;
const STRIP_PAD = 5;
const ALL_H = 34;

const stripY = (v: number, dmax: number): number =>
  STRIP_H - STRIP_PAD - (Math.min(v, dmax) / dmax) * (STRIP_H - 2 * STRIP_PAD);

/** The group every strip reserves for the shared readout's crosshair, last so
 *  nothing in the strip is drawn over it. */
const HOVER_MARK = `<g class="hover-mark" aria-hidden="true"></g>`;

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
  const W = STRIP_VW;
  const xpos = (s: string): number => ((td(s) - D0) / (D1 - D0 || 1)) * W;

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
      .flatMap((e) =>
        e.lt!.drift
          .map((p) => p[1])
          .concat(e.lt!.changes.map((c) => c.drift).filter((d) => d !== null))
      )
  );

  /** `y` is null when the level is unknown: the dot has nowhere honest to sit. */
  interface Mark {
    date: string;
    y: number | null;
    color: string;
    title: string;
  }

  /** The line a strip draws: the LT lane when it has points, the B3IT one otherwise
   *  (an lt lane can carry changes and no points at all -- model.py), on that
   *  method's own scale. Shared with the pointer readout, which puts its dot on the
   *  same curve. */
  function drawn(ep: ModelEndpoint): { series: [string, number][]; dmax: number; col: string } {
    const onLT = !!ep.lt?.drift.length;
    return {
      series: onLT ? ep.lt!.drift : ep.b3it ? ep.b3it.tv : [],
      dmax: onLT ? LT_MAX : B3IT_CAP,
      col: onLT || !ep.b3it ? "var(--accent)" : "var(--b3it)",
    };
  }

  /** One provider's drift line, with each change dot at the level that change reached.
   *  An LT dot and a B3IT dot never share a scale: nats go through the model-wide LT
   *  scale, total variation through its own 0..B3IT_CAP one. */
  function strip(ep: ModelEndpoint): string {
    const { series: sig, dmax, col } = drawn(ep);
    const path =
      sig.length > 1
        ? sig
            .map((p, i) => `${i ? "L" : "M"}${xpos(p[0]).toFixed(1)} ${stripY(p[1], dmax).toFixed(1)}`)
            .join(" ")
        : "";
    const marks: Mark[] = [
      ...(ep.lt
        ? ep.lt.changes.map(
            (c): Mark => ({
              date: c.date,
              y: c.drift === null ? null : stripY(c.drift, LT_MAX),
              color: "var(--accent)",
              title: `LT ${c.date} · ${c.sigma}, drift ${c.drift === null ? "—" : `${c.drift} nats`}`,
            })
          )
        : []),
      ...(ep.b3it
        ? ep.b3it.changes.map(
            (c): Mark => ({
              date: c.date,
              y: c.peakTV === null ? null : stripY(c.peakTV, B3IT_CAP),
              color: "var(--b3it)",
              title: `B3IT ${c.date} · peak TV ${c.peakTV === null ? "—" : c.peakTV}`,
            })
          )
        : []),
    ];
    const dots = marks
      .map((m) => {
        const x = xpos(m.date).toFixed(1);
        // Unknown level: a dashed rule the full height of the strip, never a dot —
        // a dot sits at a level, and this change was never measured at one.
        if (m.y === null) {
          return `<line x1="${x}" y1="3" x2="${x}" y2="${STRIP_H - 3}" stroke="${m.color}" stroke-width="1" stroke-dasharray="2 2" opacity="0.6"><title>${esc(m.title)}</title></line>`;
        }
        return `<line x1="${x}" y1="${STRIP_H - 3}" x2="${x}" y2="${m.y.toFixed(1)}" stroke="${m.color}" stroke-width="1" opacity="0.35"/>
      <circle cx="${x}" cy="${m.y.toFixed(1)}" r="3.4" fill="${m.color}"><title>${esc(m.title)}</title></circle>`;
      })
      .join("");
    return `<svg viewBox="0 0 ${W} ${STRIP_H}" preserveAspectRatio="none"${stripTip(ep.provider, marks.map((m) => m.title))}>${gridLines(STRIP_H)}
      ${path ? `<path d="${path} L${W} ${STRIP_H} L0 ${STRIP_H} Z" fill="${col}" opacity="0.10"/>
      <path d="${path}" fill="none" stroke="${col}" stroke-width="1.3" opacity="0.65" vector-effect="non-scaling-stroke"/>` : ""}
      ${dots}${HOVER_MARK}</svg>`;
  }

  /** Every change for the model on the shared axis: when did this model move at all. */
  function allStrip(): string {
    const title = (c: ModelChange): string =>
      `${c.date} · ${c.provider} · ${c.method.toUpperCase()}`;
    const marks = changes
      .map((c) => {
        const x = xpos(c.date).toFixed(1);
        const col = c.method === "lt" ? "var(--accent)" : "var(--b3it)";
        return `<line x1="${x}" y1="6" x2="${x}" y2="${ALL_H - 6}" stroke="${col}" stroke-width="2" opacity="0.85"><title>${esc(title(c))}</title></line>`;
      })
      .join("");
    return `<svg viewBox="0 0 ${W} ${ALL_H}" preserveAspectRatio="none"${stripTip("All endpoints", changes.map(title))}>${gridLines(ALL_H)}${marks}${HOVER_MARK}</svg>`;
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
    // on the points, not on the lane: Math.max of an empty lane is -Infinity
    const peak = ep.lt?.drift.length
      ? `${Math.max(...ep.lt.drift.map((p) => p[1])).toFixed(2)} nats`
      : ep.b3it?.tv.length
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
    // `.read` is where the shared readout writes this row's value on the hovered
    // day, in place of the counts it stands next to the rest of the time.
    return `<div class="row" data-slug="${esc(ep.slug)}">
      <div class="pv"><a href="${epHref}">${esc(ep.provider)}</a>
        <div class="mm">${methodBadges(ep.methods)}${headlineBadge(ep.status.headline)}<a href="${epHref}">endpoint →</a></div></div>
      <div class="spark">${strip(ep)}</div>
      <div class="meta"><div class="static"><span class="${ep.n_changes ? "some" : "zero"}">${ep.n_changes || "—"} chg</span>
        <div class="peak">${ep.n_changes && last ? "last " + esc(last) : peak}</div></div>
        <div class="read" hidden aria-hidden="true"></div></div>
    </div>`;
  }

  /** Hand every drawn strip to the shared readout. Called once the panel is in the
   *  DOM: each row is found through the markup `row` and `allStrip` just wrote. */
  function bindReadout(panel: HTMLElement): void {
    const tipEl = document.getElementById("cmptip");
    const wrap = panel.parentElement;
    if (!tipEl || !wrap) return;
    const strips: StripRow[] = [];

    const allSvg = panel.querySelector(".allrow .spark svg");
    const allMark = allSvg?.querySelector(".hover-mark");
    if (allSvg && allMark) {
      // the all-endpoints strip marks the days this model moved on, not a level:
      // it takes the crosshair and leaves the values to the rows under it
      strips.push({
        svg: allSvg, mark: allMark, height: ALL_H, read: null, stat: null,
        dot: () => null, cells: () => [],
      });
    }

    const bySlug = new Map(D.endpoints.map((e) => [e.slug, e]));
    for (const el of panel.querySelectorAll<HTMLElement>(".row[data-slug]")) {
      const ep = bySlug.get(el.dataset.slug ?? "");
      const svg = el.querySelector(".spark svg");
      const mark = svg?.querySelector(".hover-mark");
      const read = el.querySelector<HTMLElement>(".meta .read");
      const stat = el.querySelector<HTMLElement>(".meta .static");
      if (!ep || !svg || !mark || !read || !stat) continue;
      const { series, dmax, col } = drawn(ep);
      strips.push({
        svg, mark, height: STRIP_H, read, stat,
        // the dot sits on the curve the strip actually draws, at the shared date --
        // so a row whose series stops short shows the rule alone
        dot: (date) => {
          const v = sampleAt(series, date);
          return v === null ? null : { y: stripY(v, dmax), col };
        },
        cells: (date) => readCells(ep, date),
      });
    }
    bindSharedHover(panel, wrap, tipEl, strips, xpos, (f) => dayAt(D0, D1, f));
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
      <div class="spark">${allStrip()}</div>
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
  // document.body, not cmpEl: the status badge up in #summary (headlineBadge, above)
  // wants the same popover as the ones in #cmp, and one binding covers both.
  bindTips(document.body);
  if (hasTimeline) bindReadout(cmpEl);

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
