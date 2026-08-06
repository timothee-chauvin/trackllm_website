// The shared timeline: one strip per endpoint on one date axis, the all-endpoints
// strip above them, and the shared readout (model_hover.ts) across the lot. The
// model page names rows by provider and groups them under the company; the
// provider page is the mirror image (rows named by model, grouped per model).
// Everything else -- scales, strips, marks, readout, axis -- is common.
import {
  B3IT_CAP,
  MONTH_NAMES,
  esc,
  headlineBadge,
  methodBadges,
  monthTicks,
  plural,
  stripTip,
  td,
} from "./components";
import { areaPath, sampleAt, segments, strokePath } from "./chart_geom";
import { type StripRow, STRIP_VW, bindSharedHover, dayAt, readCells } from "./model_hover";

// drift/peakTV are null when the level the change reached is unknown: the series
// has no point on or after it (timeline.py, mirroring feed.py and endpoint.ts).
export interface LTChange {
  date: string;
  sigma: string;
  drift: number | null;
}

export interface B3ITChange {
  date: string;
  peakTV: number | null;
}

export interface EndpointStatusJSON {
  lt: string;
  bi: string;
  headline: string;
  reason: string;
}

export interface TimelineEndpoint {
  slug: string;
  provider: string;
  base: string;
  providerSlug: string;
  model: string;
  modelSlug: string;
  methods: string[];
  first: string | null;
  last: string | null;
  // the day the endpoint last answered a query, null if it never did. The rows
  // arrive sorted by it (timeline.py); the groups are sorted on it here.
  last_query: string | null;
  n_changes: number;
  // `breaks` are the indices into the drawn points that start a new run of
  // consecutive observed days -- the strip's series is thinned at build time, so
  // timeline.py has to publish where the missing days fell (see chart_geom.ts).
  lt: { drift: [string, number][]; breaks: number[]; changes: LTChange[] } | null;
  b3it: { tv: [string, number][]; breaks: number[]; changes: B3ITChange[] } | null;
  status: EndpointStatusJSON;
}

export interface TimelineChange {
  date: string;
  method: "lt" | "b3it";
  provider: string;
  model: string;
}

export interface TimelineData {
  date_min: string | null;
  date_max: string | null;
  changes: TimelineChange[];
  endpoints: TimelineEndpoint[];
}

/** How the page names its rows and groups them. */
export interface TimelineLabels {
  name(ep: TimelineEndpoint): string;
  changeName(c: TimelineChange): string;
  // the banner over sibling rows: what they share, and where its page is
  group(ep: TimelineEndpoint): { key: string; label: string; href: string; page: string };
}

/** A page can be all catalog, no series: rows without strips, and no axis. */
export const hasTimeline = (D: TimelineData): boolean =>
  D.endpoints.some((e) => e.methods.length) && !!D.date_min && !!D.date_max;

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

/** Render the whole panel into `panel` and wire the shared readout (which reads
 *  the #cmptip sibling model.html.j2 and provider.html.j2 both carry). */
export function renderTimeline(panel: HTMLElement, D: TimelineData, labels: TimelineLabels): void {
  const drawnAxis = hasTimeline(D);
  const changes = D.changes;
  const D0 = drawnAxis ? td(D.date_min!) : 0;
  const D1 = drawnAxis ? td(D.date_max!) : 1;
  const W = STRIP_VW;
  const xpos = (s: string): number => ((td(s) - D0) / (D1 - D0 || 1)) * W;

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
   *  (an lt lane can carry changes and no points at all -- timeline.py), on that
   *  method's own scale. Shared with the pointer readout, which puts its dot on the
   *  same curve. */
  function drawn(ep: TimelineEndpoint): {
    series: [string, number][];
    breaks: number[];
    dmax: number;
    col: string;
  } {
    const onLT = !!ep.lt?.drift.length;
    return {
      series: onLT ? ep.lt!.drift : ep.b3it ? ep.b3it.tv : [],
      breaks: (onLT ? ep.lt!.breaks : ep.b3it?.breaks) ?? [],
      dmax: onLT ? LT_MAX : B3IT_CAP,
      col: onLT || !ep.b3it ? "var(--accent)" : "var(--b3it)",
    };
  }

  /** One endpoint's drift line, with each change dot at the level that change
   *  reached. An LT dot and a B3IT dot never share a scale: nats go through the
   *  page-wide LT scale, total variation through its own 0..B3IT_CAP one. */
  function strip(ep: TimelineEndpoint): string {
    const { series: sig, breaks, dmax, col } = drawn(ep);
    const at = (p: [string, number]): string =>
      `${xpos(p[0]).toFixed(1)} ${stripY(p[1], dmax).toFixed(1)}`;
    const runs = segments(sig, breaks);
    const path = strokePath(runs, at);
    // The fill is this row's missing-data indicator, so it is closed on the run's own
    // first and last day -- not on the strip's edges, which would spread it across
    // every month the *page* spans and every day this endpoint went unobserved.
    const areas = areaPath(runs, (p) => xpos(p[0]), (p) => stripY(p[1], dmax), STRIP_H);
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
    return `<svg viewBox="0 0 ${W} ${STRIP_H}" preserveAspectRatio="none"${stripTip(labels.name(ep), marks.map((m) => m.title))}>${gridLines(STRIP_H)}
      ${path ? `<path d="${areas}" fill="${col}" opacity="0.10"/>
      <path d="${path}" fill="none" stroke="${col}" stroke-width="1.3" opacity="0.65" vector-effect="non-scaling-stroke"/>` : ""}
      ${dots}${HOVER_MARK}</svg>`;
  }

  /** Every change on the shared axis: when did this page's fleet move at all. */
  function allStrip(): string {
    const title = (c: TimelineChange): string =>
      `${c.date} · ${labels.changeName(c)} · ${c.method.toUpperCase()}`;
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
  function untrackedRow(ep: TimelineEndpoint): string {
    const epHref = `../endpoints/${esc(ep.slug)}.html`;
    return `<div class="row untracked">
      <div class="pv"><a href="${epHref}">${esc(labels.name(ep))}</a>
        <div class="mm">${headlineBadge(ep.status.headline)}</div></div>
      <div class="reason">${esc(ep.status.reason)}</div>
      <div class="meta"><a class="golink" href="${epHref}">endpoint →</a></div>
    </div>`;
  }

  function row(ep: TimelineEndpoint): string {
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
    // the row names one side of the pairing but identifies an endpoint: the
    // endpoint page is where both the model and the provider are one click away.
    const epHref = `../endpoints/${esc(ep.slug)}.html`;
    // `.read` is where the shared readout writes this row's value on the hovered
    // day, in place of the counts it stands next to the rest of the time.
    return `<div class="row" data-slug="${esc(ep.slug)}">
      <div class="pv"><a href="${epHref}">${esc(labels.name(ep))}</a>
        <div class="mm">${methodBadges(ep.methods)}${headlineBadge(ep.status.headline)}<a href="${epHref}">endpoint →</a></div></div>
      <div class="spark">${strip(ep)}</div>
      <div class="meta"><div class="static"><span class="${ep.n_changes ? "some" : "zero"}">${ep.n_changes || "—"} chg</span>
        <div class="peak">${ep.n_changes && last ? "last " + esc(last) : peak}</div></div>
        <div class="read" hidden aria-hidden="true"></div></div>
    </div>`;
  }

  /** Hand every drawn strip to the shared readout. Called once the panel is in the
   *  DOM: each row is found through the markup `row` and `allStrip` just wrote. */
  function bindReadout(): void {
    const tipEl = document.getElementById("cmptip");
    const wrap = panel.parentElement;
    if (!tipEl || !wrap) return;
    const strips: StripRow[] = [];

    const allSvg = panel.querySelector(".allrow .spark svg");
    const allMark = allSvg?.querySelector(".hover-mark");
    if (allSvg && allMark) {
      // the all-endpoints strip marks the days this fleet moved on, not a level:
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

  // one group of sibling rows per labels.group key, freshest first (rows keep the
  // order timeline.py sorted them in). A group is as fresh as its liveliest
  // endpoint, so one retired sibling can't sink the whole company; groups that
  // tie -- everything still queried today does -- fall back to most-changed.
  const byKey = new Map<string, TimelineEndpoint[]>();
  for (const e of D.endpoints) {
    const k = labels.group(e).key;
    byKey.set(k, (byKey.get(k) || []).concat(e));
  }
  const total = (list: TimelineEndpoint[]): number =>
    list.reduce((s, e) => s + e.n_changes, 0);
  const freshest = (list: TimelineEndpoint[]): string =>
    list.reduce((m, e) => (e.last_query && e.last_query > m ? e.last_query : m), "");
  const groups = [...byKey.values()].sort(
    (a, b) => freshest(b).localeCompare(freshest(a)) || total(b) - total(a)
  );

  const tracked = D.endpoints.filter((e) => e.methods.length);
  const nChanged = tracked.filter((e) => e.n_changes).length;

  panel.innerHTML =
    (drawnAxis
      ? `<div class="allrow">
      <div class="k">All endpoints<small>${plural(changes.length, "change")}</small></div>
      <div class="spark">${allStrip()}</div>
      <div class="meta"><span class="${nChanged ? "some" : "zero"}">${nChanged}/${tracked.length}</span><div class="peak">affected</div></div>
    </div>`
      : "") +
    groups
      .map((list) => {
        const n = total(list);
        const g = labels.group(list[0]);
        // The banner exists to tie sibling rows together. Over a lone row it says
        // nothing the row's own name doesn't -- `venice/fp8` already reads as
        // venice -- so it is a band of surface for no information.
        if (list.length <= 1) return list.map(row).join("");
        // grouped pages only exist for groups with tracked endpoints; an
        // all-untracked group names itself without linking anywhere
        const header = list.some((e) => e.methods.length)
          ? `<div class="grp-h"><a href="${g.href}">${esc(g.label)}</a>
               <span>${plural(list.length, "variant")} · ${plural(n, "change")} · ${g.page} →</span></div>`
          : `<div class="grp-h"><span class="base">${esc(g.label)}</span>
               <span>${plural(list.length, "variant")}</span></div>`;
        return header + list.map(row).join("");
      })
      .join("") +
    (drawnAxis
      ? `<div class="axis"><div class="ticks">${monthTicks(D0, D1)
          .filter((_, i) => i % 2 === 0)
          .map(
            (d) =>
              `<span style="left:${((xpos(d.toISOString().slice(0, 10)) / W) * 100).toFixed(2)}%">${MONTH_NAMES[d.getUTCMonth()]} ${String(d.getUTCFullYear()).slice(2)}</span>`
          )
          .join("")}</div></div>`
      : "");
  if (drawnAxis) bindReadout();
}
