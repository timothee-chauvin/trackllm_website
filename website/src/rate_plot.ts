// The provider drift-rate dot plot: one row per rateable provider company, a dot
// at the rate with its 95% Poisson interval as a whisker, on one shared axis.
// Drawn on the Overview (top slice, LT only) and the Providers page (every
// rateable row, in the method the reader picks: LT, B3IT, or both stacked).
import { MIN_ENDPOINT_YEARS, brandHtml, esc, plural } from "./components";
import { ProviderRate, Stats } from "./overview_data";

const NICE_STEPS = [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10];
const MIN_TICKS = 4; // fewer and the axis reads as "0 ... 4"

export type Method = "lt" | "b3it";
export type RateMode = Method | "both";
export const RATE_MODES: RateMode[] = ["lt", "b3it", "both"];
export const METHOD_LABEL: Record<Method, string> = { lt: "LT", b3it: "B3IT" };
const modeMethods = (mode: RateMode): Method[] => mode === "both" ? ["lt", "b3it"] : [mode];

/** One method's block of a provider row (provider.py::_method_block, flattened). */
interface MethodRate {
  endpoints: number;
  years: number;
  changes: number;
  rate: number | null;
  ci: [number, number] | null;
}
export const methodRate = (p: ProviderRate, m: Method): MethodRate => ({
  endpoints: p[`${m}_endpoints`], years: p[`${m}_years`], changes: p[`${m}_changes`],
  rate: p[`${m}_rate`], ci: p[`${m}_ci`],
});
const rated = (p: ProviderRate, m: Method): boolean => p[`${m}_rate`] !== null && p[`${m}_ci`] !== null;

/** Providers with a rate at all, most drift-prone first; a zero rate sorts by its
 *  upper bound so the best-monitored quiet provider comes last. In "both", a row
 *  needs either rate, rows with both come first, and a row ranks by its higher
 *  rate. */
export function rateablePlotRows(provs: ProviderRate[], mode: RateMode): ProviderRate[] {
  const ms = modeMethods(mode);
  const has = (p: ProviderRate): number => ms.filter(m => rated(p, m)).length;
  const best = (p: ProviderRate, f: (r: MethodRate) => number): number =>
    Math.max(...ms.filter(m => rated(p, m)).map(m => f(methodRate(p, m))));
  return provs
    .filter(p => has(p) > 0)
    .sort((a, b) => has(b) - has(a)
      || best(b, r => r.rate!) - best(a, r => r.rate!)
      || best(b, r => r.ci![1]) - best(a, r => r.ci![1])
      || a.name.localeCompare(b.name));
}

/** Tick positions 0..top at the coarsest nice step that still yields MIN_TICKS. */
export function axisTicks(top: number): number[] {
  const step = [...NICE_STEPS].reverse().find(s => top / s >= MIN_TICKS - 1) ?? NICE_STEPS[0];
  const ticks: number[] = [];
  for (let v = 0; v <= top + 1e-9; v += step) ticks.push(+v.toFixed(4));
  return ticks;
}

const rateSentence = (r: MethodRate): string =>
  `${r.rate!.toFixed(2)} changes per endpoint-year (95% interval ${r.ci![0].toFixed(2)}–${r.ci![1].toFixed(2)}): ` +
  `${plural(r.changes, "change")} in ${r.years.toFixed(1)} endpoint-years`;

export function ratePlot(rows: ProviderRate[], root: string, mode: RateMode): string {
  if (!rows.length) return '<div class="empty">Nothing rateable yet.</div>';
  const ms = modeMethods(mode);
  const both = mode === "both";
  // the axis spans every whisker drawn, so no interval is ever clipped
  const top = Math.max(...rows.flatMap(p => ms.filter(m => rated(p, m)).map(m => p[`${m}_ci`]![1])));
  const ticks = axisTicks(top);
  const domain = Math.max(top, ticks[ticks.length - 1]);
  const pc = (v: number): string => ((v / domain) * 100).toFixed(2) + "%";
  const grid = ticks.map(t => `<i class="gl" style="left:${pc(t)}"></i>`).join("");
  // LT is the accent by default and carries no modifier, so the LT plot's markup is the same on every page
  const cls = (m: Method): string => m === "lt" ? "" : ` ${m}`;
  const body = rows.map(p => {
    const drawn = ms.filter(m => rated(p, m));
    const what = drawn.map(m => (both ? `${METHOD_LABEL[m]}: ` : "") + rateSentence(methodRate(p, m))).join("\n");
    const years = ms.map(m => p[`${m}_years`].toFixed(1)).join(" / ");
    const marks = drawn.map(m => {
      const [lo, hi] = p[`${m}_ci`]!;
      return `<i class="ci${cls(m)}" style="left:${pc(lo)};width:${pc(hi - lo)}"></i><i class="dot${cls(m)}" style="left:${pc(p[`${m}_rate`]!)}"></i>`;
    }).join("");
    const vals = both
      ? ms.map(m => rated(p, m) ? `<span class="${m}">${p[`${m}_rate`]!.toFixed(2)}</span>` : '<span class="none">—</span>').join("")
      : p[`${ms[0]}_rate`]!.toFixed(2);
    return `<a class="rrow" href="${root}providers/${esc(p.slug)}.html" title="${esc(what)}">
      <span class="rname">${brandHtml(p.brand, root)}<small>${p.n_endpoints} ep · ${years} ep-yr</small></span>
      <span class="rtrack">${grid}${marks}</span>
      <span class="rval">${vals}</span>
    </a>`;
  }).join("");
  const axis = ticks.map(t => `<span style="left:${pc(t)}">${t}</span>`).join("");
  return `${body}<div class="raxis"><span></span><span class="ticks">${axis}</span><span></span></div>`;
}

/** What one method's plot leaves out: the changes at providers too thin for a rate. */
export interface MethodTotals {
  rateable: number;
  changes: number; // at rateable providers, i.e. on the plot
  allChanges: number;
  years: number; // at rateable providers
  thin: number; // providers with changes but no rate
  thinChanges: number;
}
export function methodTotals(provs: ProviderRate[], m: Method): MethodTotals {
  const rs = provs.filter(p => rated(p, m)).map(p => methodRate(p, m));
  const thin = provs.filter(p => !rated(p, m)).map(p => methodRate(p, m)).filter(r => r.changes);
  const sum = (xs: MethodRate[], f: (r: MethodRate) => number): number => xs.reduce((s, r) => s + f(r), 0);
  return {
    rateable: rs.length, changes: sum(rs, r => r.changes), allChanges: sum(provs.map(p => methodRate(p, m)), r => r.changes),
    years: sum(rs, r => r.years), thin: thin.length, thinChanges: sum(thin, r => r.changes),
  };
}

export function rateStatLine(provs: ProviderRate[], mode: RateMode): string {
  return modeMethods(mode).map(m => {
    const t = methodTotals(provs, m);
    return `<b>${t.rateable}</b> rateable providers · <b>${t.changes}</b> of ${t.allChanges} ${METHOD_LABEL[m]} changes` +
      ` over <b>${t.years.toFixed(1)}</b> ep-yr` +
      (t.thinChanges ? ` (${t.thinChanges} more at ${plural(t.thin, "provider")} under the ${MIN_ENDPOINT_YEARS} ep-yr floor)` : "");
  }).join(" &nbsp;|&nbsp; ");
}

export function rateLegend(mode: RateMode): string {
  const color = (m: Method): string => m === "lt" ? "var(--accent)" : "var(--b3it)";
  return modeMethods(mode).map(m => `<span class="k"><i style="background:${color(m)}"></i>${METHOD_LABEL[m]} rate, 95% interval</span>`).join("") +
    (mode === "both" ? '<span class="k">— = not rateable by that method</span>' : "");
}

export function rateNote(provs: ProviderRate[], stats: Stats, mode: RateMode): string {
  if (mode === "both") {
    return "<b>Both methods on one axis.</b> Rows that have both rates come first. Where the two disagree, look at the" +
      " exposure under the name (LT / B3IT endpoint-years): B3IT rates rest on far fewer endpoint-years.";
  }
  if (mode === "lt") {
    const none = provs.filter(p => !p.lt_endpoints).length;
    return "<b>LT covers only providers that expose logprobs.</b> Providers that never return logprobs" +
      ` (${none} of the ${provs.length} here) cannot appear at all, which is the fairness gap B3IT closes.`;
  }
  const t = methodTotals(provs, "b3it");
  const monitored = provs.filter(p => p.b3it_endpoints);
  const underYear = monitored.filter(p => p.b3it_years < 1).length;
  const since = stats.b3it_since ? ` in ${stats.b3it_since}` : "";
  return `<b>B3IT exposure is thin:</b> monitoring only started${since}` +
    ` and ${underYear} of the ${monitored.length} providers under it have under a year of endpoint-time, so intervals` +
    ` are wide and a single change can put a provider at the top. ${t.thinChanges} of ${t.allChanges} B3IT changes sit at providers` +
    ` with under ${MIN_ENDPOINT_YEARS} endpoint-years and are not on the plot. A provider needs at least ${MIN_ENDPOINT_YEARS}` +
    " endpoint-years to get a rate at all (same rule as LT).";
}
