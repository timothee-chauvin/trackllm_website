// The provider drift-rate dot plot: one row per rateable provider company, a dot
// at the LT rate with its 95% Poisson interval as a whisker, on one shared axis.
// Drawn on the Overview (top slice) and the Providers page (every rateable row).
import { esc, plural } from "./components";
import { ProviderRate } from "./overview_data";

const NICE_STEPS = [0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10];
const MIN_TICKS = 4; // fewer and the axis reads as "0 ... 4"

/** Providers with a rate at all, most drift-prone first; a zero rate sorts by its
 *  upper bound so the best-monitored quiet provider comes last. */
export function rateablePlotRows(provs: ProviderRate[]): ProviderRate[] {
  return provs
    .filter(p => p.lt_rate !== null && p.lt_ci !== null)
    .sort((a, b) => b.lt_rate! - a.lt_rate! || b.lt_ci![1] - a.lt_ci![1] || a.name.localeCompare(b.name));
}

/** Tick positions 0..top at the coarsest nice step that still yields MIN_TICKS. */
export function axisTicks(top: number): number[] {
  const step = [...NICE_STEPS].reverse().find(s => top / s >= MIN_TICKS - 1) ?? NICE_STEPS[0];
  const ticks: number[] = [];
  for (let v = 0; v <= top + 1e-9; v += step) ticks.push(+v.toFixed(4));
  return ticks;
}

export function ratePlot(rows: ProviderRate[], root: string): string {
  if (!rows.length) return '<div class="empty">Nothing rateable yet.</div>';
  // the axis spans every whisker drawn, so no interval is ever clipped
  const top = Math.max(...rows.map(p => p.lt_ci![1]));
  const ticks = axisTicks(top);
  const domain = Math.max(top, ticks[ticks.length - 1]);
  const pc = (v: number): string => ((v / domain) * 100).toFixed(2) + "%";
  const grid = ticks.map(t => `<i class="gl" style="left:${pc(t)}"></i>`).join("");
  const body = rows.map(p => {
    const [lo, hi] = p.lt_ci!;
    const rate = p.lt_rate!;
    const what = `${rate.toFixed(2)} changes per endpoint-year (95% interval ${lo.toFixed(2)}–${hi.toFixed(2)}): ` +
      `${plural(p.lt_changes, "change")} in ${p.lt_years.toFixed(1)} endpoint-years`;
    return `<a class="rrow" href="${root}providers/${esc(p.slug)}.html" title="${esc(what)}">
      <span class="rname">${esc(p.name)}<small>${p.n_endpoints} ep · ${p.lt_years.toFixed(1)} ep-yr</small></span>
      <span class="rtrack">${grid}<i class="ci" style="left:${pc(lo)};width:${pc(hi - lo)}"></i><i class="dot" style="left:${pc(rate)}"></i></span>
      <span class="rval">${rate.toFixed(2)}</span>
    </a>`;
  }).join("");
  const axis = ticks.map(t => `<span style="left:${pc(t)}">${t}</span>`).join("");
  return `${body}<div class="raxis"><span></span><span class="ticks">${axis}</span><span></span></div>`;
}
