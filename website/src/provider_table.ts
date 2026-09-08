// The sortable, searchable provider-company table on the Providers page.
import { MIN_ENDPOINT_YEARS, brandHtml, esc, plural, rateBar, volGrid } from "./components";
import { initSortHeaders } from "./directory";
import { ProviderRate } from "./overview_data";
import { Method, methodRate } from "./rate_plot";

type RateKey = `${Method}_rate`;
type ProviderSortKey = "name" | "n_endpoints" | RateKey | "last_change";
const RATE_KEYS: RateKey[] = ["lt_rate", "b3it_rate"];

export function initProviderTable(provs: ProviderRate[]): void {
  // One scale for every bar in the table, both methods. Spans only what is drawn —
  // the rates and their interval upper bounds — so no band is clipped; zero-rate
  // rows render as text, so their (rule-of-three) ceilings must not set the scale.
  const drawn = provs.flatMap(p => (["lt", "b3it"] as Method[]).map(m => methodRate(p, m))).filter(r => r.rate !== null && r.rate > 0);
  const maxRate = Math.max(1, ...drawn.map(r => r.rate!), ...drawn.map(r => r.ci?.[1] ?? 0));

  // a method that never monitored the provider is a dash, not "not enough monitoring"
  const rateCell = (p: ProviderRate, m: Method): string => {
    const r = methodRate(p, m);
    return r.endpoints ? rateBar(r.years, r.rate, r.ci, maxRate) : '<span class="org-cell">—</span>';
  };

  const provQ = document.getElementById("provQ") as HTMLInputElement;
  const provSort = initSortHeaders<ProviderSortKey>(
    "psort", "lt_rate", -1, ["n_endpoints", ...RATE_KEYS, "last_change"], render);

  function render(): void {
    const q = provQ.value.trim().toLowerCase();
    const list = provs.filter(p => !q || `${p.name} ${p.brand.name}`.toLowerCase().includes(q));
    list.sort((a, b) => {
      let av: string | number, bv: string | number;
      if (RATE_KEYS.includes(provSort.key as RateKey)) {
        const k = provSort.key as RateKey;
        // an unmeasurable rate is not a low rate: park those rows at the bottom in
        // both directions, so reversing the sort never promotes them to the top
        if ((a[k] === null) !== (b[k] === null)) return a[k] === null ? 1 : -1;
        av = a[k] ?? 0; bv = b[k] ?? 0;
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
        <td><a class="model-cell" href="providers/${esc(p.slug)}.html">${brandHtml(p.brand, "")}</a>
          <div class="org-cell">${p.n_variants > 1 ? plural(p.n_variants, "serving variant") : "single variant"} · ${plural(p.n_models, "model")}</div></td>
        <td class="r"><span class="cc">${p.n_endpoints}</span></td>
        <td style="min-width:150px">${rateCell(p, "lt")}</td>
        <td class="col-hide">${volGrid(p.lt_years)}</td>
        <td style="min-width:150px">${rateCell(p, "b3it")}</td>
        <td class="col-hide">${p.b3it_endpoints
          ? `<span class="vol"><span class="lbl">${p.b3it_endpoints} ep · ${p.b3it_years.toFixed(1)} ep-yr</span></span>`
          : '<span class="org-cell">—</span>'}</td>
        <td class="r col-hide"><span class="cc ${p.last_change ? "some" : "zero"}">${p.last_change ? esc(p.last_change) : "—"}</span></td>
      </tr>`).join("") || '<tr><td colspan="7"><div class="empty">No providers match.</div></td></tr>';
    const unrated = provs.filter(p => p.lt_rate === null && p.b3it_rate === null).length;
    document.getElementById("provFoot")!.textContent =
      `${list.length} of ${provs.length} providers · ${unrated} under ${MIN_ENDPOINT_YEARS} endpoint-years by either method, so not yet rateable`;
    provSort.paintSort();
  }
  provQ.addEventListener("input", render);
  render();
}
