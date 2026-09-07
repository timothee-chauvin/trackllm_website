// The sortable, searchable provider-company table on the Providers page.
import { MIN_ENDPOINT_YEARS, brandHtml, esc, plural, rateBar, volGrid } from "./components";
import { initSortHeaders } from "./directory";
import { ProviderRate } from "./overview_data";

type ProviderSortKey = "name" | "n_endpoints" | "lt_rate" | "last_change";

export function initProviderTable(provs: ProviderRate[]): void {
  // One scale for every bar in the table. Spans only what is drawn — the rates
  // and their interval upper bounds — so no band is clipped; zero-rate rows render
  // as text, so their (rule-of-three) ceilings must not set the scale.
  const drawn = provs.filter(p => p.lt_rate !== null && p.lt_rate > 0);
  const maxRate = Math.max(1, ...drawn.map(p => p.lt_rate!), ...drawn.map(p => p.lt_ci?.[1] ?? 0));

  const provQ = document.getElementById("provQ") as HTMLInputElement;
  const provSort = initSortHeaders<ProviderSortKey>(
    "psort", "lt_rate", -1, ["n_endpoints", "lt_rate", "last_change"], render);

  function render(): void {
    const q = provQ.value.trim().toLowerCase();
    const list = provs.filter(p => !q || `${p.name} ${p.brand.name}`.toLowerCase().includes(q));
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
        <td><a class="model-cell" href="providers/${esc(p.slug)}.html">${brandHtml(p.brand, "")}</a>
          <div class="org-cell">${p.n_variants > 1 ? plural(p.n_variants, "serving variant") : "single variant"} · ${plural(p.n_models, "model")}</div></td>
        <td class="r"><span class="cc">${p.n_endpoints}</span></td>
        <td style="min-width:150px">${rateBar(p.lt_years, p.lt_rate, p.lt_ci, maxRate)}</td>
        <td class="col-hide">${volGrid(p.lt_years)}</td>
        <td class="col-hide">${p.b3it_endpoints
          ? `<span class="vol"><span class="lbl">${p.b3it_endpoints} ep · ${p.b3it_years.toFixed(1)} ep-yr</span></span>`
          : '<span class="org-cell">—</span>'}</td>
        <td class="r col-hide"><span class="cc ${p.last_change ? "some" : "zero"}">${p.last_change ? esc(p.last_change) : "—"}</span></td>
      </tr>`).join("") || '<tr><td colspan="6"><div class="empty">No providers match.</div></td></tr>';
    const unrated = provs.filter(p => p.lt_rate === null).length;
    document.getElementById("provFoot")!.textContent =
      `${list.length} of ${provs.length} providers · ${unrated} under ${MIN_ENDPOINT_YEARS} endpoint-years, so not yet rateable`;
    provSort.paintSort();
  }
  provQ.addEventListener("input", render);
  render();
}
