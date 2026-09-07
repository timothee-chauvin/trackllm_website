// The All-endpoints page: the full directory with search, status / method /
// change-history chips and sortable columns. The `init` export lets the smoke
// tests re-render a fresh document without busting the module cache.
import { bindActivation, bindTips, toggleChip } from "./components";
import { initDirectory, overviewLeadCells } from "./directory";
import { fmtInt, loadOverview } from "./overview_data";

export async function init(): Promise<void> {
  const DATA = await loadOverview("lede");
  const S = DATA.stats;
  const rows = DATA.endpoints;

  document.getElementById("lede")!.innerHTML =
    `<b>${fmtInt(rows.length)} endpoints</b> in the catalog, ${fmtInt(S.active)} of them under active tracking` +
    ` — <b style="color:var(--changed)">${fmtInt(S.changes_total)} changes</b> detected across ${S.changed_endpoints} of them.` +
    ` The status column says whether and why each is tracked.`;

  // provider pages only exist for providers with tracked endpoints; a row whose
  // provider has none must name it without linking it
  const providerPages = new Set(DATA.providers.map(p => p.slug));

  // Three chip groups, each OR within itself and AND with the others and with the
  // search: a row shows when it passes every group that has any chip on.
  const statusFilters = new Set<string>(["tracked"]);
  const methodFilters = new Set<string>();
  const changeFilters = new Set<string>();
  const render = initDirectory({
    rows,
    root: "",
    q: document.getElementById("q") as HTMLInputElement,
    body: document.getElementById("dirBody")!,
    foot: document.getElementById("dirFoot")!,
    descending: ["stableDays", "nChanges"],
    providerValue: r => r.provider.toLowerCase(),
    list: q => {
      const ql = q.toLowerCase();
      return rows.filter(r => {
        if (ql && !`${r.model} ${r.provider} ${r.org}`.toLowerCase().includes(ql)) return false;
        if (statusFilters.size && !statusFilters.has(r.headline)) return false;
        if (methodFilters.size && !r.methods.some(m => methodFilters.has(m))) return false;
        if (changeFilters.size) {
          const ever = changeFilters.has("everchanged") && r.nChanges > 0;
          const recent = changeFilters.has("recent") && r.status === "changed";
          if (!ever && !recent) return false;
        }
        return true;
      });
    },
    leadCells: overviewLeadCells(providerPages),
  });
  bindActivation(document.getElementById("chips")!, ".chip", chip => {
    const d = chip.dataset;
    if (d.st) toggleChip(chip, statusFilters, d.st);
    else if (d.f) toggleChip(chip, methodFilters, d.f);
    else toggleChip(chip, changeFilters, d.c!);
    render();
  });
  // one binding for the status chips above and every directory badge/pill render() draws
  bindTips(document.body);
}

init();
