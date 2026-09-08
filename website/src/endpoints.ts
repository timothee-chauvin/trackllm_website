// The All-endpoints page: the full directory with search, status / method /
// change-history chips and sortable columns. The `init` export lets the smoke
// tests re-render a fresh document without busting the module cache.
import { bindActivation, bindTips, toggleChip } from "./components";
import { initDirectory, overviewLeadCells } from "./directory";
import { fmtInt, loadOverview } from "./overview_data";

/** One chip at most per row: picking one switches its siblings off. With
 *  `required` the picked chip cannot be switched off again (a radio group). */
function pickChip(chip: HTMLElement, attr: "st" | "c", set: Set<string>, required: boolean): void {
  const value = chip.dataset[attr]!;
  if (required && set.has(value)) return;
  for (const other of chip.parentElement!.querySelectorAll<HTMLElement>(".chip.on")) {
    if (other !== chip) toggleChip(other, set, other.dataset[attr]!);
  }
  toggleChip(chip, set, value);
}

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

  // Three chip rows, AND-ed with each other and with the search. Status is a
  // radio: exactly one headline shows. Method chips conjoin: both on means
  // endpoints tracked by both. The change row is one filter at a time, or none.
  const status = new Set<string>(["tracked"]);
  const methods = new Set<string>();
  const change = new Set<string>();
  const render = initDirectory({
    rows,
    root: "",
    q: document.getElementById("q") as HTMLInputElement,
    body: document.getElementById("dirBody")!,
    foot: document.getElementById("dirFoot")!,
    descending: ["lastChange", "nChanges"],
    providerValue: r => r.provider.toLowerCase(),
    list: q => {
      const ql = q.toLowerCase();
      return rows.filter(r => {
        if (ql && !`${r.model} ${r.provider} ${r.org}`.toLowerCase().includes(ql)) return false;
        if (!status.has(r.headline)) return false;
        for (const m of methods) if (!r.methods.includes(m)) return false;
        if (change.has("everchanged") && r.nChanges === 0) return false;
        if (change.has("recent") && !r.recent) return false;
        return true;
      });
    },
    leadCells: overviewLeadCells(providerPages),
  });
  bindActivation(document.getElementById("chips")!, ".chip", chip => {
    const d = chip.dataset;
    if (d.st) pickChip(chip, "st", status, true);
    else if (d.f) toggleChip(chip, methods, d.f);
    else pickChip(chip, "c", change, false);
    render();
  });
  // one binding for the status chips above and every directory badge/pill render() draws
  bindTips(document.body);
}

init();
