// The endpoint directory shared by the Overview and the provider pages: same row
// shape (overview.py builds the dicts, provider.py reuses them), same sort rules,
// same trailing cells. The pages differ only in how rows are filtered and how the
// two leading cells (model, provider) are drawn.
import {
  B3IT_CAP,
  LT_CAP,
  bindActivation,
  esc,
  methodBadges,
  sparkline,
  statusPill,
  statusRank,
  untrackedDirCells,
} from "./components";

export interface EndpointRow {
  slug: string;
  model: string;
  modelSlug: string;
  org: string;
  provider: string;
  providerSlug: string;
  methods: string[];
  status: "stable" | "changed" | "retired" | null; // null on rows with no series (untracked)
  stableDays: number | null;
  nChanges: number;
  trace: number[];
  headline: string;
  reason: string;
}

export type DirSortKey = "model" | "provider" | "status" | "nChanges" | "stableDays";

/** Sortable column headers for one table: holds (key, dir), flips them on header
 *  clicks or Enter/Space, and paints the ▼/▲ arrows plus the aria-sort a screen
 *  reader announces. `attr` is the th dataset key ("sort", "psort"), so two
 *  tables on one page keep independent state. A first click on a key listed in
 *  `descending` sorts high-to-low. The headers become focusable here rather than
 *  in the template: without this script they sort nothing, so they are no more
 *  tabbable than the plain headers beside them. */
export function initSortHeaders<K extends string>(
  attr: string,
  initial: K,
  initialDir: number,
  descending: readonly K[],
  render: () => void
): { key: K; dir: number; paintSort: () => void } {
  const headers = document.querySelectorAll<HTMLElement>(`th[data-${attr}]`);
  const state = {
    key: initial,
    dir: initialDir,
    paintSort(): void {
      headers.forEach((th) => {
        const active = th.dataset[attr] === state.key;
        const arr = th.querySelector(".arr");
        if (arr) arr.textContent = active ? (state.dir < 0 ? "▼" : "▲") : "";
        th.setAttribute(
          "aria-sort",
          active ? (state.dir < 0 ? "descending" : "ascending") : "none"
        );
      });
    },
  };
  headers.forEach((th) => {
    th.tabIndex = 0;
    bindActivation(th, `th[data-${attr}]`, () => {
      const k = th.dataset[attr] as K;
      if (state.key === k) state.dir *= -1;
      else { state.key = k; state.dir = descending.includes(k) ? -1 : 1; }
      render();
    });
  });
  return state;
}

/** In-place directory sort; `providerValue` supplies the provider column's
 *  comparable (full name on the Overview, bare variant on a provider page).
 *  Model name breaks every tie so equal rows keep a stable order. */
export function sortEndpointRows(
  list: EndpointRow[],
  key: DirSortKey,
  dir: number,
  providerValue: (r: EndpointRow) => string
): void {
  list.sort((a, b) => {
    let av: string | number, bv: string | number;
    if (key === "status") { av = statusRank(a); bv = statusRank(b); }
    else if (key === "stableDays") { av = a.stableDays ?? -1; bv = b.stableDays ?? -1; }
    else if (key === "nChanges") { av = a.nChanges; bv = b.nChanges; }
    else if (key === "provider") { av = providerValue(a); bv = providerValue(b); }
    else { av = a.model.toLowerCase(); bv = b.model.toLowerCase(); }
    if (av < bv) return -dir;
    if (av > bv) return dir;
    return a.model.localeCompare(b.model);
  });
}

function stableCell(r: EndpointRow): string {
  if (r.status === "retired" || r.stableDays === null) return `<span class="org-cell">—</span>`;
  const d = r.stableDays;
  return `<span class="cc">${d >= 365 ? (d / 365).toFixed(1) + "y" : d + "d"}</span>`;
}

/** The five directory cells after model/provider for a row with a series --
 *  the untracked counterpart is components.ts::untrackedDirCells. */
function trackedDirCells(r: EndpointRow, root: string): string {
  const isLT = r.methods.includes("lt");
  return `<td><a href="${root}endpoints/${esc(r.slug)}.html">${statusPill(r.status!)}</a></td>
    <td class="r"><span class="cc ${r.nChanges ? "some" : "zero"}">${r.nChanges}</span></td>
    <td class="col-hide"><span class="methods">${methodBadges(r.methods)}</span></td>
    <td class="r col-hide">${stableCell(r)}</td>
    <td class="col-hide spark-cell">${sparkline(r.trace, isLT ? LT_CAP : B3IT_CAP, isLT ? "var(--accent)" : "var(--b3it)", null)}</td>`;
}

export interface DirectoryConfig {
  rows: EndpointRow[];
  root: string; // link prefix back to the site root ("" or "../")
  q: HTMLInputElement;
  body: HTMLElement;
  foot: HTMLElement;
  /** Sort keys whose first header click sorts high-to-low. */
  descending: readonly DirSortKey[];
  providerValue: (r: EndpointRow) => string;
  /** Page-specific search + chip filtering, given the trimmed (uncased) query. */
  list: (q: string) => EndpointRow[];
  /** The leading model and provider cells. */
  leadCells: (r: EndpointRow, q: string) => string;
}

/** Wire up a directory table (search input, sortable headers, first paint) and
 *  return its render function for the page's own chip handlers to call. */
export function initDirectory(cfg: DirectoryConfig): () => void {
  const sort = initSortHeaders<DirSortKey>("sort", "nChanges", -1, cfg.descending, render);
  function render(): void {
    const q = cfg.q.value.trim();
    const list = cfg.list(q);
    sortEndpointRows(list, sort.key, sort.dir, cfg.providerValue);
    cfg.body.innerHTML = list.map((r) => {
      const head = `<tr>${cfg.leadCells(r, q)}`;
      const cells = r.methods.length ? trackedDirCells(r, cfg.root) : untrackedDirCells(r, cfg.root);
      return `${head}${cells}</tr>`;
    }).join("") || '<tr><td colspan="7"><div class="empty">No endpoints match.</div></td></tr>';
    cfg.foot.textContent = `${list.length} of ${cfg.rows.length} endpoints`;
    sort.paintSort();
  }
  cfg.q.addEventListener("input", render);
  render();
  return render;
}
