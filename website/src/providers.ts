// The Providers page: the drift-rate dot plot over every rateable provider
// company, in the method the reader picks (LT, B3IT, or both stacked), then the
// full sortable table. The mode lives in the URL hash so a link can open the page
// on it. `init` is exported for the smoke tests.
import { MIN_ENDPOINT_YEARS, bindActivation, pickChip } from "./components";
import { fmtInt, loadOverview } from "./overview_data";
import { initProviderTable } from "./provider_table";
import { RATE_MODES, RateMode, rateLegend, rateNote, rateStatLine, rateablePlotRows, ratePlot } from "./rate_plot";

const DEFAULT_MODE: RateMode = "lt"; // the front page's plot, the chip the template starts on, and the one the hash leaves unnamed

/** `#b3it` / `#both` name a mode; anything else (no hash included) is the default. */
export function modeFromHash(hash: string): RateMode {
  const m = hash.replace(/^#/, "") as RateMode;
  return RATE_MODES.includes(m) ? m : DEFAULT_MODE;
}

export async function init(): Promise<void> {
  const DATA = await loadOverview("lede");
  const S = DATA.stats;
  const provs = DATA.providers;
  const rateable = (mode: RateMode): number => rateablePlotRows(provs, mode).length;

  document.getElementById("lede")!.innerHTML =
    `<b>${fmtInt(S.provider_companies)} provider companies</b> serving ${fmtInt(S.providers)} variants` +
    ` and ${fmtInt(S.endpoints)} endpoints we have tracked. ${rateable("lt")} of them have at least` +
    ` ${MIN_ENDPOINT_YEARS} endpoint-years of logprob (LT) monitoring and ${rateable("b3it")} of B3IT monitoring, enough for a drift rate.`;

  const chips = document.getElementById("rateMode")!;
  const plot = document.getElementById("provPlot")!;
  const picked = new Set<string>([DEFAULT_MODE]);
  function show(mode: RateMode): void {
    pickChip(chips.querySelector<HTMLElement>(`.chip[data-m="${mode}"]`)!, "m", picked, true);
    plot.classList.toggle("both", mode === "both");
    plot.innerHTML = ratePlot(rateablePlotRows(provs, mode), "", mode);
    document.getElementById("provStat")!.innerHTML = rateStatLine(provs, mode);
    document.getElementById("provLegend")!.innerHTML = rateLegend(mode);
    document.getElementById("provNote")!.innerHTML = rateNote(provs, S, mode);
    // replace, not push: flipping the toggle is not a page the Back button should revisit
    history.replaceState(null, "", mode === DEFAULT_MODE ? location.pathname + location.search : `#${mode}`);
  }
  bindActivation(chips, ".chip", chip => show(chip.dataset.m as RateMode));
  window.addEventListener("hashchange", () => show(modeFromHash(location.hash)));
  show(modeFromHash(location.hash));

  initProviderTable(provs);
}

init();
