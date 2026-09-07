// The Providers page: the drift-rate dot plot over every rateable provider
// company, then the full sortable table. `init` is exported for the smoke tests.
import { MIN_ENDPOINT_YEARS } from "./components";
import { fmtInt, loadOverview } from "./overview_data";
import { initProviderTable } from "./provider_table";
import { rateablePlotRows, ratePlot } from "./rate_plot";

export async function init(): Promise<void> {
  const DATA = await loadOverview("lede");
  const S = DATA.stats;
  const provs = DATA.providers;
  const rateable = rateablePlotRows(provs);

  document.getElementById("lede")!.innerHTML =
    `<b>${fmtInt(S.provider_companies)} provider companies</b> serving ${fmtInt(S.providers)} variants` +
    ` and ${fmtInt(S.endpoints)} endpoints we have tracked. ${rateable.length} of them have at least` +
    ` ${MIN_ENDPOINT_YEARS} endpoint-years of logprob monitoring, enough for a drift rate.`;

  document.getElementById("provPlot")!.innerHTML = ratePlot(rateable, "");
  initProviderTable(provs);
}

init();
