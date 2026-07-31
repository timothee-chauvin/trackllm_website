import Plotly from "plotly.js-dist-min";

import { showLoadError } from "./components";

interface SpendData {
  group_order: string[];
  cumulative: Record<string, number>;
  last_30d: Record<string, number>;
  daily: { date: string; groups: Record<string, number> }[];
  by_endpoint: { slug: string; groups: Record<string, number>; total: number }[];
}

// Group names and order come from spend.json (single source: generate_site/spend.py);
// only the presentation colors live here. Colors reuse the design system's token
// palette (dark-theme hex, which stays legible against both light and dark page
// backgrounds) rather than a single flat color, since a stacked chart needs one
// hue per group to remain readable.
const GROUP_COLOR: Record<string, string> = {
  onboarding: "#B98BE8", // --b3it
  monitoring: "#37C2E0", // --accent
  lt: "#3FB77E", // --stable
  vetting: "#E0A94A", // --changed
};
const DEFAULT_GROUP_COLOR = "#8A97A8";
const MUTED_GRID = "rgba(140,150,165,0.25)";
// --text-dim, dark theme; only reached if the stylesheet has not applied yet.
const MUTED_TEXT_FALLBACK = "#7F8EA0";

/** The chart's text cannot be a fixed hex: no single color clears AA against both
 *  the dark and the light page background, so it follows the --text-dim token and
 *  is repainted when the theme toggle (base.html.j2) flips data-theme. */
function mutedText(): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue("--text-dim");
  return v.trim() || MUTED_TEXT_FALLBACK;
}

/** Every layout field carrying that color, as relayout attribute paths. */
function textColors(c: string): Record<string, string> {
  return {
    "font.color": c,
    "title.font.color": c,
    "xaxis.color": c,
    "yaxis.color": c,
    "legend.font.color": c,
  };
}

async function init(): Promise<void> {
  const el = document.getElementById("spend-chart");
  if (!el) return;
  let data: SpendData;
  try {
    const res = await fetch("data/spend.json");
    if (!res.ok) throw new Error(`spend.json: HTTP ${res.status}`);
    data = await res.json();
  } catch (err) {
    showLoadError("spend-chart", "spend data");
    throw err;
  }
  if (!data.daily?.length) return;
  const dates = data.daily.map((d) => new Date(d.date));
  const traces = (data.group_order ?? [])
    .filter((g) => data.daily.some((d) => d.groups[g]))
    .map((g) => ({
      x: dates,
      y: data.daily.map((d) => d.groups[g] ?? 0),
      type: "bar" as const,
      name: g,
      marker: { color: GROUP_COLOR[g] ?? DEFAULT_GROUP_COLOR },
    }));
  await Plotly.newPlot(
    el,
    traces,
    {
      barmode: "stack",
      title: { text: "Daily spend by category", font: { size: 14 } },
      xaxis: { title: { text: "Date" }, gridcolor: MUTED_GRID },
      yaxis: { title: { text: "USD" }, gridcolor: MUTED_GRID, rangemode: "tozero" },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      legend: { bgcolor: "rgba(0,0,0,0)" },
      height: 400,
      margin: { t: 40, r: 20, b: 50, l: 60 },
    },
    { responsive: true, displayModeBar: false }
  );
  const paint = (): void => {
    Plotly.relayout(el, textColors(mutedText()) as Partial<Plotly.Layout>);
  };
  paint();
  new MutationObserver(paint).observe(document.documentElement, {
    attributeFilter: ["data-theme"],
  });
}

init();
