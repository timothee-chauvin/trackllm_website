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
const MUTED_TEXT = "#8A97A8";
const MUTED_GRID = "rgba(140,150,165,0.25)";

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
  Plotly.newPlot(
    el,
    traces,
    {
      barmode: "stack",
      title: { text: "Daily spend by category", font: { color: MUTED_TEXT, size: 14 } },
      xaxis: { title: { text: "Date" }, gridcolor: MUTED_GRID, color: MUTED_TEXT },
      yaxis: { title: { text: "USD" }, gridcolor: MUTED_GRID, color: MUTED_TEXT, rangemode: "tozero" },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: MUTED_TEXT },
      legend: { bgcolor: "rgba(0,0,0,0)", font: { color: MUTED_TEXT } },
      height: 400,
      margin: { t: 40, r: 20, b: 50, l: 60 },
    },
    { responsive: true, displayModeBar: false }
  );
}

init();
