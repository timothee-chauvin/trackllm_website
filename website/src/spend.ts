import Plotly from "plotly.js-dist-min";

interface SpendData {
  group_order: string[];
  cumulative: Record<string, number>;
  last_30d: Record<string, number>;
  daily: { date: string; groups: Record<string, number> }[];
  by_endpoint: { slug: string; groups: Record<string, number>; total: number }[];
}

// Group names and order come from spend.json (single source: generate_site/spend.py);
// only the presentation colors live here.
const GROUP_COLOR: Record<string, string> = {
  onboarding: "#8250df",
  monitoring: "#0969da",
  lt: "#1a7f37",
  vetting: "#9a6700",
};
const DEFAULT_GROUP_COLOR = "#57606a";

async function init(): Promise<void> {
  const el = document.getElementById("spend-chart");
  if (!el) return;
  const res = await fetch("data/spend.json");
  if (!res.ok) return;
  const data: SpendData = await res.json();
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
      title: { text: "Daily spend by category", font: { color: "#1f2328", size: 14 } },
      xaxis: { title: { text: "Date" }, gridcolor: "#d0d7de" },
      yaxis: { title: { text: "USD" }, gridcolor: "#d0d7de", rangemode: "tozero" },
      paper_bgcolor: "#f6f8fa",
      plot_bgcolor: "#ffffff",
      font: { color: "#1f2328" },
      height: 400,
      margin: { t: 40, r: 20, b: 50, l: 60 },
    },
    { responsive: true, displayModeBar: false }
  );
}

init();
