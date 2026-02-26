import Plotly from "plotly.js-dist-min";

interface PromptManifest {
  slug: string;
  prompt: string;
  months: string[];
}

interface EndpointManifest {
  model: string;
  provider: string;
  slug: string;
  prompts: PromptManifest[];
}

interface LogprobQuery {
  date: Date;
  tokens: string[];
  logprobs: number[];
}

interface LogprobsData {
  seen_tokens: string[];
  seen_logprobs: { tokens: number[]; logprobs: number[] }[];
}

interface LTScoresData {
  n_per_test: number;
  dates: string[];
  scores: number[];
  sigmas: (number | null)[];
  changes: { index: number; sigma: number }[];
}

type TimeRange = "5d" | "1m" | "3m" | "all";

const TIME_RANGES: { value: TimeRange; label: string }[] = [
  { value: "5d", label: "5 Days" },
  { value: "1m", label: "1 Month" },
  { value: "3m", label: "3 Months" },
  { value: "all", label: "All" },
];

let currentRange: TimeRange = "3m";
let showTokens: boolean = false;
let changeDates: Date[] = [];
const cachedData = new Map<string, LogprobQuery[]>();
const chartElements: {
  plotDiv: HTMLElement;
  promptName: string;
  slug: string;
}[] = [];
const radioGroups: HTMLElement[] = [];
const tokenToggleInputs: HTMLInputElement[] = [];

async function fetchLogprobsForMonth(
  endpointSlug: string,
  promptSlug: string,
  month: string
): Promise<LogprobQuery[]> {
  const baseUrl = `../data/${endpointSlug}/${promptSlug}/${month}`;
  try {
    const [queriesRes, logprobsRes] = await Promise.all([
      fetch(`${baseUrl}/queries.json`),
      fetch(`${baseUrl}/logprobs.json`),
    ]);
    if (!queriesRes.ok || !logprobsRes.ok) return [];

    const queries: [string, number | string][] = await queriesRes.json();
    const logprobsData: LogprobsData = await logprobsRes.json();

    const [year, monthNum] = month.split("-").map(Number);
    const entries: LogprobQuery[] = [];

    for (const [dateStr, idx] of queries) {
      if (typeof idx === "string") continue;

      const [day, time] = dateStr.split(" ");
      const [hour, minute, second] = time.split(":").map(Number);
      const date = new Date(
        year,
        monthNum - 1,
        parseInt(day),
        hour,
        minute,
        second
      );

      const logprobVec = logprobsData.seen_logprobs[idx];
      if (!logprobVec || logprobVec.logprobs.length === 0) continue;

      const tokens = logprobVec.tokens.map((i) => logprobsData.seen_tokens[i]);
      entries.push({ date, tokens, logprobs: logprobVec.logprobs });
    }

    return entries;
  } catch {
    return [];
  }
}

async function fetchAllLogprobsForPrompt(
  manifest: EndpointManifest,
  prompt: PromptManifest
): Promise<LogprobQuery[]> {
  const allEntries = (
    await Promise.all(
      prompt.months.map((month) =>
        fetchLogprobsForMonth(manifest.slug, prompt.slug, month)
      )
    )
  ).flat();
  allEntries.sort((a, b) => a.date.getTime() - b.date.getTime());
  return allEntries;
}

function filterByTimeRange(
  entries: LogprobQuery[],
  range: TimeRange
): LogprobQuery[] {
  if (range === "all") return entries;
  const now = new Date();
  let cutoff: Date;
  switch (range) {
    case "5d":
      cutoff = new Date(now.getTime() - 5 * 24 * 60 * 60 * 1000);
      break;
    case "1m":
      cutoff = new Date(now.getFullYear(), now.getMonth() - 1, now.getDate());
      break;
    case "3m":
      cutoff = new Date(now.getFullYear(), now.getMonth() - 3, now.getDate());
      break;
  }
  return entries.filter((e) => e.date >= cutoff);
}

function reprToken(token: string): string {
  const escaped = JSON.stringify(token).slice(1, -1);
  if (escaped.length > 15) return `'${escaped.slice(0, 12)}…'`;
  return `'${escaped}'`;
}

const SCORE_COLOR = "#0969da";
const SIGMA_COLOR = "#cf222e";

function formatChangeSummary(data: LTScoresData): string {
  const dates = data.dates.map((d) => new Date(d));
  const first = dates[0];
  const last = dates[dates.length - 1];
  const fmt = (d: Date) => d.toISOString().slice(0, 10);

  if (data.changes.length === 0) {
    return `No detected changes between ${fmt(first)} and ${fmt(last)}.`;
  }

  const header = `Detected changes between ${fmt(first)} and ${fmt(last)}:`;
  const items = data.changes.map((cp) => {
    const d = dates[cp.index];
    return `  • ${fmt(d)} (score = ${data.scores[cp.index].toFixed(3)}, deviation = ${cp.sigma.toFixed(1)}σ)`;
  });
  return header + "\n" + items.join("\n");
}

function renderAnomalyChart(
  container: HTMLElement,
  data: LTScoresData
): void {
  const dates = data.dates.map((d) => new Date(d));

  // Left y-axis: test statistic
  const traces: Plotly.Data[] = [
    {
      x: dates,
      y: data.scores,
      type: "scatter",
      mode: "lines",
      name: "Test statistic",
      yaxis: "y",
      line: { width: 1.5, color: SCORE_COLOR },
      hovertemplate: "%{x}<br>Score: %{y:.4f}<extra></extra>",
    },
    {
      x: [dates[0], dates[dates.length - 1]],
      y: [1.0, 1.0],
      type: "scatter",
      mode: "lines",
      name: "Score threshold (1.0)",
      yaxis: "y",
      line: { width: 1, color: SCORE_COLOR, dash: "dot" },
      hoverinfo: "skip",
    } as Plotly.Data,
  ];

  // Right y-axis: deviation in sigmas
  const sigmaDates: Date[] = [];
  const sigmaVals: number[] = [];
  for (let i = 0; i < data.sigmas.length; i++) {
    if (data.sigmas[i] !== null) {
      sigmaDates.push(dates[i]);
      sigmaVals.push(data.sigmas[i] as number);
    }
  }
  if (sigmaDates.length > 0) {
    traces.push(
      {
        x: sigmaDates,
        y: sigmaVals,
        type: "scatter",
        mode: "lines",
        name: "Deviation (σ)",
        yaxis: "y2",
        line: { width: 1.5, color: SIGMA_COLOR },
        hovertemplate: "%{x}<br>Deviation: %{y:.1f}σ<extra></extra>",
      },
      {
        x: [sigmaDates[0], sigmaDates[sigmaDates.length - 1]],
        y: [12, 12],
        type: "scatter",
        mode: "lines",
        name: "σ threshold (12)",
        yaxis: "y2",
        line: { width: 1, color: SIGMA_COLOR, dash: "dot" },
        hoverinfo: "skip",
      } as Plotly.Data
    );
  }

  const shapes = makeChangeShapes(data.changes.map((cp) => dates[cp.index]));

  const annotations: Partial<Plotly.Annotations>[] = data.changes.map(
    (cp) => ({
      x: dates[cp.index],
      y: 1,
      yref: "paper" as const,
      text: `${cp.sigma.toFixed(0)}σ`,
      showarrow: false,
      font: { color: "#888888", size: 10 },
      yanchor: "bottom" as const,
    })
  );

  const layout: Partial<Plotly.Layout> = {
    title: {
      text: `Anomaly Score (${data.scores.length} points, window=${data.n_per_test})`,
      font: { color: "#1f2328", size: 14 },
    },
    xaxis: {
      title: { text: "Date", font: { color: "#57606a" } },
      gridcolor: "#d0d7de",
      tickfont: { color: "#57606a" },
    },
    yaxis: {
      title: { text: "Test Statistic", font: { color: SCORE_COLOR } },
      gridcolor: "#d0d7de",
      tickfont: { color: SCORE_COLOR },
      rangemode: "tozero",
    },
    yaxis2: {
      title: { text: "Deviation (σ)", font: { color: SIGMA_COLOR } },
      tickfont: { color: SIGMA_COLOR },
      overlaying: "y",
      side: "right",
      rangemode: "tozero",
      showgrid: false,
    },
    paper_bgcolor: "#f6f8fa",
    plot_bgcolor: "#ffffff",
    font: { color: "#1f2328" },
    legend: {
      font: { color: "#1f2328", size: 10 },
      bgcolor: "transparent",
      orientation: "h",
      y: -0.2,
    },
    height: 400,
    margin: { t: 40, r: 60, b: 80, l: 60 },
    shapes,
    annotations,
  };

  container.innerHTML = "";
  Plotly.newPlot(container, traces, layout, {
    responsive: true,
    displayModeBar: false,
  });
}

function makeChangeShapes(
  changeDts: Date[],
): Partial<Plotly.Shape>[] {
  return changeDts.map((d) => ({
    type: "line" as const,
    x0: d,
    x1: d,
    y0: 0,
    y1: 1,
    yref: "paper" as const,
    line: { color: "#888888", width: 1.5, dash: "dot" as const },
  }));
}

function renderPromptChart(
  container: HTMLElement,
  promptName: string,
  entries: LogprobQuery[]
): void {
  if (entries.length === 0) {
    container.innerHTML = '<div class="no-data">No logprob data available</div>';
    return;
  }

  const sorted = [...entries].sort(
    (a, b) => a.date.getTime() - b.date.getTime()
  );

  const maxLogprobs = Math.max(...sorted.map((e) => e.logprobs.length));

  const tokenAtPosition: string[] = [];
  for (let i = 0; i < maxLogprobs; i++) {
    const tokenCounts = new Map<string, number>();
    for (const entry of sorted) {
      if (i < entry.tokens.length) {
        const token = entry.tokens[i];
        tokenCounts.set(token, (tokenCounts.get(token) || 0) + 1);
      }
    }
    let mostCommon = `pos ${i + 1}`;
    let maxCount = 0;
    for (const [token, count] of tokenCounts) {
      if (count > maxCount) {
        maxCount = count;
        mostCommon = token;
      }
    }
    tokenAtPosition.push(mostCommon);
  }

  const traces = [];
  for (let i = 0; i < maxLogprobs; i++) {
    const x: Date[] = [];
    const y: number[] = [];
    const hoverTokens: string[] = [];

    for (const entry of sorted) {
      if (i < entry.logprobs.length) {
        x.push(entry.date);
        y.push(entry.logprobs[i]);
        hoverTokens.push(reprToken(entry.tokens[i] || "?"));
      }
    }

    if (x.length > 0) {
      const label = reprToken(tokenAtPosition[i]);
      traces.push({
        x,
        y,
        type: "scatter",
        mode: "lines",
        name: label,
        line: { width: 1 },
        text: hoverTokens,
        hovertemplate: `<b>%{text}</b><br>%{x}<br>Logprob: %{y:.4f}<extra></extra>`,
      });
    }
  }

  const layout = {
    title: {
      text: `Logprobs for "${promptName}" (${sorted.length} Queries)`,
      font: { color: "#1f2328", size: 14 },
    },
    xaxis: {
      title: { text: "Date", font: { color: "#57606a" } },
      gridcolor: "#d0d7de",
      tickfont: { color: "#57606a" },
    },
    yaxis: {
      title: { text: "Log Probability", font: { color: "#57606a" } },
      gridcolor: "#d0d7de",
      tickfont: { color: "#57606a" },
    },
    paper_bgcolor: "#f6f8fa",
    plot_bgcolor: "#ffffff",
    font: { color: "#1f2328" },
    legend: {
      font: { color: "#1f2328", size: 10 },
      bgcolor: "transparent",
      orientation: "h" as const,
      y: -0.2,
    },
    height: 560,
    margin: { t: 40, r: 20, b: 80, l: 60 },
    showlegend: showTokens && maxLogprobs <= 25,
    shapes: makeChangeShapes(changeDates.filter(
      (d) => d >= sorted[0].date && d <= sorted[sorted.length - 1].date
    )),
  };

  const config = {
    responsive: true,
    displayModeBar: false,
  };

  container.innerHTML = "";
  Plotly.newPlot(container, traces, layout, config);
}

function createRangeSelector(): HTMLElement {
  const container = document.createElement("div");
  container.className = "range-selector";

  for (const { value, label } of TIME_RANGES) {
    const optLabel = document.createElement("label");
    optLabel.className =
      "range-option" + (value === currentRange ? " active" : "");

    const input = document.createElement("input");
    input.type = "radio";
    input.name = `range-${radioGroups.length}`;
    input.value = value;
    input.checked = value === currentRange;
    input.addEventListener("change", () => {
      currentRange = value;
      syncAllRadios();
      rerenderAllCharts();
    });

    optLabel.appendChild(input);
    optLabel.appendChild(document.createTextNode(label));
    container.appendChild(optLabel);
  }

  const tokenLabel = document.createElement("label");
  tokenLabel.className = "range-option token-toggle" + (showTokens ? " active" : "");

  const tokenInput = document.createElement("input");
  tokenInput.type = "checkbox";
  tokenInput.checked = showTokens;
  tokenInput.style.display = "none";
  tokenInput.addEventListener("change", () => {
    showTokens = tokenInput.checked;
    syncAllTokenToggles();
    rerenderAllCharts();
  });

  tokenLabel.appendChild(tokenInput);
  tokenLabel.appendChild(document.createTextNode("show tokens"));
  container.appendChild(tokenLabel);
  tokenToggleInputs.push(tokenInput);

  radioGroups.push(container);
  return container;
}

function syncAllRadios(): void {
  for (const group of radioGroups) {
    const labels = group.querySelectorAll<HTMLLabelElement>(".range-option");
    for (const label of labels) {
      const input = label.querySelector<HTMLInputElement>("input[type=radio]");
      if (input) {
        input.checked = input.value === currentRange;
        label.classList.toggle("active", input.value === currentRange);
      }
    }
  }
}

function syncAllTokenToggles(): void {
  for (const input of tokenToggleInputs) {
    input.checked = showTokens;
    const label = input.closest<HTMLLabelElement>("label");
    if (label) label.classList.toggle("active", showTokens);
  }
}

function rerenderAllCharts(): void {
  for (const { plotDiv, promptName, slug } of chartElements) {
    const entries = cachedData.get(slug);
    if (entries) {
      renderPromptChart(plotDiv, promptName, filterByTimeRange(entries, currentRange));
    }
  }
}

async function renderCharts(manifest: EndpointManifest): Promise<void> {
  const chartsContainer = document.getElementById("charts-container");
  if (!chartsContainer) return;

  chartsContainer.innerHTML = "";

  // Anomaly score plot
  changeDates = [];
  try {
    const res = await fetch(`../data/${manifest.slug}/lt_scores.json`);
    if (res.ok) {
      const ltData: LTScoresData = await res.json();
      if (ltData.scores.length > 0) {
        const dates = ltData.dates.map((d) => new Date(d));
        changeDates = ltData.changes.map((cp) => dates[cp.index]);

        const summaryEl = document.createElement("pre");
        summaryEl.className = "change-summary";
        summaryEl.textContent = formatChangeSummary(ltData);
        chartsContainer.appendChild(summaryEl);

        const anomalyDiv = document.createElement("div");
        anomalyDiv.className = "chart";
        const plotDiv = document.createElement("div");
        anomalyDiv.appendChild(plotDiv);
        chartsContainer.appendChild(anomalyDiv);
        renderAnomalyChart(plotDiv, ltData);
      }
    }
  } catch {
    // No lt_scores available
  }

  // Detailed per-prompt charts inside a dropdown
  const details = document.createElement("details");
  details.className = "detailed-plots";
  const summary = document.createElement("summary");
  summary.textContent = "Detailed plots";
  details.appendChild(summary);
  chartsContainer.appendChild(details);

  for (const prompt of manifest.prompts) {
    const chartDiv = document.createElement("div");
    chartDiv.className = "chart";

    chartDiv.appendChild(createRangeSelector());

    const plotDiv = document.createElement("div");
    plotDiv.innerHTML = '<div class="loading">Loading chart...</div>';
    chartDiv.appendChild(plotDiv);

    details.appendChild(chartDiv);
    chartElements.push({
      plotDiv,
      promptName: prompt.prompt,
      slug: prompt.slug,
    });

    fetchAllLogprobsForPrompt(manifest, prompt).then((entries) => {
      cachedData.set(prompt.slug, entries);
      renderPromptChart(
        plotDiv,
        prompt.prompt,
        filterByTimeRange(entries, currentRange)
      );
    });
  }
}

function init(): void {
  const manifestEl = document.getElementById("manifest");
  if (!manifestEl) {
    console.error("Manifest element not found");
    return;
  }

  const manifest: EndpointManifest = JSON.parse(manifestEl.textContent || "{}");
  renderCharts(manifest);
}

init();
