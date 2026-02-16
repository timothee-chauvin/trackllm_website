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

interface QueryResult {
  total: number;
  errors: number;
}

interface PromptData {
  prompt: string;
  total: number;
  errors: number;
  success: number;
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

type TimeRange = "5d" | "1m" | "3m" | "all";

const TIME_RANGES: { value: TimeRange; label: string }[] = [
  { value: "5d", label: "5 Days" },
  { value: "1m", label: "1 Month" },
  { value: "3m", label: "3 Months" },
  { value: "all", label: "All" },
];

let currentRange: TimeRange = "3m";
let showTokens: boolean = false;
const cachedData = new Map<string, LogprobQuery[]>();
const chartElements: {
  plotDiv: HTMLElement;
  promptName: string;
  slug: string;
}[] = [];
const radioGroups: HTMLElement[] = [];
const tokenToggleInputs: HTMLInputElement[] = [];

async function fetchQueries(
  endpointSlug: string,
  promptSlug: string,
  month: string
): Promise<QueryResult> {
  const url = `../data/${endpointSlug}/${promptSlug}/${month}/queries.json`;
  try {
    const res = await fetch(url);
    if (!res.ok) return { total: 0, errors: 0 };
    const queries: [string, number | string][] = await res.json();
    const total = queries.length;
    const errors = queries.filter(
      (q) => typeof q[1] === "string" && q[1].startsWith("e")
    ).length;
    return { total, errors };
  } catch {
    return { total: 0, errors: 0 };
  }
}

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

  for (const prompt of manifest.prompts) {
    const chartDiv = document.createElement("div");
    chartDiv.className = "chart";

    chartDiv.appendChild(createRangeSelector());

    const plotDiv = document.createElement("div");
    plotDiv.innerHTML = '<div class="loading">Loading chart...</div>';
    chartDiv.appendChild(plotDiv);

    chartsContainer.appendChild(chartDiv);
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

async function loadPromptData(
  manifest: EndpointManifest,
  prompt: PromptManifest
): Promise<PromptData> {
  const results = await Promise.all(
    prompt.months.map((month) => fetchQueries(manifest.slug, prompt.slug, month))
  );
  const total = results.reduce((sum, r) => sum + r.total, 0);
  const errors = results.reduce((sum, r) => sum + r.errors, 0);
  return { prompt: prompt.prompt, total, errors, success: total - errors };
}

function formatNum(n: number): string {
  return n.toLocaleString();
}

function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function init(): Promise<void> {
  const manifestEl = document.getElementById("manifest");
  if (!manifestEl) {
    console.error("Manifest element not found");
    return;
  }

  const manifest: EndpointManifest = JSON.parse(manifestEl.textContent || "{}");

  renderCharts(manifest);

  const promptData = await Promise.all(
    manifest.prompts.map((p) => loadPromptData(manifest, p))
  );

  promptData.sort((a, b) => b.total - a.total);

  const totalQueries = promptData.reduce((sum, p) => sum + p.total, 0);
  const totalErrors = promptData.reduce((sum, p) => sum + p.errors, 0);
  const totalSuccess = totalQueries - totalErrors;

  const totalEl = document.getElementById("total-count");
  const successEl = document.getElementById("success-count");
  const errorEl = document.getElementById("error-count");

  if (totalEl) totalEl.textContent = formatNum(totalQueries);
  if (successEl) successEl.textContent = formatNum(totalSuccess);
  if (errorEl) errorEl.textContent = formatNum(totalErrors);

  const tbody = document.getElementById("prompts-table");
  if (tbody) {
    tbody.innerHTML = promptData
      .map(
        (p) => `
            <tr>
                <td class="prompt"><code>${escapeHtml(p.prompt)}</code></td>
                <td class="num">${formatNum(p.total)}</td>
                <td class="num success">${formatNum(p.success)}</td>
                <td class="num error">${formatNum(p.errors)}</td>
            </tr>`
      )
      .join("");
  }
}

init();
