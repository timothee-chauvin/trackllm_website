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
  logprobs: number[];
}

interface LogprobsData {
  seen_tokens: string[];
  seen_logprobs: { tokens: number[]; logprobs: number[] }[];
}


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
      // Skip errors
      if (typeof idx === "string") continue;

      // Parse date "DD HH:MM:SS"
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

      entries.push({ date, logprobs: logprobVec.logprobs });
    }

    return entries;
  } catch {
    return [];
  }
}

async function fetchLogprobsForPrompt(
  manifest: EndpointManifest,
  prompt: PromptManifest,
  limit: number
): Promise<LogprobQuery[]> {
  const allEntries: LogprobQuery[] = [];

  // Process months in reverse order (newest first)
  const sortedMonths = [...prompt.months].sort().reverse();

  for (const month of sortedMonths) {
    const entries = await fetchLogprobsForMonth(
      manifest.slug,
      prompt.slug,
      month
    );
    allEntries.push(...entries);

    // Early exit if we have enough
    if (allEntries.length >= limit) break;
  }

  // Sort by date descending and take latest
  allEntries.sort((a, b) => b.date.getTime() - a.date.getTime());
  return allEntries.slice(0, limit);
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

  // Reverse to show oldest first (left to right)
  const sorted = [...entries].sort(
    (a, b) => a.date.getTime() - b.date.getTime()
  );

  // Find max number of logprobs
  const maxLogprobs = Math.max(...sorted.map((e) => e.logprobs.length));

  // Create a trace for each logprob position
  const traces = [];
  for (let i = 0; i < maxLogprobs; i++) {
    const x: Date[] = [];
    const y: number[] = [];

    for (const entry of sorted) {
      if (i < entry.logprobs.length) {
        x.push(entry.date);
        y.push(entry.logprobs[i]);
      }
    }

    if (x.length > 0) {
      traces.push({
        x,
        y,
        type: "scatter",
        mode: "lines",
        name: `Token ${i + 1}`,
        line: { width: 1 },
        hovertemplate: `Token ${i + 1}<br>%{x}<br>Logprob: %{y:.4f}<extra></extra>`,
      });
    }
  }

  const layout = {
    title: {
      text: `Logprobs for "${promptName}" (Latest ${sorted.length} Queries)`,
      font: { color: "#c9d1d9", size: 14 },
    },
    xaxis: {
      title: { text: "Date", font: { color: "#8b949e" } },
      gridcolor: "#30363d",
      tickfont: { color: "#8b949e" },
    },
    yaxis: {
      title: { text: "Log Probability", font: { color: "#8b949e" } },
      gridcolor: "#30363d",
      tickfont: { color: "#8b949e" },
    },
    paper_bgcolor: "#161b22",
    plot_bgcolor: "#0d1117",
    font: { color: "#c9d1d9" },
    legend: {
      font: { color: "#c9d1d9", size: 10 },
      bgcolor: "transparent",
      orientation: "h" as const,
      y: -0.2,
    },
    margin: { t: 40, r: 20, b: 80, l: 60 },
    showlegend: maxLogprobs <= 25,
  };

  const config = {
    responsive: true,
    displayModeBar: false,
  };

  Plotly.newPlot(container, traces, layout, config);
}

async function renderCharts(manifest: EndpointManifest): Promise<void> {
  const chartsContainer = document.getElementById("charts-container");
  if (!chartsContainer) return;

  chartsContainer.innerHTML = "";

  for (const prompt of manifest.prompts) {
    // Create chart container for this prompt
    const chartDiv = document.createElement("div");
    chartDiv.className = "chart";
    chartDiv.innerHTML = '<div class="loading">Loading chart...</div>';
    chartsContainer.appendChild(chartDiv);

    // Fetch and render (don't await, let them load in parallel visually)
    fetchLogprobsForPrompt(manifest, prompt, 100).then((entries) => {
      renderPromptChart(chartDiv, prompt.prompt, entries);
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

  // Start loading charts (async, will render as they complete)
  renderCharts(manifest);

  // Load stats
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
