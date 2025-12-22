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

