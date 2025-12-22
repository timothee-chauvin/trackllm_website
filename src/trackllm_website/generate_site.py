#!/usr/bin/env python3
"""Generate a static GitHub Pages website from the trackllm data directory."""

import json
from dataclasses import dataclass
from pathlib import Path

WEBSITE_DIR = Path("website")
DATA_DIR = WEBSITE_DIR / "data"
ENDPOINTS_DIR = WEBSITE_DIR / "endpoints"


@dataclass
class PromptInfo:
    """Info about a prompt directory."""

    slug: str
    prompt: str
    months: list[str]


@dataclass
class EndpointInfo:
    """Info about an endpoint."""

    model: str
    provider: str
    slug: str
    prompts: list[PromptInfo]


def get_endpoint_info(endpoint_dir: Path) -> EndpointInfo | None:
    """Get info about an endpoint directory."""
    prompts: list[PromptInfo] = []
    model = None
    provider = None

    for prompt_dir in sorted(endpoint_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue

        info_file = prompt_dir / "info.json"
        if not info_file.exists():
            continue

        with open(info_file, "r") as f:
            info = json.load(f)

        if model is None:
            endpoint = info.get("endpoint", {})
            model = endpoint.get("model", "Unknown")
            provider = endpoint.get("provider", "Unknown")

        months = sorted(
            [d.name for d in prompt_dir.iterdir() if d.is_dir() and "-" in d.name]
        )

        if months:
            prompts.append(
                PromptInfo(
                    slug=prompt_dir.name,
                    prompt=info.get("prompt", "Unknown"),
                    months=months,
                )
            )

    if not prompts or model is None:
        return None

    return EndpointInfo(
        model=model, provider=provider, slug=endpoint_dir.name, prompts=prompts
    )


def generate_index_html(endpoints: list[EndpointInfo]) -> str:
    """Generate the main index.html page."""
    endpoints = sorted(endpoints, key=lambda e: e.model.lower())

    rows = "\n".join(
        f"""        <tr>
            <td><a href="endpoints/{ep.slug}.html">{ep.model}</a></td>
            <td>{ep.provider}</td>
        </tr>"""
        for ep in endpoints
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrackLLM</title>
    <link rel="stylesheet" href="style.css">
</head>
<body class="index">
    <div class="container">
        <h1>TrackLLM</h1>
        <p class="subtitle">Tracking LLM API logprob responses across {len(endpoints)} endpoints</p>

        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Provider</th>
                </tr>
            </thead>
            <tbody>
{rows}
            </tbody>
        </table>

        <footer>
            Data collected via OpenRouter API
        </footer>
    </div>
</body>
</html>
"""


def generate_endpoint_html(ep: EndpointInfo) -> str:
    """Generate an endpoint detail page with JS data fetching."""
    manifest = {
        "model": ep.model,
        "provider": ep.provider,
        "slug": ep.slug,
        "prompts": [
            {"slug": p.slug, "prompt": p.prompt, "months": p.months} for p in ep.prompts
        ],
    }
    manifest_json = json.dumps(manifest)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{ep.model} @ {ep.provider} | TrackLLM</title>
    <link rel="stylesheet" href="../style.css">
</head>
<body class="endpoint">
    <div class="container">
        <a href="../index.html" class="back">← all endpoints</a>
        <h1>{ep.model}</h1>
        <p class="provider">@ {ep.provider}</p>

        <div class="stats">
            <div class="stat">
                <div class="stat-value" id="total-count">...</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat">
                <div class="stat-value success" id="success-count">...</div>
                <div class="stat-label">Success</div>
            </div>
            <div class="stat">
                <div class="stat-value error" id="error-count">...</div>
                <div class="stat-label">Errors</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Prompt</th>
                    <th class="num">Total</th>
                    <th class="num">Success</th>
                    <th class="num">Errors</th>
                </tr>
            </thead>
            <tbody id="prompts-table">
                <tr><td colspan="4" class="loading">Loading data...</td></tr>
            </tbody>
        </table>

        <footer>
            <a href="../index.html">← back to all endpoints</a>
        </footer>
    </div>

    <script>
    const MANIFEST = {manifest_json};

    async function fetchQueries(endpointSlug, promptSlug, month) {{
        const url = `../data/${{endpointSlug}}/${{promptSlug}}/${{month}}/queries.json`;
        try {{
            const res = await fetch(url);
            if (!res.ok) return {{ total: 0, errors: 0 }};
            const queries = await res.json();
            const total = queries.length;
            const errors = queries.filter(q => typeof q[1] === 'string' && q[1].startsWith('e')).length;
            return {{ total, errors }};
        }} catch {{
            return {{ total: 0, errors: 0 }};
        }}
    }}

    async function loadPromptData(prompt) {{
        const results = await Promise.all(
            prompt.months.map(month => fetchQueries(MANIFEST.slug, prompt.slug, month))
        );
        const total = results.reduce((sum, r) => sum + r.total, 0);
        const errors = results.reduce((sum, r) => sum + r.errors, 0);
        return {{ prompt: prompt.prompt, total, errors, success: total - errors }};
    }}

    function formatNum(n) {{
        return n.toLocaleString();
    }}

    async function init() {{
        const promptData = await Promise.all(MANIFEST.prompts.map(loadPromptData));
        promptData.sort((a, b) => b.total - a.total);

        const totalQueries = promptData.reduce((sum, p) => sum + p.total, 0);
        const totalErrors = promptData.reduce((sum, p) => sum + p.errors, 0);
        const totalSuccess = totalQueries - totalErrors;

        document.getElementById('total-count').textContent = formatNum(totalQueries);
        document.getElementById('success-count').textContent = formatNum(totalSuccess);
        document.getElementById('error-count').textContent = formatNum(totalErrors);

        document.getElementById('prompts-table').innerHTML = promptData.map(p => `
            <tr>
                <td class="prompt"><code>${{p.prompt}}</code></td>
                <td class="num">${{formatNum(p.total)}}</td>
                <td class="num success">${{formatNum(p.success)}}</td>
                <td class="num error">${{formatNum(p.errors)}}</td>
            </tr>
        `).join('');
    }}

    init();
    </script>
</body>
</html>
"""


def main():
    """Generate the static site."""
    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} does not exist")
        return

    WEBSITE_DIR.mkdir(parents=True, exist_ok=True)
    ENDPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect endpoint info
    endpoints: list[EndpointInfo] = []

    for endpoint_dir in sorted(DATA_DIR.iterdir()):
        if not endpoint_dir.is_dir():
            continue

        info = get_endpoint_info(endpoint_dir)
        if info:
            endpoints.append(info)
            print(f"  {info.model} @ {info.provider}: {len(info.prompts)} prompts")

    print(f"\nFound {len(endpoints)} endpoints")

    # Generate index
    (WEBSITE_DIR / "index.html").write_text(generate_index_html(endpoints))
    print("Generated index.html")

    # Clean old endpoint pages
    for f in ENDPOINTS_DIR.glob("*.html"):
        f.unlink()

    # Generate endpoint pages
    for ep in endpoints:
        (ENDPOINTS_DIR / f"{ep.slug}.html").write_text(generate_endpoint_html(ep))
    print(f"Generated {len(endpoints)} endpoint pages in endpoints/")

    print(f"\nSite generated in {WEBSITE_DIR}/")


if __name__ == "__main__":
    main()
