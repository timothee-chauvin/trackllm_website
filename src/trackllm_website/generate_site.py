#!/usr/bin/env python3
"""Generate a static GitHub Pages website from the trackllm data directory."""

import json
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

WEBSITE_DIR = Path("website")
DATA_DIR = WEBSITE_DIR / "data"
ENDPOINTS_DIR = WEBSITE_DIR / "endpoints"
TEMPLATES_DIR = WEBSITE_DIR / "templates"


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


def main():
    """Generate the static site."""
    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} does not exist")
        return

    WEBSITE_DIR.mkdir(parents=True, exist_ok=True)
    ENDPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set up Jinja2
    env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=True)
    index_template = env.get_template("index.html.j2")
    endpoint_template = env.get_template("endpoint.html.j2")

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

    # Sort endpoints alphabetically
    endpoints.sort(key=lambda e: e.model.lower())

    # Generate index
    index_html = index_template.render(
        endpoints=endpoints,
        css_path="style.css",
        body_class="index",
    )
    (WEBSITE_DIR / "index.html").write_text(index_html)
    print("Generated index.html")

    # Clean old endpoint pages
    for f in ENDPOINTS_DIR.glob("*.html"):
        f.unlink()

    # Generate endpoint pages
    for ep in endpoints:
        manifest = {
            "model": ep.model,
            "provider": ep.provider,
            "slug": ep.slug,
            "prompts": [
                {"slug": p.slug, "prompt": p.prompt, "months": p.months}
                for p in ep.prompts
            ],
        }

        endpoint_html = endpoint_template.render(
            endpoint=ep,
            manifest_json=json.dumps(manifest),
            css_path="../style.css",
            body_class="endpoint",
        )
        (ENDPOINTS_DIR / f"{ep.slug}.html").write_text(endpoint_html)

    print(f"Generated {len(endpoints)} endpoint pages in endpoints/")
    print(f"\nSite generated in {WEBSITE_DIR}/")


if __name__ == "__main__":
    main()

