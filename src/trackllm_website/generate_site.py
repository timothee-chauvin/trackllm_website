#!/usr/bin/env python3
"""Generate a static GitHub Pages website from the trackllm data directory."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

WEBSITE_DIR = Path("website")
DATA_DIR = WEBSITE_DIR / "data"
ENDPOINTS_DIR = WEBSITE_DIR / "endpoints"
TEMPLATES_DIR = WEBSITE_DIR / "templates"

# Endpoints with last query older than this are considered inactive
INACTIVE_THRESHOLD_DAYS = 3


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
    last_query_date: datetime | None = None

    @property
    def is_active(self) -> bool:
        if self.last_query_date is None:
            return False
        threshold = datetime.now(timezone.utc) - timedelta(days=INACTIVE_THRESHOLD_DAYS)
        return self.last_query_date > threshold

    @property
    def last_query_str(self) -> str:
        if self.last_query_date is None:
            return "Never"
        return self.last_query_date.strftime("%Y-%m-%d")


def get_last_query_date(endpoint_dir: Path) -> datetime | None:
    """Get the date of the last successful query for an endpoint."""
    latest_date: datetime | None = None

    for prompt_dir in endpoint_dir.iterdir():
        if not prompt_dir.is_dir():
            continue

        # Get month directories sorted in reverse (newest first)
        month_dirs = sorted(
            [d for d in prompt_dir.iterdir() if d.is_dir() and "-" in d.name],
            reverse=True,
        )

        for month_dir in month_dirs:
            queries_file = month_dir / "queries.json"
            if not queries_file.exists():
                continue

            try:
                with open(queries_file, "r") as f:
                    queries = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            if not queries:
                continue

            # Parse year-month from directory name
            try:
                year, month = map(int, month_dir.name.split("-"))
            except ValueError:
                continue

            # Find the last successful query (not an error)
            for date_str, idx in reversed(queries):
                # Skip errors (e.g., "e0", "e1")
                if isinstance(idx, str) and idx.startswith("e"):
                    continue

                # Parse date "DD HH:MM:SS"
                try:
                    day_time = datetime.strptime(date_str, "%d %H:%M:%S")
                    query_date = datetime(
                        year,
                        month,
                        day_time.day,
                        day_time.hour,
                        day_time.minute,
                        day_time.second,
                        tzinfo=timezone.utc,
                    )

                    if latest_date is None or query_date > latest_date:
                        latest_date = query_date

                    # Found latest in this month, move to next prompt
                    break
                except ValueError:
                    continue

            # If we found a date in the latest month, no need to check older months
            if latest_date is not None:
                break

    return latest_date


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

    last_query_date = get_last_query_date(endpoint_dir)

    return EndpointInfo(
        model=model,
        provider=provider,
        slug=endpoint_dir.name,
        prompts=prompts,
        last_query_date=last_query_date,
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
            status = "active" if info.is_active else f"inactive ({info.last_query_str})"
            print(f"  {info.model} @ {info.provider}: {status}")

    # Split into active and inactive
    active = [e for e in endpoints if e.is_active]
    inactive = [e for e in endpoints if not e.is_active]

    # Sort alphabetically
    active.sort(key=lambda e: e.model.lower())
    inactive.sort(key=lambda e: e.model.lower())

    print(f"\nFound {len(active)} active, {len(inactive)} inactive endpoints")

    # Generate index
    index_html = index_template.render(
        active_endpoints=active,
        inactive_endpoints=inactive,
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
