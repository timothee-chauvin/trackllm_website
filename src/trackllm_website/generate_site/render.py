import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .lt import EndpointInfo, discover_lt_endpoints


def render_site(website_dir: Path) -> None:
    """Generate the static site."""
    data_dir = website_dir / "data" / "lt"
    endpoints_dir = website_dir / "endpoints"
    templates_dir = website_dir / "templates"

    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return

    website_dir.mkdir(parents=True, exist_ok=True)
    endpoints_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader(templates_dir), autoescape=True)
    index_template = env.get_template("index.html.j2")
    endpoint_template = env.get_template("endpoint.html.j2")

    endpoints: list[EndpointInfo] = []
    for ep in discover_lt_endpoints(data_dir):
        endpoints.append(ep)
        status = "active" if ep.is_active else f"inactive ({ep.last_query_str})"
        print(f"  {ep.model} @ {ep.provider}: {status}")

    active = sorted([e for e in endpoints if e.is_active], key=lambda e: e.model.lower())
    inactive = sorted([e for e in endpoints if not e.is_active], key=lambda e: e.model.lower())

    print(f"\nFound {len(active)} active, {len(inactive)} inactive endpoints")

    index_html = index_template.render(
        active_endpoints=active,
        inactive_endpoints=inactive,
        css_path="style.css",
        body_class="index",
    )
    (website_dir / "index.html").write_text(index_html)
    print("Generated index.html")

    for f in endpoints_dir.glob("*.html"):
        f.unlink()

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
        (endpoints_dir / f"{ep.slug}.html").write_text(endpoint_html)

    print(f"Generated {len(endpoints)} endpoint pages in endpoints/")
    print(f"\nSite generated in {website_dir}/")
