import json
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from trackllm_website.generate_site import b3it as b3it_mod
from trackllm_website.generate_site import changes as changes_mod
from trackllm_website.generate_site import model as model_mod
from trackllm_website.generate_site import overview as overview_mod
from trackllm_website.generate_site import spend as spend_mod

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
    spend_template = env.get_template("spend.html.j2")

    endpoints: list[EndpointInfo] = []
    for ep in discover_lt_endpoints(data_dir):
        endpoints.append(ep)
        status = "active" if ep.is_active else f"inactive ({ep.last_query_str})"
        print(f"  {ep.model} @ {ep.provider}: {status}")

    lt_by_slug = {e.slug: e for e in endpoints}

    b3it_views = b3it_mod.discover_b3it_views(
        website_dir / "data" / "b3it" / "state",
        website_dir / "data" / "b3it" / "phase_2",
    )
    for slug, view in b3it_views.items():
        out_dir = website_dir / "data" / "b3it" / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "b3it.json").write_text(json.dumps(b3it_mod.to_json(view)))

    lt_changes_file = website_dir / "data" / "lt" / "lt_changes.json"
    lt_changes = (
        json.loads(lt_changes_file.read_text()) if lt_changes_file.exists() else {}
    )

    events = changes_mod.merge_changes(lt_changes, lt_by_slug, b3it_views)
    changes_json = changes_mod.to_json(events)
    (website_dir / "data").mkdir(parents=True, exist_ok=True)
    (website_dir / "data" / "changes.json").write_text(json.dumps(changes_json))

    n_active = sum(1 for ep in endpoints if ep.is_active)
    n_total = len(set(lt_by_slug) | set(b3it_views))
    print(f"\nFound {n_active} active, {n_total - n_active} inactive endpoints")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    spend = spend_mod.aggregate_spend(website_dir / "data" / "spend", today)
    (website_dir / "data" / "spend.json").write_text(json.dumps(spend))

    (website_dir / "data" / "overview.json").write_text(
        json.dumps(overview_mod.build_overview(website_dir, endpoints, b3it_views))
    )
    models_dir = website_dir / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_views = model_mod.build_model_views(website_dir, endpoints, b3it_views)
    for mslug, view in model_views.items():
        (models_dir / f"{mslug}.json").write_text(json.dumps(view))

    # slug -> (model_slug, n_providers) so endpoint pages can link to their model
    # page with a provider count consistent with that model's own page (Task 7).
    slug_to_model_slug: dict[str, str] = {}
    slug_to_n_providers: dict[str, int] = {}
    for mslug, view in model_views.items():
        n_providers = len(view["endpoints"])
        for e in view["endpoints"]:
            slug_to_model_slug[e["slug"]] = mslug
            slug_to_n_providers[e["slug"]] = n_providers

    index_html = index_template.render(
        css_path="style.css",
        body_class="index",
    )
    (website_dir / "index.html").write_text(index_html)
    print("Generated index.html")

    spend_html = spend_template.render(
        spend=spend,
        css_path="style.css",
        body_class="spend",
    )
    (website_dir / "spend.html").write_text(spend_html)
    print("Generated spend.html")

    for f in endpoints_dir.glob("*.html"):
        f.unlink()

    for slug in sorted(set(lt_by_slug) | set(b3it_views)):
        methods: list[str] = []
        if slug in lt_by_slug:
            ep = lt_by_slug[slug]
            model = ep.model
            provider = ep.provider
            manifest = {
                "model": ep.model,
                "provider": ep.provider,
                "slug": ep.slug,
                "prompts": [
                    {"slug": p.slug, "prompt": p.prompt, "months": p.months}
                    for p in ep.prompts
                ],
            }
            methods.append("LT")
        else:
            ep = None
            view = b3it_views[slug]
            model = view.model
            provider = view.provider
            manifest = {
                "model": view.model,
                "provider": view.provider,
                "slug": slug,
                "prompts": [],
            }
        if slug in b3it_views:
            methods.append("B3IT")

        endpoint_html = endpoint_template.render(
            endpoint=ep,
            model=model,
            org=model.split("/")[0],
            model_name=model.split("/")[-1],
            provider=provider,
            methods=methods,
            manifest_json=json.dumps(manifest),
            css_path="../style.css",
            body_class="endpoint",
            nav_prefix="../",
            model_slug=slug_to_model_slug.get(slug, ""),
            n_providers=slug_to_n_providers.get(slug, 1),
        )
        (endpoints_dir / f"{slug}.html").write_text(endpoint_html)

    total = len(set(lt_by_slug) | set(b3it_views))
    print(f"Generated {total} endpoint pages in endpoints/")
    print(f"\nSite generated in {website_dir}/")
