import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from trackllm_website.generate_site import b3it as b3it_mod
from trackllm_website.generate_site import changes as changes_mod
from trackllm_website.generate_site import spend as spend_mod

from .lt import EndpointInfo, discover_lt_endpoints

RECENT_CHANGE_DAYS = 14
FEED_LIMIT = 25


@dataclass
class IndexRow:
    slug: str
    model: str
    provider: str
    lt_status: str | None
    b3it_status: str | None
    b3it_reason: str | None
    recent_change: bool


def build_index_rows(
    lt_endpoints: list[EndpointInfo],
    b3it_views: dict,
    recent_slugs: set[str],
) -> list[IndexRow]:
    lt_by_slug = {e.slug: e for e in lt_endpoints}
    all_slugs = sorted(
        set(lt_by_slug) | set(b3it_views),
        key=lambda s: (lt_by_slug.get(s) or b3it_views[s]).model.lower(),
    )
    rows = []
    for slug in all_slugs:
        ep = lt_by_slug.get(slug)
        view = b3it_views.get(slug)
        if ep:
            model, provider = ep.model, ep.provider
        else:
            model, provider = view.model, view.provider
        lt_status = ("monitoring" if ep.is_active else "retired") if ep else None
        b3it_status = view.status if view else None
        b3it_reason = view.retired_reason if view else None
        rows.append(
            IndexRow(
                slug,
                model,
                provider,
                lt_status,
                b3it_status,
                b3it_reason,
                slug in recent_slugs,
            )
        )
    return rows


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

    cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_CHANGE_DAYS)
    recent_slugs: set[str] = {
        e.slug for e in events if datetime.fromisoformat(e.date) > cutoff
    }

    rows = build_index_rows(endpoints, b3it_views, recent_slugs)

    n_active = sum(1 for r in rows if r.lt_status == "monitoring")
    print(f"\nFound {n_active} active, {len(rows) - n_active} inactive endpoints")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    spend = spend_mod.aggregate_spend(website_dir / "data" / "spend", today)
    (website_dir / "data" / "spend.json").write_text(json.dumps(spend))

    index_html = index_template.render(
        rows=rows,
        changes=changes_json[:FEED_LIMIT],
        spend=spend,
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
            provider=provider,
            methods=methods,
            manifest_json=json.dumps(manifest),
            css_path="../style.css",
            body_class="endpoint",
        )
        (endpoints_dir / f"{slug}.html").write_text(endpoint_html)

    total = len(set(lt_by_slug) | set(b3it_views))
    print(f"Generated {total} endpoint pages in endpoints/")
    print(f"\nSite generated in {website_dir}/")
