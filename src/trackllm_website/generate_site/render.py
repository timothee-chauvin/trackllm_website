import hashlib
import json
from datetime import datetime
from pathlib import Path

import orjson
from jinja2 import Environment, FileSystemLoader

from trackllm_website.bi.state import load_all_states
from trackllm_website.config import HeroConfig
from trackllm_website.generate_site import b3it as b3it_mod
from trackllm_website.generate_site import changes as changes_mod
from trackllm_website.generate_site import changes_page as changes_page_mod
from trackllm_website.generate_site import home as home_mod
from trackllm_website.generate_site import lt_raw as lt_raw_mod
from trackllm_website.generate_site import machine as machine_mod
from trackllm_website.generate_site import manifest as manifest_mod
from trackllm_website.generate_site import model as model_mod
from trackllm_website.generate_site import org as org_mod
from trackllm_website.generate_site import overview as overview_mod
from trackllm_website.generate_site import provider as provider_mod
from trackllm_website.generate_site import spend as spend_mod
from trackllm_website.generate_site.brands import brand_json, load_brands
from trackllm_website.generate_site.naming import base_provider, variant_name
from trackllm_website.generate_site.papers import PAPERS
from trackllm_website.generate_site.status import STATUS_COPY, status_json
from trackllm_website.generate_site.status_io import (
    StatusInputs,
    lt_stalled_slugs,
    resolve_site_statuses,
)
from trackllm_website.generate_site.tracked import with_observations
from trackllm_website.util import format_cost, format_price, slugify

from .lt import EndpointInfo, discover_lt_endpoints, load_all_lt_data


def write_json_dir(directory: Path, views: dict[str, dict]) -> None:
    """Rewrite a generated `<slug>.json` directory from scratch.

    Pruning first is what the page directories already do: an entity that has left
    the site keeps serving its last JSON otherwise, at a URL the site itself no
    longer links but anything else may still fetch.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for f in directory.glob("*.json"):
        f.unlink()
    for slug, view in views.items():
        (directory / f"{slug}.json").write_text(json.dumps(view))


def asset_version(website_dir: Path) -> str:
    """Content hash over the fingerprintless assets the templates reference."""
    h = hashlib.sha256()
    for f in [website_dir / "style.css", *sorted((website_dir / "js").glob("*.js"))]:
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


def render_site(
    website_dir: Path,
    hero_pin: HeroConfig | None,
    status_inputs: StatusInputs,
    built_at: datetime,
) -> None:
    """Generate the static site.

    `hero_pin`, `status_inputs` and `built_at` are threaded in rather than read
    from config/committed files/the wall clock so a synthetic site can be
    rendered without them; the real build passes `config.hero`,
    `load_status_inputs()` and the current time, and a pin that cannot resolve
    raises.
    """
    data_dir = website_dir / "data" / "lt"
    endpoints_dir = website_dir / "endpoints"
    templates_dir = website_dir / "templates"

    # fail the build rather than deploy an empty site
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    website_dir.mkdir(parents=True, exist_ok=True)
    endpoints_dir.mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader(templates_dir), autoescape=True)
    # headline_badge and the "st" filter chips look their tooltip text up here, so
    # STATUS_COPY (status.py) stays the one place that copy is written.
    env.globals["STATUS_COPY"] = STATUS_COPY
    env.globals["PAPERS"] = PAPERS
    env.globals["SITE_URL"] = machine_mod.SITE_URL
    env.globals["CODE_REPO_URL"] = machine_mod.CODE_REPO_URL
    env.globals["DATA_REPO_URL"] = machine_mod.DATA_REPO_URL
    # Cache busting: GitHub Pages serves with max-age=600, so without this a
    # visitor loading freshly deployed HTML keeps rendering it against a stale
    # cached stylesheet or entrypoint for up to 10 minutes. bun's shared chunks
    # are already content-hashed; only style.css and the entrypoints need it.
    env.globals["ASSET_V"] = asset_version(website_dir)
    env.filters["fmt_cost"] = format_cost
    env.filters["fmt_price"] = format_price
    endpoint_template = env.get_template("endpoint.html.j2")
    model_template = env.get_template("model.html.j2")
    provider_template = env.get_template("provider.html.j2")
    org_template = env.get_template("org.html.j2")
    changes_template = env.get_template("changes.html.j2")

    # `built_at` is the one clock of the build: every age on the site ("stable for
    # N days", "14d ago", the 30-day windows, the retired cutoff) is measured
    # against it. The build time, not the newest observation: that one lagged a
    # day behind the calendar, and every "stable for" count with it.
    # Every page's render context is kept: machine.py renders the .md twin from
    # the same context, so the two can never show different data.
    md_pages: dict[str, list[dict]] = {}
    html_pages: list[str] = []

    def page(kind: str, slug: str, json_paths: list[str], **ctx) -> dict:
        ctx["machine"] = machine_mod.links(kind, slug, json_paths)
        md_pages.setdefault(kind, []).append(ctx)
        html_pages.append(ctx["machine"]["html"])
        return ctx

    discovered: list[EndpointInfo] = []
    for ep in discover_lt_endpoints(data_dir):
        discovered.append(ep)
        status = "active" if ep.is_active else f"inactive ({ep.last_query_str})"
        print(f"  {ep.model} @ {ep.provider}: {status}")

    b3it_dir = website_dir / "data" / "b3it"
    b3it_views = b3it_mod.discover_b3it_views(
        b3it_dir / "state",
        b3it_dir / "phase_2",
        b3it_dir / "scan_backfill.json",
    )

    # Parsed once for the overview, provider and changes-page builders (~400
    # files); build_model_views is not on it yet, it re-reads each
    # lt_scores.json itself. Loaded here, ahead of every builder, because the
    # fleet the site shows is the one that has a series at all (tracked.py).
    lt_data = load_all_lt_data(data_dir, [e.slug for e in discovered])
    n_discovered = len({e.slug for e in discovered} | set(b3it_views))
    lt_by_slug, b3it_views = with_observations(
        {e.slug: e for e in discovered}, lt_data, b3it_views
    )
    endpoints = [e for e in discovered if e.slug in lt_by_slug]
    # The drift lane's hover readout: daily.json beside each lt_scores.json.
    lt_raw_mod.write_lt_daily(data_dir, endpoints, lt_data)
    n_skipped = n_discovered - len(set(lt_by_slug) | set(b3it_views))
    if n_skipped:
        print(f"Skipping {n_skipped} endpoints with nothing to show (no series)")

    # Every catalog / previously-tracked endpoint gets a status (status.py); the
    # series-bearing fleet above additionally gets charts (tracked.py).
    bi_states = load_all_states(website_dir / "data" / "b3it" / "state")
    lt_stalled = lt_stalled_slugs(data_dir, status_inputs.endpoints_lt, set(lt_by_slug))
    site = resolve_site_statuses(status_inputs, lt_by_slug, lt_stalled, bi_states)

    # Only the generated b3it.json and votes.json are pruned (and the directory
    # they leave empty): state/ and phase_2/ sit under the same parent and are
    # collected data.
    for f in [*b3it_dir.glob("*/b3it.json"), *b3it_dir.glob("*/votes.json")]:
        f.unlink()
        if not any(f.parent.iterdir()):
            f.parent.rmdir()
    for slug, view in b3it_views.items():
        out_dir = b3it_dir / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "b3it.json").write_text(json.dumps(b3it_mod.to_json(view)))
        (out_dir / "votes.json").write_bytes(orjson.dumps(b3it_mod.votes_json(view)))

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

    today = built_at.strftime("%Y-%m-%d")
    spend = spend_mod.aggregate_spend(website_dir / "data" / "spend", today)
    (website_dir / "data" / "spend.json").write_text(json.dumps(spend))

    overview = overview_mod.build_overview(
        website_dir, lt_data, endpoints, b3it_views, hero_pin, site, built_at
    )
    provider_views = provider_mod.build_provider_views(
        website_dir,
        lt_data,
        endpoints,
        b3it_views,
        overview["endpoints"],
        site,
        built_at,
    )
    overview["providers"] = provider_mod.overview_rows(provider_views)
    (website_dir / "data" / "overview.json").write_text(json.dumps(overview))

    write_json_dir(website_dir / "data" / "providers", provider_views)

    provider_pages_dir = website_dir / "providers"
    provider_pages_dir.mkdir(parents=True, exist_ok=True)
    for f in provider_pages_dir.glob("*.html"):
        f.unlink()

    for pslug, view in provider_views.items():
        (provider_pages_dir / f"{pslug}.html").write_text(
            provider_template.render(
                **page(
                    "provider",
                    pslug,
                    [f"data/providers/{pslug}.json"],
                    provider=view["name"],
                    provider_slug=pslug,
                    brand=view["brand"],
                    view=view,
                    css_path="../style.css",
                    body_class="provider",
                    nav_prefix="../",
                )
            )
        )
    print(f"Generated {len(provider_views)} provider pages in providers/")

    changes_page = changes_page_mod.build_changes_page(
        website_dir, lt_data, b3it_views, built_at
    )
    (website_dir / "data" / "changes_page.json").write_text(json.dumps(changes_page))
    (website_dir / "changes.html").write_text(
        changes_template.render(
            **page(
                "changes",
                "",
                ["data/changes_page.json", "data/changes.json"],
                changes_page=changes_page,
                css_path="style.css",
                body_class="changes",
                nav_prefix="",
            )
        )
    )
    print("Generated changes.html")

    for name in ("methodology", "about", "github"):
        (website_dir / f"{name}.html").write_text(
            env.get_template(f"{name}.html.j2").render(
                **page(
                    name, "", [], css_path="style.css", body_class=name, nav_prefix=""
                )
            )
        )
        print(f"Generated {name}.html")

    # GitHub Pages serves 404.html for every missing path, at any depth, so its
    # links are absolute. Not registered through page(): no md twin, no sitemap entry.
    (website_dir / "404.html").write_text(
        env.get_template("404.html.j2").render(
            machine=machine_mod.links("404", "", []),
            css_path="/style.css",
            body_class="notfound",
            nav_prefix="/",
        )
    )
    print("Generated 404.html")

    model_views = model_mod.build_model_views(website_dir, endpoints, b3it_views, site)
    write_json_dir(website_dir / "data" / "models", model_views)

    model_pages_dir = website_dir / "models"
    model_pages_dir.mkdir(parents=True, exist_ok=True)
    for f in model_pages_dir.glob("*.html"):
        f.unlink()

    for mslug, view in model_views.items():
        model_html = model_template.render(
            **page(
                "model",
                mslug,
                [f"data/models/{mslug}.json"],
                model_slug=mslug,
                model=view["model"],
                org=view["org"],
                org_slug=slugify(view["org"]),
                view=view,
                css_path="../style.css",
                body_class="model",
                nav_prefix="../",
            )
        )
        (model_pages_dir / f"{mslug}.html").write_text(model_html)
    print(f"Generated {len(model_views)} model pages in models/")

    org_views = org_mod.build_org_views(model_views)
    org_pages_dir = website_dir / "orgs"
    org_pages_dir.mkdir(parents=True, exist_ok=True)
    for f in org_pages_dir.glob("*.html"):
        f.unlink()

    for oslug, view in org_views.items():
        (org_pages_dir / f"{oslug}.html").write_text(
            org_template.render(
                **page(
                    "org",
                    oslug,
                    [],
                    org=view["name"],
                    view=view,
                    css_path="../style.css",
                    body_class="org",
                    nav_prefix="../",
                )
            )
        )
    print(f"Generated {len(org_views)} org pages in orgs/")

    manifests = manifest_mod.build_manifests(overview["endpoints"], model_views)

    # slug -> (model_slug, n_endpoints) so endpoint pages can link to their model
    # page with an endpoint count consistent with that model's own page (Task 7).
    slug_to_model_slug: dict[str, str] = {}
    slug_to_n_endpoints: dict[str, int] = {}
    slug_to_n_providers: dict[str, int] = {}
    slug_to_status_summary: dict[str, str] = {}
    for mslug, view in model_views.items():
        n_endpoints = view["n_endpoints"]
        n_providers = view["n_providers"]
        for e in view["endpoints"]:
            slug_to_model_slug[e["slug"]] = mslug
            slug_to_n_endpoints[e["slug"]] = n_endpoints
            slug_to_n_providers[e["slug"]] = n_providers
            slug_to_status_summary[e["slug"]] = view["status_summary"]

    # The two pages the front page's provider and endpoint sections are slices
    # of; both render from overview.json.
    for kind in ("providers", "endpoints"):
        (website_dir / f"{kind}.html").write_text(
            env.get_template(f"{kind}.html.j2").render(
                **page(
                    kind,
                    "",
                    ["data/overview.json"],
                    overview=overview,
                    css_path="style.css",
                    body_class=kind,
                    nav_prefix="",
                )
            )
        )
        print(f"Generated {kind}.html")

    # The front page renders from its own slice, inlined into the HTML (home.py);
    # `bun run prerender` then fills its sections in place (website/tools/prerender.ts).
    home = home_mod.build_home(overview)
    (website_dir / "data" / "home.json").write_text(json.dumps(home))
    (website_dir / "index.html").write_text(
        env.get_template("index.html.j2").render(
            **page(
                "index",
                "",
                ["data/home.json"],
                home=home,
                css_path="style.css",
                body_class="index",
                nav_prefix="",
            )
        )
    )
    print("Generated index.html")

    for f in endpoints_dir.glob("*.html"):
        f.unlink()

    # the endpoint head shows the serving company as the provider page does; a
    # provider without a page still gets its brand (the slug, when unlisted)
    brands = load_brands(website_dir)

    # One page per status universe entry: tracked ones with their series, the
    # rest with the status + catalog metadata explaining why there is no chart.
    for slug in sorted(site.statuses):
        methods: list[str] = []
        if slug in lt_by_slug:
            ep = lt_by_slug[slug]
            model = ep.model
            provider = ep.provider
            methods.append("LT")
        elif slug in b3it_views:
            ep = None
            view = b3it_views[slug]
            model = view.model
            provider = view.provider
        else:
            ep = None
            model, provider = site.names[slug]
        if slug in b3it_views:
            methods.append("B3IT")

        entry = site.entries.get(slug)

        provider_slug = slugify(base_provider(provider))
        model_slug = slug_to_model_slug.get(slug, "")
        series_json = [f"data/models/{model_slug}.json"] if model_slug else []
        if slug in b3it_views:
            series_json.append(f"data/b3it/{slug}/b3it.json")
        endpoint_html = endpoint_template.render(
            **page(
                "endpoint",
                slug,
                series_json,
                endpoint=ep,
                model=model,
                org=model.split("/")[0],
                org_slug=slugify(model.split("/")[0]),
                model_name=model.split("/")[-1],
                provider=provider,
                brand=brand_json(brands, provider_slug),
                variant=variant_name(provider),
                methods=methods,
                status=status_json(site.statuses[slug]),
                meta=entry.as_meta() if entry else None,
                spend=spend["by_endpoint"].get(slug),
                group_label=spend_mod.GROUP_LABEL,
                manifest=manifests[slug],
                css_path="../style.css",
                body_class="endpoint",
                nav_prefix="../",
                provider_base=base_provider(provider),
                provider_slug=provider_slug,
                # only providers with tracked endpoints get a page; never link a 404
                provider_has_page=provider_slug in provider_views,
                model_slug=model_slug,
                n_endpoints=slug_to_n_endpoints.get(slug, 1),
                n_providers=slug_to_n_providers.get(slug, 1),
                n_models=provider_views.get(provider_slug, {}).get("n_models", 1),
                status_summary=slug_to_status_summary.get(slug, ""),
            )
        )
        (endpoints_dir / f"{slug}.html").write_text(endpoint_html)

    print(f"Generated {len(site.statuses)} endpoint pages in endpoints/")

    machine_mod.render_markdown(website_dir, built_at, md_pages)
    machine_mod.render_sitemap(website_dir, built_at, html_pages)
    machine_mod.render_feeds(
        website_dir,
        built_at,
        changes_page["items"],
        {
            "endpoint": set(site.statuses),
            "model": set(model_views),
            "provider": set(provider_views),
            "org": set(org_views),
        },
    )
    print(f"\nSite generated in {website_dir}/")
