"""Machine access: the Markdown mirror of every page, and the Atom feeds.

Every `X.html` gets an `X.md` sibling rendered from the same page JSON the
TypeScript consumes, so an agent navigates the site the way a human does. Feeds
are one entry per detected change, filtered per endpoint / model / provider /
org; the global one is capped.

`links(kind, slug)` is the one place that says where a page's md, feed and JSON
live -- the HTML templates (footer, <link rel=alternate>) and the md header both
read it, so the two can never point at different files.
"""

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from trackllm_website.generate_site.home import (
    DIR_PREVIEW,
    PLOT_PREVIEW,
    rateable_rows,
)
from trackllm_website.util import format_cost, format_price, slugify

SITE_URL = "https://www.trackllm.net"
CODE_REPO_URL = "https://github.com/timothee-chauvin/trackllm_website"
DATA_REPO_URL = "https://github.com/timothee-chauvin/trackllm_data"
# The characters util.slugify keeps as they are; everything else in a slug is that
# character's two-digit hex code. llms.txt spells this out for agents.
SLUG_KEPT_CHARS = "A-Z a-z 0-9 . _ - + = @ ~ ,"
GLOBAL_FEED_CAP = 200
# Links in the .md twins are absolute: the pages are made to be copied into an
# LLM context, where a relative link points nowhere.
_BASE = f"{SITE_URL}/"

# scope kind -> (page directory, feed directory); "" = a top-level page
_PAGE_DIRS = {
    "endpoint": "endpoints",
    "model": "models",
    "provider": "providers",
    "org": "orgs",
}
FEED_SCOPES = ("endpoint", "model", "provider", "org")
# What a page's JSON link is called: by the directory the file lives in, else by
# its name -- "JSON 1 · JSON 2" told a reader nothing.
_JSON_LABELS = {
    "models": "model JSON",
    "providers": "provider JSON",
    "b3it": "B3IT JSON",
    "overview.json": "overview JSON",
    "home.json": "home JSON",
    "changes.json": "changes JSON",
    "changes_page.json": "change log JSON",
}


def _json_link(path: str) -> dict:
    parts = path.split("/")
    label = _JSON_LABELS.get(parts[1]) or _JSON_LABELS.get(parts[-1]) or parts[-1]
    return {"path": path, "label": label}


def _scope_slugs(item: dict) -> dict[str, str]:
    return {
        "endpoint": item["slug"],
        "model": item["modelSlug"],
        "provider": item["providerSlug"],
        "org": slugify(item["org"]),
    }


def links(kind: str, slug: str, json_paths: list[str]) -> dict:
    """Site-root-relative paths for one page's machine-readable siblings.

    `kind` is a scope kind or a top-level page name ("index", "changes", ...).
    Top-level pages have no feed of their own except the index, which carries the
    global feed. The 404 page is served for every missing path, so it has no
    markdown twin: there is no page for one to mirror.
    """
    if kind in _PAGE_DIRS:
        page = f"{_PAGE_DIRS[kind]}/{slug}"
        feed = f"feeds/{_PAGE_DIRS[kind]}/{slug}.xml"
        scope = f"{kind}/{slug}"
        noun = "organization" if kind == "org" else kind
        subscribe_label = f"Subscribe to this {noun}'s changes"
    else:
        page = kind
        feed = "feeds/all.xml" if kind == "index" else None
        scope = "all" if kind == "index" else None
        subscribe_label = "Subscribe to all changes" if feed else None
    return {
        "html": f"{page}.html",
        "md": None if kind == "404" else f"{page}.md",
        "feed": feed,
        "scope": scope,
        "subscribe_label": subscribe_label,
        "json": [_json_link(p) for p in json_paths],
    }


def _link(text: str, href: str | None) -> str:
    return f"[{text}]({href})" if href else text


def _ep(nav: str, slug: str, text: str) -> str:
    return _link(text, f"{nav}endpoints/{slug}.md")


def _model(nav: str, slug: str | None, text: str) -> str:
    return _link(text, f"{nav}models/{slug}.md" if slug else None)


def _provider(nav: str, slug: str | None, text: str) -> str:
    return _link(text, f"{nav}providers/{slug}.md" if slug else None)


def _org(nav: str, org: str) -> str:
    return _link(org, f"{nav}orgs/{slugify(org)}.md")


def _methods(ms: list[str]) -> str:
    return " ".join(f"`{m.upper()}`" for m in ms)


def _rate(r: float | None) -> str:
    return "—" if r is None else f"{r:.2f}"


def _ci(ci: list | None) -> str:
    return "—" if ci is None else f"{ci[0]:.2f}–{ci[1]:.2f}"


# Table-row builders, one per table the TypeScript renders from page JSON. They
# take the JSON row verbatim, so a field the JSON stops carrying fails the build
# here instead of rendering "undefined" in the md.
def _dir_row(r: dict, nav: str) -> tuple:
    return (
        _ep(nav, r["slug"], r["model"]),
        _provider(nav, r["providerSlug"], r["provider"]),
        _org(nav, r["org"]),
        ", ".join(h.replace("_", " ") for h in r["headlines"]),
        r["nChanges"],
        _methods(r["methods"]),
        r["lastChange"] or "—",
    )


def _model_ep_row(e: dict, nav: str) -> tuple:
    return (
        _ep(nav, e["slug"], e["slug"]),
        _provider(nav, e["providerSlug"], e["provider"]),
        _methods(e["methods"]),
        e["status"]["headline"].replace("_", " "),
        f"{e['first']} – {e['last']}" if e["first"] and e["last"] else None,
        e["n_changes"],
    )


def _model_change_row(c: dict) -> tuple:
    return (c["date"], c["method"].upper(), c["provider"])


def _variant_row(v: dict) -> tuple:
    return (
        v["name"],
        v["n_endpoints"],
        _rate(v["lt"]["rate"]),
        _ci(v["lt"]["ci"]),
        v["lt"]["changes"],
        v["b3it"]["endpoints"],
        v["b3it"]["changes"],
    )


def _change_row(c: dict, nav: str) -> tuple:
    return (
        c["date"],
        _ep(nav, c["slug"], f"{c['model']} @ {c['provider']}"),
        c["method"].upper(),
        c["primary"],
        c["desc"],
    )


def _org_model_row(m: dict, nav: str) -> tuple:
    status = m["headline"].replace("_", " ")
    if m["headline"] != "tracked":
        status += f" ({m['status_summary']})"
    return (
        _model(nav, m["slug"], m["name"]),
        m["n_endpoints"],
        m["n_providers"],
        m["n_changes"],
        f"{m['n_changed']}/{m['n_endpoints']}",
        m["last_change"],
        status,
    )


def _provider_row(p: dict, nav: str) -> tuple:
    return (
        _provider(nav, p["slug"], p["brand"]["name"]),
        p["n_endpoints"],
        p["n_models"],
        p["n_variants"],
        p["lt_years"],
        p["lt_changes"],
        _rate(p["lt_rate"]),
        _ci(p["lt_ci"]),
        p["b3it_endpoints"],
        p["b3it_years"],
        p["b3it_changes"],
        _rate(p["b3it_rate"]),
        _ci(p["b3it_ci"]),
        p["last_change"],
    )


def _top_row(t: dict, nav: str) -> tuple:
    return (_ep(nav, t["slug"], f"{t['model']} @ {t['provider']}"), t["n"], t["last"])


def _plot_row(p: dict, nav: str) -> tuple:
    return (
        _provider(nav, p["slug"], p["brand"]["name"]),
        p["n_endpoints"],
        p["lt_years"],
        p["lt_changes"],
        _rate(p["lt_rate"]),
        _ci(p["lt_ci"]),
    )


def _spend_group_row(item: tuple, labels: dict) -> tuple:
    group, cost = item
    return (labels[group], f"${format_cost(cost)}")


def _spend_endpoint_row(ep: dict, order: list[str], tracked: set[str]) -> tuple:
    name = _ep(_BASE, ep["slug"], ep["name"]) if ep["slug"] in tracked else ep["name"]
    costs = tuple(
        f"${format_cost(ep['groups'][g])}" if g in ep["groups"] else None for g in order
    )
    return (name, *costs, f"${format_cost(ep['total'])}")


_ROW_FILTERS = {
    "dir_row": _dir_row,
    "model_ep_row": _model_ep_row,
    "model_change_row": _model_change_row,
    "variant_row": _variant_row,
    "change_row": _change_row,
    "org_model_row": _org_model_row,
    "provider_row": _provider_row,
    "top_row": _top_row,
    "plot_row": _plot_row,
    "rateable": rateable_rows,  # providers.md: the plot's rows, as home.py selects them
    "spend_group_row": _spend_group_row,
    "spend_endpoint_row": _spend_endpoint_row,
    "label": lambda g, labels: labels[g],
}


def _md_env(templates_dir: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(templates_dir / "md"),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.do"],
    )
    env.filters["cell"] = lambda v: (
        "—" if v in (None, "") else str(v).replace("|", "\\|").replace("\n", " ")
    )
    env.filters["slug"] = slugify
    env.filters["fmt_price"] = format_price
    env.filters["fmt_cost"] = format_cost
    env.filters["rate"] = _rate
    env.filters["ci"] = _ci
    env.filters.update(_ROW_FILTERS)
    return env


def _write_dir(directory: Path, suffix: str, files: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for f in directory.glob(f"*{suffix}"):
        f.unlink()
    for name, text in files.items():
        (directory / f"{name}{suffix}").write_text(text)


def render_markdown(
    website_dir: Path, built_at: datetime, pages: dict[str, list[dict]]
) -> None:
    """`pages` maps a template name ("endpoint", "index", ...) to the list of
    render contexts the HTML side used, each already carrying `machine` from
    links(). Everything else in the context is the page JSON, verbatim."""
    env = _md_env(website_dir / "templates")
    env.globals.update(
        SITE_URL=SITE_URL,
        CODE_REPO_URL=CODE_REPO_URL,
        DATA_REPO_URL=DATA_REPO_URL,
        SLUG_KEPT_CHARS=SLUG_KEPT_CHARS,
        built_at=built_at.strftime("%Y-%m-%d %H:%M UTC"),
        FEED_CAP=GLOBAL_FEED_CAP,
        PLOT_PREVIEW=PLOT_PREVIEW,
        DIR_PREVIEW=DIR_PREVIEW,
    )
    by_dir: dict[Path, dict[str, str]] = {}
    for kind, contexts in pages.items():
        template = env.get_template(f"{kind}.md.j2")
        for ctx in contexts:
            out = website_dir / ctx["machine"]["md"]
            by_dir.setdefault(out.parent, {})[out.stem] = template.render(
                {**ctx, "nav_prefix": _BASE}
            )
    for directory, files in by_dir.items():
        _write_dir(directory, ".md", files)
    (website_dir / "llms.txt").write_text(env.get_template("llms.txt.j2").render())
    print(f"Generated {sum(len(f) for f in by_dir.values())} markdown pages + llms.txt")


def _entry(item: dict) -> dict:
    return {
        "id": f"tag:trackllm.net,2026:{item['slug']}:{item['method']}:{item['iso']}",
        "title": f"{item['org']}/{item['model']} @ {item['provider']}: {item['method'].upper()} change"
        + (f" ({item['primary']})" if item["primary"] else ""),
        "updated": item["iso"],
        "desc": item["desc"],
        "url": f"{SITE_URL}/endpoints/{item['slug']}.html",
        "md_url": f"{SITE_URL}/endpoints/{item['slug']}.md",
        "json_url": f"{SITE_URL}/data/models/{item['modelSlug']}.json",
    }


def render_feeds(
    website_dir: Path,
    built_at: datetime,
    items: list[dict],
    scopes: dict[str, set[str]],
) -> None:
    """One Atom feed per scope slug plus the global one.

    `items` are changes_page["items"] (newest first). `scopes` maps a scope kind
    to every slug that has a page, so an entity with no change yet still gets an
    empty feed to subscribe to.
    """
    env = Environment(
        loader=FileSystemLoader(website_dir / "templates"),
        autoescape=select_autoescape(default=True),
    )
    template = env.get_template("feed.xml.j2")
    fallback = built_at.strftime("%Y-%m-%dT%H:%M:%SZ")

    def render(title: str, path: str, entries: list[dict]) -> str:
        return template.render(
            title=title,
            site_url=SITE_URL,
            self_url=f"{SITE_URL}/{path}",
            updated=entries[0]["iso"] if entries else fallback,
            entries=[_entry(i) for i in entries],
        )

    feeds_dir = website_dir / "feeds"
    _write_dir(
        feeds_dir,
        ".xml",
        {
            "all": render(
                "TrackLLM: all detected changes",
                "feeds/all.xml",
                items[:GLOBAL_FEED_CAP],
            )
        },
    )
    n = 1
    scoped = [(i, _scope_slugs(i)) for i in items]
    for kind in FEED_SCOPES:
        files = {}
        for slug in scopes[kind]:
            path = links(kind, slug, [])["feed"]
            files[slug] = render(
                f"TrackLLM: changes for {kind} {slug}",
                path,
                [i for i, s in scoped if s[kind] == slug],
            )
        _write_dir(feeds_dir / _PAGE_DIRS[kind], ".xml", files)
        n += len(files)
    print(f"Generated {n} Atom feeds in feeds/")


def render_sitemap(website_dir: Path, built_at: datetime, pages: list[str]) -> None:
    """sitemap.xml over every HTML page, site-root-relative paths in."""
    day = built_at.strftime("%Y-%m-%d")
    urls = "\n".join(
        f"  <url><loc>{SITE_URL}/{p}</loc><lastmod>{day}</lastmod></url>"
        for p in sorted(pages)
    )
    (website_dir / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n</urlset>\n"
    )


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
