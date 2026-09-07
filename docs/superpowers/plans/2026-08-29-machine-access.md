# Machine access — implementation plan

Spec: `docs/superpowers/specs/2026-08-29-machine-access-design.md`.

1. `generate_site/machine.py`: `links(kind, slug)` (md/feed/json paths for one
   page, shared by HTML and md renderers), `render_markdown(...)`,
   `render_feeds(...)`. Feeds from `changes_page["items"]` (already carries
   every slug a scope needs). Own Jinja env over `templates/md/` (no autoescape)
   and the Atom template (autoescape on).
2. Templates: `templates/md/{_base,index,changes,spend,methodology,endpoint,
   model,provider,org}.md.j2`, `templates/feed.xml.j2`.
3. `base.html.j2`: `<link rel=alternate>` for md + feed; footer "Machine access"
   line with `data-goatcounter-click`. `index.html.j2`: Machine access section.
4. `render.py`: pass `machine=links(...)` to every page render; call
   `render_markdown` / `render_feeds` at the end; prune `feeds/`.
5. Tests: `tests/test_generate_site_machine.py` (md per html, links resolve,
   feeds parse, scopes exact); `smoke.test.ts` dead-link check covers
   `link[rel=alternate]` and md siblings. `make clean` prunes new outputs.
