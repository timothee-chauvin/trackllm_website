# Machine access: Markdown mirror, Atom feeds, discoverability

## Goal

Make the site as navigable for AI agents as it is for humans, and give humans a
way to be notified of changes without the site storing anything about them.

## Non-goals

- No `llms.txt`: the homepage documents machine access instead.
- No chart-to-table conversion: charts become a link to the full JSON series.
- No email, no accounts, no server. Everything is generated at build time.

## 1. Layout

Every generated `X.html` gets an `X.md` sibling: `index.md`, `changes.md`,
`spend.md`, `methodology.md`, `endpoints/*.md`, `models/*.md`, `providers/*.md`,
`orgs/*.md`.

Atom feeds under `website/feeds/`:

- `feeds/all.xml` (global, capped to the latest 200 changes)
- `feeds/endpoints/<slug>.xml`, `feeds/models/<slug>.xml`,
  `feeds/providers/<slug>.xml`, `feeds/orgs/<slug>.xml` (uncapped)

Feed and `.md` directories are pruned and rewritten each build, like the page
directories (`write_json_dir` semantics).

New module `generate_site/machine.py` renders both, from Jinja templates
`website/templates/md/*.md.j2` and `website/templates/feed.xml.j2`. Inputs are the
same page JSON the TypeScript consumes (overview, model/provider/org views,
manifests, changes, spend); Python does no new computation.

## 2. Markdown pages

Each `.md` starts with a header block:

- canonical HTML URL (`https://www.trackllm.net/...`)
- Atom feed URL for this scope (pages that have one)
- JSON URL(s) this page is rendered from
- link to the data repo (`https://github.com/timothee-chauvin/trackllm_data`)
- build timestamp (UTC)

Then the HTML page's sections, in the same order, with the same breadcrumb —
links point at `.md` siblings so an agent navigates exactly like a human. Tables
the TypeScript builds from JSON (directory, change log, per-endpoint rows,
provider rows, spend) become markdown tables. Each chart becomes one line:
`Full series: <json url>`.

## 3. Atom feeds

One entry per detected change, from `changes.json`:

- `id`: `tag:trackllm.net,2026:<slug>:<method>:<date>`
- `title`: `<model> @ <provider>: <method> change` plus magnitude display when present
- `link`: the endpoint HTML page; `content`: a short sentence plus links to the
  endpoint `.md` and JSON
- `updated`: the change date; feed `updated` = latest entry (or build time when empty)

Scoped feeds filter the same list by endpoint slug, model slug, provider slug
or org slug, using the same slug functions the pages use.

## 4. Discoverability

- `<head>` of every page: `<link rel="alternate" type="text/markdown" href=...>`
  and, where a feed exists, `<link rel="alternate" type="application/atom+xml">`.
- Every page: a "Subscribe (Atom) · Markdown · JSON" line under the breadcrumb.
  Clicking Subscribe fires `goatcounter.count({path: "subscribe/<scope>/<slug>",
  event: true})` so per-scope interest is measurable without storing anything.
- Homepage (HTML and `.md`): a "Machine access" section explaining the `.md`
  mirror, the feeds, the JSON files (`overview.json`, `changes.json`,
  `changes_page.json`, `spend.json`, `models/<slug>.json`,
  `providers/<slug>.json`, `b3it/<slug>/b3it.json`, `lt/<slug>/...`) and the data
  repo.

## 5. Testing

- pytest: render a synthetic site (existing fixtures), check every `.html` has a
  `.md`, every relative link in every `.md` resolves, every feed parses as XML
  with the Atom namespace and required elements, scoped feeds contain exactly
  the changes of their scope.
- bun: extend the existing link checker to `.md` links and `<link rel=alternate>`
  targets.
