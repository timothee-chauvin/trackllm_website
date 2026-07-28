# Aggregate pages: Provider, Changes, and cross-linking — design

**Status:** approved · **Date:** 2026-07-28 ·
**Extends:** [2026-07-27-frontend-redesign-design.md](2026-07-27-frontend-redesign-design.md) §10 "Later"

## 1. Context

The frontend redesign shipped its MVP (#36, #37): design system, LT drift signal,
Overview, Endpoint and Model pages, restyled Spend. It deferred three items:
a **Provider page**, a global **Changes** timeline, and **Rankings**.

This design covers those three, plus the cross-linking that makes them worth
having. Mockups on real built data were reviewed before writing it.

Three scope decisions came out of that review and change what gets built:

- **Rankings is not a page.** The Overview directory already sorts by change
  count and stable-for; a leaderboard page would restate ranked data. Rankings
  become two ranked boards inside the Overview's provider section, and a
  most-changed-endpoints board on Changes.
- **There is no Providers index page.** The Overview's "Provider drift rate"
  panel and a Providers index would show the same table twice. The panel is
  **replaced** by the fuller provider section; per-provider pages hang off it.
- **The Model page already exists** and already does what a cross-provider
  comparison needs. It gets five targeted edits (§6.4), not a rebuild.

## 2. Goals

1. Answer "which providers move, and how confidently do we know that" without
   letting short-monitored providers produce nonsense rates.
2. Give every detected change a permanent, filterable home.
3. Make the site navigable: from any page, reach the model, the provider, and
   the endpoint in one click.

Non-goal: any change to detection. Everything here is display, derived from
already-generated build outputs.

## 3. The provider unit: company, with variants inside

OpenRouter identifies endpoints by a variant-level provider string —
`chutes`, `chutes/fp8`, `mancer/fp4`. Today the Overview panel treats each
variant as its own row, giving 153 rows for 74 companies.

**A provider is the part before the `/`.** Variants are grouped inside its page
and compared there. Rationale:

- It matches the question being asked ("does this provider drift?").
- Evidence stops being split across near-duplicate rows, so more providers clear
  the monitoring threshold.
- It makes the quantization comparison possible at all. In the current build,
  `chutes/fp8` drifts at 8.87 changes/endpoint-year against 1.11 for
  `chutes/bf16` — invisible when the variants live on separate pages.

The variant string is still shown everywhere an endpoint is named; only
aggregation and page identity move to the company level.

## 4. Normalised drift rate

**rate = detected changes / endpoint-years of monitoring.** An endpoint-year is
one endpoint watched for one year; a provider's exposure is the sum over its
endpoints of the span from first to last observation. This is the existing
`PROVIDER_MIN_ENDPOINT_YEARS` idea from `overview.py`, kept, with three
corrections.

### 4.1 A rate needs exposure — `MIN_ENDPOINT_YEARS = 0.5`

Below half an endpoint-year we publish **no rate**, on any surface. The row still
appears, still links to its page, and shows accumulated exposure instead. One
change in three weeks of monitoring computes to ~17/year, which is not a
measurement. In the current build this withholds a rate from ~40 of 74
providers.

### 4.2 Every rate carries a 95% Poisson interval

Counts over unequal exposure are not comparable as point estimates. For k
changes over T endpoint-years:

- `k > 0`: `[(k − 1.96√k)/T, (k + 1.96√k)/T]`, lower bound clamped at 0.
- `k = 0`: `[0, 3/T]` — the rule of three. Zero changes is evidence of a rate
  *below* 3/T, not evidence of zero.

The rule-of-three case is what makes the "nothing detected yet" board a real
claim: `openai`, at 0 changes over 10.6 endpoint-years, gets "true rate
< 0.28/year", and ranking that board by exposure ranks it by how tight the
ceiling is.

The bar renders the point estimate with the interval as a band behind it.

### 4.3 LT and B3IT rates stay separate

Two rates side by side, each with its own exposure. Never pooled: the methods
have different sensitivity and very different monitoring lengths (LT since
Jun 2025, B3IT since Jan 2026), so a pooled rate would move when B3IT coverage
grows rather than when behaviour changes.

Today this means **no provider has a publishable B3IT rate** — median B3IT
`tv_series` length is 5 points, and the largest provider exposure is 0.64
endpoint-years. The B3IT card therefore shows accumulated exposure and the
changes seen so far, and starts showing a rate when it crosses the threshold.
This is the honest state of the data, not a placeholder.

## 5. Build and data changes

All new data is derived from existing build outputs — `lt_scores.json`'s
`drift`/`drift_dates`, B3IT views, `changes.json`. Nothing reads raw logprobs.

### 5.1 New module: `generate_site/rates.py`

Pure, no I/O. `poisson_interval(k, exposure)`, `drift_rate(k, exposure)`
returning `None` under the threshold, and the `MIN_ENDPOINT_YEARS` constant.
Single source of truth for the gate, shared by the provider views and the
Overview section.

### 5.2 New module: `generate_site/feed.py` (refactor)

`overview.py` currently owns the change-feed enrichment — `_nearest_index`,
`_feed_window`, `_build_lt_feed_item`, `_build_b3it_feed_item` — and applies it
to the latest 10 changes only. The Changes page needs the same enrichment for
all of them.

Move that logic into `feed.py`, exposing
`build_feed_items(changes, lt_data, b3it_views, now)` over an arbitrary change
list. `overview.py` calls it for its slice; the changes page calls it for
everything. No duplicated windowing.

One deliberate behaviour change comes with the move: B3IT feed items are built
from `changes.json` — the canonical merged list, which includes live epoch
closures — instead of from `B3ITView.changes`, which carries TV onsets only. The
Overview's latest-changes feed and the Changes log therefore cannot disagree
about which changes exist. Feed items also gain the link slugs (`slug`,
`modelSlug`, `providerSlug`) that the cross-linking in §6.5 needs.

### 5.3 New module: `generate_site/provider.py`

`build_provider_views(website_dir, lt_endpoints, b3it_views)` →
`{provider_slug: view}`, each view carrying:

- `name`, `n_endpoints`, `n_models`, `n_variants`, `first`, `last`
- `lt` / `b3it`: `{endpoints, years, changes, rate, ci}` (rate/ci `None` under threshold)
- `variants`: per variant, the same counts plus its own rate and interval
- `monitoring`: per variant, a monthly count of endpoints under monitoring —
  the exposure the rate divides by, rendered as the timeline's grey area
- `changes`: enriched feed items for this provider
- `endpoints`: directory rows (reusing the Overview row shape)

Written to `data/providers/<slug>.json`, page at `providers/<slug>.html`.
Slug via `slugify` per CLAUDE.md.

### 5.4 `overview.py` — provider section replaces the drift-rate panel

`overview.py` stops computing `provider_stats` itself and instead takes the
base-provider rollup from `provider.py`, emitting for each: name, slug, endpoint
and model counts, variant count, LT rate + interval + exposure, B3IT exposure,
and last change date. The old per-variant `providers` array
(`{name, endpoint_years, months, n_changes, rate, conf}`) is replaced; the `conf`
heuristic (`PROVIDER_CONF_FLOOR`, `PROVIDER_CONF_FULL_YEARS`) goes away, its job
now done by the interval and the exposure gate.

Overview keeps its three sections in order: change feed, providers, endpoint
directory. Directory rows gain links (§6.5).

### 5.5 New page data: `data/changes_page.json`

Enriched feed for every change, plus the per-month LT/B3IT histogram counts and
the most-changed-endpoint board. Built by `generate_site/changes_page.py` from
`changes.json` + `feed.py`. `changes.py` (the merge) is untouched.

### 5.6 Frontend entrypoints

New `website/src/provider.ts` and `website/src/changes.ts`, added to the bun
`build`/`watch` scripts in `package.json`. New templates `provider.html.j2`,
`changes.html.j2`. Shared rendering helpers (sparkline, rate bar, volume grid,
status pill, method badge) that both new pages and `overview.ts` need move into
`website/src/components.ts` rather than being copied.

Nav in `base.html.j2` becomes `Overview · Changes · Spend` + external links. No
Providers nav item — provider pages are reached by clicking through.

## 6. Page specifications

### 6.1 Overview — providers section (replaces "Provider drift rate")

- Two ranked boards, side by side: **Most drift-prone** (LT rate, descending)
  and **Nothing detected yet** (zero changes, ranked by exposure), 5 rows each,
  bar + interval band.
- Sortable, searchable table of all providers: name (+ variant and model counts),
  endpoints, LT rate bar, monitoring volume grid (1 square = one endpoint-month),
  B3IT exposure, last change. Scroll-capped.
- Filter chips: has changes / rateable / runs B3IT. Providers under the threshold
  sort to the bottom showing "not enough monitoring"; the table footer counts
  them. No separate section for them.

### 6.2 Provider page (`providers/<slug>.html`)

Breadcrumb `Home / <provider>`. Header with endpoint, model and variant counts
and the monitored range; when one variant's rate exceeds another's by >1.8×, the
lede names both, since that contrast is the page's most useful finding.

1. **Summary row** — endpoints, still active, changes detected, endpoints affected.
2. **Drift rate** — two cards, LT and B3IT (§4.3), each with rate, 95% interval,
   and exposure, or the "not enough monitoring" state.
3. **Monitoring & changes over time** — one lane per variant on a shared axis:
   grey area = endpoints under monitoring that month, dots = detected changes
   sized by magnitude (LT drift in nats, B3IT peak TV). Reading the rate's
   numerator and denominator off one picture is the point.
4. **Serving variants** — per-variant endpoints, rate, exposure, changes, B3IT.
5. **Endpoints** — searchable directory of the provider's endpoints with drift
   sparklines, model names linking to model pages.

### 6.3 Changes page (`changes.html`)

Breadcrumb `Home / Changes`. Header + summary (changes, endpoints affected,
providers involved, last 30 days, largest LT drift).

1. **Timeline** — stacked LT/B3IT bars per month; clicking a month filters the
   log; clicking again clears.
2. **Most-changed endpoints** — 5 rows, the endpoint-level ranking.
3. **Log** — every change, grouped by month with sticky headers, each row: date +
   relative age, model @ provider (both linked), drift sparkline around the
   changepoint, magnitude, and σ as separate confidence for LT. Search plus
   chips: LT / B3IT / last 90 days / large drift.

### 6.4 Model page — five edits to the shipped page

`models/<slug>.html` already puts every provider of a model on one shared
timeline with drift lines, change dots, per-provider change counts and a shared
per-method y-scale. Changes:

1. Change dots move from mid-strip (`cy = H/2`) to **the drift level reached**,
   so dot height reads on the same axis as the line.
2. An **all-providers strip** above the rows: every change for the model on the
   shared axis, so "when did this model move" is one glance.
3. Rows **grouped by provider company**, group header linking to the provider page.
4. Provider names link to the provider page.
5. The meta column shows **date of last change** when there is one, peak drift
   otherwise.

Edits 2 and 3 need `model.py` to emit the model's flattened change list and each
endpoint's base provider; the rest are `model.ts` only.

### 6.5 Cross-linking

The navigation fix the aggregate pages exist to support:

| From | To | Where |
| --- | --- | --- |
| Overview directory | model page | model name |
| Overview directory | provider page | provider name |
| Overview / Changes feed | model, provider | model name, `@ provider` |
| Endpoint page | provider page | new link beside the existing model banner |
| Model page | provider page | provider name + group header |
| Provider page | model page, endpoint pages | model name in the directory |

The endpoint page already links to its model page (breadcrumb + compare banner);
only the provider link is new.

## 7. Testing

TDD, tests first, `uv run pytest`, following existing
`tests/test_generate_site_*.py` conventions (build a fixture tree, run the
builder, assert on the emitted JSON).

- `test_generate_site_rates.py` — `poisson_interval` for k=0 (rule of three),
  k>0 symmetry, clamping at 0; `drift_rate` returns `None` below the threshold
  and at the boundary exactly.
- `test_generate_site_provider.py` — variant grouping from provider strings with
  and without `/`; exposure summed across endpoints; LT and B3IT kept separate;
  a provider with a sub-threshold exposure emits a `None` rate; monthly
  monitoring counts match endpoint spans; provider slug round-trips.
- `test_generate_site_feed.py` — extraction refactor preserves current output:
  the enrichment applied to the latest changes equals what `overview.py`
  produced before.
- `test_generate_site_changes_page.py` — every change appears once; month
  histogram totals equal the change count; most-changed board ordering.
- `test_generate_site_overview.py` (extend) — `providers` rows are
  base-provider, carry intervals, and sub-threshold providers appear with no rate.
- `test_generate_site_model.py` (extend) — flattened change list and `base` field.
- `test_generate_site_render.py` (extend) — provider pages and `changes.html`
  are written; counts match.

`bunx tsc --noEmit` must stay clean; `prek run --all-files` after each change.

## 8. Out of scope

- Detection algorithms and parameters — untouched, as in the MVP.
- OG/meta images (still deferred from the redesign's "Later" list).
- A Methodology page — still dropped in favour of linking the blog post.
- Changing how B3IT accumulates `tv_series` history; the short series is why
  B3IT rates are withheld, and that resolves with monitoring time, not display.

## 9. Build order

1. `rates.py` + tests (pure, no dependencies).
2. `tests/conftest.py` — hoist the fixture writers duplicated across test files.
3. `feed.py` extraction + tests.
4. `provider.py` + tests → provider page data.
5. `overview.py` provider section swap + tests.
6. `changes_page.py` + tests.
7. `render.py` + the two new templates, nav, endpoint provider link.
8. Frontend: `components.ts` + `provider.ts`.
9. Frontend: `changes.ts` + the Overview's provider section.
10. Model page edits.

Detailed plan: [../plans/2026-07-28-aggregate-pages.md](../plans/2026-07-28-aggregate-pages.md).
