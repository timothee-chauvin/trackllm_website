# Every OpenRouter endpoint on the site, with a status

Approved 2026-07-30 (sections 1-4 in chat; spec review waived by user).

## Goal

Every OpenRouter endpoint appears on the website with an explanation of its
tracking state: tracked, retired, bad temperature, too expensive, errors out,
etc. Someone searching "gpt-5" or "fable" must find pages explaining *why*
those models can't be tracked (no temperature parameter, no logprobs —
presumably anti-distillation).

## Coverage

- The full live catalog (~1,066 endpoints incl. free ones) — free endpoints get
  a `free_excluded` status rather than being absent.
- Plus historical endpoints **we tracked** that have since left the catalog.
  Endpoints that existed but were never tracked disappear with the catalog.

## Status taxonomy

Per-method (LT and BI are independent; e.g. grok-4.5 is LT-tracked and
BI-too-expensive), plus one derived headline.

LT: `tracked` | `stalled` | `no_logprobs` | `probe_failed` | `too_expensive`
(> config.api.max_cost_mtok) | `free_excluded`.

BI: `monitoring` | `retired:<no_bis|unreachable|delisted>` | `bad_temperature`
| `too_expensive` | `liar` | `not_selected` (vetted good, policy didn't pick)
| `excluded` (policy globs) | `pending` (never successfully vetted, no reject
bucket) | `free_excluded`.

Headline (first match wins): tracked → retired → untrackable → too expensive →
not selected → errors out (liar/probe_failed/unreachable) → pending → free.
`untrackable` = no temperature AND no logprobs: neither method can ever work.

Copy: one fixed sentence per status; `bad_temperature` reads along the lines of
"this API rejects or ignores the temperature parameter, so T=0 sampling is
impossible — presumably to prevent distillation". Per-endpoint detail is
appended where recorded (e.g. the exact probe error).

## Data flow (approach A: build-time resolver over committed snapshots)

`make build` stays offline; the daily pipeline persists what it already
fetches.

1. `update_endpoints` writes `endpoints_catalog.yaml`: one entry per catalog
   endpoint — model, provider, pricing, temperature/logprobs claims
   (endpoint-level `supported_parameters`), created, free flag.
2. `update_endpoints_lt` writes `endpoints_cache_lt.yaml`: failed logprob
   probes with reason (error message, or "returned N logprobs, expected M").
   Today those failures vanish, making "errors out" underivable at build time.
3. New pure module `generate_site/status.py`: inputs are only committed files
   (catalog, both caches, `endpoints_lt.yaml`, `endpoints_bi.yaml`, BI state
   files, `bi_selection.toml`); output is
   `{slug: (lt_status, bi_status, headline, reasons)}` over the union of
   catalog + previously-tracked endpoints.
4. `tracked.py` keeps deciding who gets charts; `status.py` decides what every
   page says. Statuses are stamped into the existing fleet/model/endpoint
   JSONs.

Accepted churn: `endpoints_catalog.yaml` rewrites daily like
`endpoints_bi.yaml`.

## UI

- Overview fleet: every endpoint is a row. Default chip shows tracked;
  chips for untrackable / too expensive / retired / errors / pending / free
  reveal the rest. Untracked rows show a status badge + one-line reason in
  place of the drift trace.
- Search spans all rows, matches model names AND provider names, and
  highlights the matched substring.
- Endpoint pages for all endpoints: untracked ones render a status card (both
  per-method statuses + reasons, pricing, parameter claims) and no chart.
  Tracked pages gain the two per-method status lines.
- Model pages for all catalog models; org and provider pages list untrackable
  models with badges (the anthropic org page shows fable-5/opus-5).

This supersedes the earlier "no data means omit entirely" site rule for
catalog endpoints: an explained absence is content; there are still no empty
chart placeholders.

## Testing

1. Unit tests on `status.py`: fixtures covering every status, headline
   priority, the grok case, a historical-only endpoint.
2. Writer tests: `endpoints_catalog.yaml` round-trip; probe failure recorded
   with its reason.
3. Site smoke tests (bun): untracked endpoint page renders card and no chart;
   fleet JSON carries statuses; search matches "gpt-5" and highlights.
4. Visual verification via the headless-chromium recipe before shipping.
