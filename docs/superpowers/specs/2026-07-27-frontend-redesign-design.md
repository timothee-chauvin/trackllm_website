# TrackLLM frontend redesign — design

**Status:** draft for review · **Date:** 2026-07-27

## 1. Context & goals

The current site is a static Jinja + TypeScript build: a monospace GitHub-light
theme, one flat 200+ row endpoint table, isolated per-endpoint pages, no dark
mode, no cross-linking. It undersells genuinely rich data (logprob time series,
changepoints, B3IT total-variation series, a cross-method change feed, spend).

Audience is **two-sided**: a *public showcase* (accompanying the blog post /
paper) and a *researcher tool*. Concretely the redesign must fix:

- **Aesthetics** — dated, no identity, no dark mode.
- **Navigation** — a flat 414-row table doesn't scale; no way to find, group, or filter.
- **Provider-level patterns** — no view of which providers drift most.
- **Cross-provider comparison** — can't see the same model across its providers.
- **Cross-linking** — every page is an island.

Voice: use the language of the blog post and papers, not invented phrasing —
"LLM APIs are opaque black boxes, even for open-weight models", "undisclosed
changes", "logprob tracking (LT)", "black-box border input tracking (B3IT)",
"border inputs" = "inputs for which sampling at T=0 doesn't always give the same
output", "small changes are fine, but they should be disclosed". Never claim a
*cause* (quantization, serving swap) — we detect that outputs moved, not why.

## 2. Approach

Keep the **static-generation stack** (Jinja for HTML, TypeScript bundled by Bun,
inline SVG / Plotly for charts). The problem is design + information architecture,
not the stack; a SPA rebuild would lose the zero-infra static deploy for no gain.
Three levers:

1. A real **design system** (tokens, dark mode, type scale, reusable components).
2. **Information architecture** — a global nav plus new aggregate pages so 414
   endpoints become navigable.
3. A single client-side **`endpoints.json` index** powering search / filter / sort
   and cross-linking, with no backend.

One backend addition is required (Section 4): a **drift-from-reference series for
LT**, so the whole site can present LT and B3IT through one unified lens.

**Ship an MVP first**, not all pages at once (§10): the design system, the LT drift
signal, and the three core pages — Overview, Endpoint, Model — plus a restyled
Spend. Provider / Changes / Rankings pages come later; a Methodology page is
dropped in favour of linking the blog post.

## 3. Design system

- **Identity:** a drift/telemetry "observatory" — a precise instrument watching for
  silent movement. Cool near-black ground (`#0B0F14`, slight blue bias), a signal
  **cyan** accent (`#37C2E0`, oscilloscope-like). Full light theme via tokens.
- **Semantic state colors, separate from the accent:** stable green, changed amber,
  retired grey; B3IT gets its own purple so the two methods are always visually
  distinct from the accent and from each other.
- **Typography:** system sans for prose (no webfont-CDN dependency), monospace
  **only** for data / tokens / numerals with `tabular-nums` — the inverse of
  today's all-mono. Fixed type scale, `text-wrap: balance` on headings.
- **Theming:** token-level `:root` custom properties, redefined under
  `prefers-color-scheme` and under `data-theme` overrides, so both themes get equal
  care and the viewer's toggle wins.
- **Reusable components:** nav, breadcrumb, method badge (LT / B3IT), status pill
  (stable / changed / retired), the drift sparkline, the change feed row — defined
  once, used on every page.

## 4. The unified drift signal (key decision)

Both methods are re-expressed as **drift from a reference period** — distance of
current behaviour from what the endpoint returned in its first ~2 weeks. Reads 0
when the endpoint matches its baseline; rises and stays elevated after a change.

- **LT drift (new):** mean absolute difference between the current day's per-token
  mean log-probabilities and the reference period's, over the union of returned
  tokens (missing tokens left-censored), in **nats**. This is the LT analogue of
  B3IT's total variation.
- **B3IT drift (exists):** mean total-variation distance of the border-input output
  distribution vs the reference epoch (`epoch_tv_series`), 0–1.

**Why this replaces what the mocks first used:** the earlier display used the
rolling two-window LT test statistic (visually "smeared") and then predictive
entropy (a lossy scalar that *missed* real changes — e.g. the 2025-09-18 change on
deepseek-chat-v3-0324@fireworks showed nothing in entropy). Drift-from-reference
rises visibly at every detected change, so annotated changepoints land on real
movement, and it puts LT and B3IT on one conceptual footing.

### 4.1 Backend change — LT drift-from-reference

Add a **display-only** drift series to the LT build. It must not alter detection:
change detection continues to run on the existing two-sample statistic in
`lt_scores.py`; drift is a separate, more legible view for the site.

**Where it lives — mirror B3IT.** B3IT keeps its display series (`tv_series`) in
the *same* per-endpoint file as its changes (`b3it.json`), derived together. LT
does the analogous thing: `lt_scores.py` already loads the logprobs and builds the
per-token tensor to compute the two-sample statistic, so it computes drift from
that same data in the same pass, and writes the daily `(date, drift)` series
**into `lt_scores.json`** alongside the existing `dates`/`scores`/`changes`. No
companion file, no second data read; the frontend already fetches `lt_scores.json`.

- A pure helper (no I/O, `bi/detection.py` style), given the prompt's
  `(date, {token: logprob})` series:
  - reference = observations within the first `REFERENCE_DAYS` (≈14) of the series;
  - clip logprobs to a floor (≈ −30) to neutralise provider `-inf`/sentinel values;
  - per day, aggregate a mean log-probability vector; drift = mean over the union
    of tokens of `|day_mean − reference_mean|` with left-censoring of missing tokens;
  - short rolling-median smoothing (window ≈5) to suppress single-day sparse-data spikes.
- Each detected change carries a **drift level reached** (max drift in a short
  post-change window) as its magnitude, distinct from **σ** (confidence). Both are
  shown: σ can be high while drift is small (a confident but behaviourally tiny
  change).

### 4.2 Unification

Once LT has a drift series, the site presents both methods identically:

- Endpoint page: two stacked lanes (LT drift nats / B3IT TV 0–1) on one shared time
  axis, changepoints annotated (σ + drift for LT, peak TV for B3IT).
- Main directory + change feed: one drift sparkline per endpoint on a **fixed
  scale** (flat-low = stable, step up = change).
- Model page: every provider's drift on a shared timeline, **shared y-scale within
  each method** so magnitudes are comparable.
- Change feed magnitude column: drift level (LT) / peak TV (B3IT); σ shown as a
  separate confidence value for LT.

Historical caveat (already handled for B3IT in PR #26): derive drift/changes over
**full epoch history** with the production **top-k ranked BIs**, not the current
open epoch only — otherwise pre-detector history (incl. the paper-period changes)
stays invisible. The LT drift series should likewise cover full history.

## 5. Information architecture

The Overview page's searchable directory is the discovery surface — no separate
Models/Providers index pages are needed for the MVP. Model and endpoint pages are
reached by clicking through the directory and the change feed.

Persistent top nav (MVP): `Overview · Spend`, plus external links (Methodology →
blog post, GitHub, Paper). Deferred nav items (`Providers`, `Changes`, later
`Rankings`) are added as those pages land.

Breadcrumbs on detail pages: `Home / <org> / <model> / @ <provider>`. A shared
"endpoint chip" links consistently everywhere.

## 6. Page specifications

### 6.1 Overview (main page)
- **Hero:** headline **"LLM API outputs are unstable over time."**; blog lede
  (opaque black boxes → continuously monitor for undisclosed changes → LT where
  logprobs are exposed, B3IT where they aren't → small changes are fine but should
  be disclosed).
- **Telemetry row:** endpoints (active / retired), models, providers, queries
  logged, changes detected (LT / B3IT), with a "cheap enough to run continuously"
  spend caption ($X for N queries).
- **Latest detected changes** (centerpiece): rich feed — model @ provider, method
  badge, drift sparkline around the changepoint, drift magnitude + σ, plain
  magnitude-only description (no cause).
- **Provider drift rate:** detected changes per **endpoint-year of monitoring**
  (normalised for fleet size + monitoring length), including providers with zero
  changes; a square grid (1 square = one endpoint-month) encodes monitoring volume
  = confidence in the rate; low-data rows visibly de-emphasised.
- **Endpoint directory:** client-side search + filter chips (LT / B3IT / ever
  changed / recent / retired) + sortable columns (model, provider, status, changes,
  stable-for) + fixed-scale drift sparkline. Replaces the flat table; reconciles the
  headline change count (Σ per-endpoint changes).

### 6.2 Endpoint page
- Header + **status card** (status, monitored range, change count LT / B3IT).
- **Compare-across-providers banner** → the model page.
- **Raw-signal panel:** stacked LT-drift / B3IT-TV lanes, shared time axis,
  annotated changepoints.
- **Detected changes** table (date, method, drift reached, σ).
- Inline **methodology** cards (LT primary / B3IT fallback + corroboration).

### 6.3 Model page (cross-provider — the comparison home)
- Header + summary (N providers, N with changes, monitored range) and the framing
  that a fixed model version still drifts because the *serving* moves.
- **Drift by provider:** every provider serving the model on one shared timeline,
  drift line + change dots, **shared per-method y-scale**, per-provider peak + change
  count, each row linking to its endpoint page.

### 6.4 Spend (MVP)
Restyle the existing spend page into the design system — no new data.

### 6.5 Deferred (post-MVP)
Additive; the MVP's data + design system already support them:
- **Provider page:** all endpoints from a provider, its normalised drift rate, a
  timeline of its changes.
- **Changes:** the global change feed / timeline, filterable — a research log.
- **Rankings:** most / least stable endpoints and providers.
- A **Methodology page is dropped** for now — link the blog post instead.

## 7. Method priority (LT over B3IT)

LT is the better signal and takes precedence: authoritative wherever logprobs
exist; B3IT is coverage for opaque endpoints and one-way corroboration where both
run (LT+B3IT agree > LT only > B3IT only). Badge and method ordering put LT first.

## 8. Build / data changes summary

- **New:** LT drift-from-reference series (Section 4.1); client-side `endpoints.json`
  index; OG/meta images per page for shareability.
- **Already merged (#26):** B3IT changes derived over full history with top-20 BIs,
  surfaced to the feed.
- **Unchanged:** detection algorithms and parameters (`abs_delta` etc.); the live
  monitor. Drift is display-only.

## 9. Out of scope / follow-ups

- `abs_delta` fleet-mean sensitivity — deliberately untouched.
- Empty-string border input reachable via `parse_phase_1_results` — separate
  phase-1 hygiene fix.
- Short-monitored endpoints produce noisy drift (few days ≈ reference window);
  smoothing helps, but the directory should flag "insufficient data" rather than
  imply a large change.

## 10. Scope — MVP vs later

**MVP (this design):**
1. LT drift-from-reference in `lt_scores.py` → `lt_scores.json` (§4.1).
2. Design system (tokens, dark/light, shared components) + minimal nav.
3. Overview page — hero, telemetry, change feed, provider drift rate, searchable directory.
4. Endpoint page + Model page + client-side `endpoints.json` index.
5. Spend page restyled into the system.

**Later (additive):** Provider page, global Changes timeline, Rankings, OG/meta
images. Methodology is dropped — link the blog post.

Suggested build order within the MVP: (1) LT drift backend as its own PR, then
(2) design system + Overview, then (3) Endpoint + Model + index, then (4) Spend.

## 11. Resolved decisions

- **LT drift location:** into `lt_scores.json` alongside `scores`/`changes`,
  computed in `lt_scores.py` — mirroring B3IT's `tv_series` inside `b3it.json` (§4.1).
- **Drift sparkline scale — fixed cap, not percentile.** The directory sparklines
  share one y-axis so a stable endpoint reads flat-low and a change reads as a step
  up; the "cap" is the top of that axis. Two ways to set it:
  - *Fixed constant* (**chosen**): a hardcoded cap, e.g. 1.5 nats. Nats are a real
    unit, so this is meaningful, and — critically — stable across rebuilds: the same
    endpoint always renders identically, and a line pinned to the top consistently
    means "large drift." Bigger drifts clamp, which reads correctly as off-the-scale.
  - *Global percentile* (rejected): cap = e.g. the 95th percentile of all endpoints'
    peak drift, recomputed each build. It uses the full height, but the cap moves
    between rebuilds, so an endpoint's sparkline silently rescales over time and a few
    new big drifters flatten everyone else — comparability-over-time loses to that.
  - The Model page's per-provider strips likewise use a fixed shared scale within
    each method.
