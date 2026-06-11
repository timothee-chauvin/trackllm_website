# Design: production change detection for LT and B3IT

Goal: make the website's two methods real. LT (logprob tracking) gains scheduled,
persisted change detection. B3IT (border inputs) gains the full loop: scheduled
detection, an endpoint lifecycle (onboard / retire / resurrect), re-initialization
after detected changes, and website display.

All algorithm decisions below were validated on the real Jan 14 – Apr 7 2026
phase 2 dataset (53 endpoints; analysis in `bi/sampling_sweep.py`,
`bi/bi_quality.py`, `bi/adaptive_rule.py`, reports under `reports/`).

## Approved algorithm parameters

**Sampling**: 20 BIs per endpoint ideally, minimum 10 to monitor at all,
10 detection samples per BI per day at T=0. (~$0.36/endpoint/yr average.)

**BI selection (phase 1b)**: phase 1a discovers up to 50 BI candidates
(m=3 samples each). Phase 1b collects ~100 reference samples per candidate,
ranks candidates by top-2 balance (p2/p1 of the reference distribution), and
keeps the top 20. The reference samples double as the epoch reference.
Endpoints with fewer than 10 BIs are not monitored (`retired(no_bis)`).

**Detection rule (adaptive, LT-style)**: on the daily series of mean TV distance
vs the epoch reference. Each day's TV is compared to a trailing baseline of
`window = 14` days that excludes the most recent `exclusion = 4` days. A day
*deviates* when |TV − baseline_mean| exceeds BOTH `abs_delta = 0.2` AND
`sigma_k = 4` × baseline_std. A change fires after `persistence = 3` consecutive
deviating days, dated at the first one (the onset). `cooldown = 10` days between
events. No secondary fixed-threshold annotation.

**Instability flag**: an endpoint whose median TV over the trailing 14 days is
≥ 0.4 is flagged *unstable* (e.g. qwen3-235b on wandb sits at 0.45+ from day
two — serving drift, not a change). Detection stays active; the flag is
computed at run/build time from raw data, never persisted.

All parameters live in `config.toml` (single source of truth), recorded into
each epoch on closure for forensics.

## Core principle: facts vs derivations

State files store only **facts**: collected samples, reference samples, and
actions taken (epoch closed, re-init performed, endpoint retired). Everything
the algorithm computes — TV series, deviations, change events for display,
instability flags — is **derived** and recomputed freely. This keeps algorithm
updates from making a mess: the daily run recomputes the current epoch's TV
series from raw data and applies the rule (no persisted streak counters, fully
idempotent), and the website rebuild re-derives all displayed events with the
current algorithm. The one un-derivable fact is the re-init timing itself;
epochs record the algorithm parameters under which they were closed.

## BI epoch state

Per-endpoint state file `data_bi/state/{endpoint_slug}.json` (pydantic models
in a new `bi/state.py`):

- `endpoint`: api / model / provider
- `status`: `monitoring | retired`; when retired:
  `{reason: stalled | no_bis | delisted, since, last_recheck}`. Retirement is a
  resting state, not a tombstone — see lifecycle.
- `epochs`: list of
  `{start, border_inputs, reference, end?, end_reason?, change_date?, params?}`
  - `border_inputs`: the top-20 (by balance) BI prompts for this epoch
  - `reference`: ~100 samples per BI, collected explicitly at epoch start
  - `end_reason`: `change_detected | stalled | gap`
  - `change_date`: estimated onset (first deviating day) — what the changes
    feed shows. `end` is when re-init actually ran (the rule confirms on the
    third deviating day, so 2 days after onset, plus scheduling). Batches in
    [change_date, end] still plot against the old reference (the visible
    post-change plateau).

Daily detection batches continue to land in the existing
`data_bi/phase_2/{slug}/{YYYY-MM}.json` files, untouched in format.

## Daily BI monitor (`bi/monitor.py`, new workflow `bi-monitor.yml`, runs from main; replaces the retired `bi-phase-2.yml`)

For each `monitoring` endpoint:

1. Sample each current-epoch BI 10× at T=0 (existing phase_2 machinery).
2. Recompute the epoch TV series from raw data; apply the adaptive rule.
3. On a confirmed change: close the epoch (`change_date` = onset), run hybrid
   re-init, open a new epoch.
4. Stall handling: a run where all queries error increments a counter derived
   from the data; after 7 consecutive all-error days the endpoint becomes
   `retired(stalled)` and stops being queried. (The Jan–Jun data burned months
   of queries on dead endpoints; this is the fix.)

## Hybrid re-init (`bi/reinit.py`; also the onboarding path)

1. Re-probe each current BI 10× (generous benefit of the doubt: a true k=2 BI
   is wrongly discarded only ~0.2% of the time); keep those showing >1 distinct
   output.
2. If survivors < 20, run phase 1a discovery (budget-capped) to top candidates
   back up to 50, then phase 1b: ~100 reference samples per candidate, rank by
   balance, keep top 20.
3. Fewer than 10 BIs total → `retired(no_bis)`; the change event that triggered
   the re-init is still recorded.

## Endpoint lifecycle (extends `update_endpoints.py`, daily)

- `update_endpoints_bi()` (existing) keeps vetting the candidate list
  (`endpoints_bi.yaml` + `bad_endpoints_bi.yaml`).
- New: candidates with no state file are onboarded via the re-init path
  (phase 1a → 1b → epoch 0), capped per run to bound phase-1 cost.
- Endpoints gone from the OpenRouter catalog → `retired(delisted)`.
- Retired endpoints are re-checked every 14 days: a stalled endpoint that
  responds again, a delisted one back in the catalog, or a no_bis one that now
  yields ≥10 BIs is resurrected through onboarding → new epoch. Old epochs are
  kept; history survives resurrection.
- `endpoints_bi_phase_1.yaml` is retired; the monitored set is exactly the
  state files with `status: monitoring`.

## LT automation

- `run-main.yml` gains a scoring step after each hourly collection:
  `lt_scores` per endpoint written to `website/data/{slug}/lt_scores.json`.
- New append-only `website/data/lt_changes.json`: events
  `{endpoint, change_date, sigma, first_detected}`. A recomputed change point
  within `PEAK_DISTANCE` indices of an existing event is the same event, so
  "detected on" dates stay stable as data accumulates. The detection algorithm
  itself is unchanged.

## Website

- Endpoint pages for the union of LT-tracked and BI-monitored endpoints; each
  page shows the sections it has data for.
- New BI section per endpoint: TV-over-time plot with epoch boundaries and
  change markers, current BI count, status, and the instability badge.
- New changes feed page merging LT events and BI epoch-boundary events (date,
  endpoint, method, magnitude), newest first; recent-change badge on the index.
- All BI display values are derived at build time with the current algorithm.

## Migration & resumption

- Existing phase 2 history: synthesize epoch 0 per endpoint with reference =
  first batch, `end_reason: gap` closed at the endpoint's last successful day
  (mass die-off on Apr 7 was the feature branch's `require_parameters` bug,
  fixed on main in 9615d687e0; 23 endpoints are recoverable, 29 truly
  delisted).
- Resumption treats every live endpoint as fresh onboarding (the January
  references are stale after a two-month gap): new phase 1b → epoch 1.
  Historical epochs remain for display.
- Config updates: `[bi.phase_1] target_border_inputs` 30 → 50;
  `[bi.phase_2] queries_per_token` 20 → 10; new `[bi.detection]` section with
  the approved rule parameters; new keys for reference samples (100), top-k
  (20), minimum BIs (10), stall threshold (7), recheck interval (14d),
  instability threshold (0.4).

## Testing

Tests first, with fixtures extracted from the real phase 2 dataset (per
project practice). Key cases: the adaptive detector on real series (deepseek
family swaps, the gemma-3-27b phala flapper, qwen3-235b wandb instability —
must flag, not fire), hybrid re-init decisions with mocked sampling, LT event
dedup/stability, epoch state round-trip, migration from existing data.

## Rollout order

1. `bi/state.py` + migration of existing data (validated against history).
2. `bi/detection.py` (adaptive rule extracted from `bi/adaptive_rule.py`) +
   `bi/monitor.py`; enable `bi-monitor.yml` daily from main.
3. Re-init + lifecycle in `update_endpoints.py`.
4. LT scoring step + events file.
5. Website: BI plots, changes feed, instability badge.
