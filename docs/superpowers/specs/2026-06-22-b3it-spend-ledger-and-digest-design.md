# Design: actual-spend ledger and B3IT daily digest emails

Goal: get a daily email (B3IT only, not LT) after the daily onboarding that says,
per endpoint, which were successfully onboarded (and how many BIs), which stopped
after the 3h deadline and resume next day, which didn't get enough BIs, and which
had a detected change and were re-onboarded — with actual dollars spent on each and
in total, split into onboarding spend and monitoring spend, both for the day and
cumulative.

Getting there cleanly requires two things the codebase doesn't have yet: a record of
*actual* spend (today only projections exist, and the per-query cost is computed then
discarded), and a structured record of onboarding outcomes (today only logged). It
also surfaces a pre-existing inconsistency — LT data lives under `website/data` while
B3IT data lives in a separate `data_bi/` root — that a universal spend ledger would
otherwise be forced to take sides on.

So the work splits into three sequenced PRs, each independently shippable:

1. **Layout unification** — move B3IT's data root under `website/data`, so both
   methods share one root and the ledger has a consistent home.
2. **Universal spend ledger** — capture actual `$` for every request (LT and B3IT)
   into an append-only per-endpoint ledger.
3. **B3IT daily digest emails** — two in-process emails (one per daily B3IT
   workflow) built from run outcomes + the ledger.

## Current state (as of commit 438363400f)

- **Onboarding** is `update_endpoints.py::update_endpoints_bi_lifecycle()`, run by
  `update-endpoints.yml` (daily 00:53 UTC). Per endpoint it calls `reinit()` under a
  `config.bi.reinit.onboard_timeout_seconds` (3h) `asyncio.wait_for`. Outcomes:
  success (`result.epoch` set → `monitoring`, `len(border_inputs)` BIs); not enough
  BIs (`result.epoch is None`, `reason=no_bis` → `retired(no_bis)`); 3h timeout
  (`asyncio.TimeoutError`, logged "will resume next run", **no state saved** → retried
  next run); `bad_temperature` (cached, skipped). Rechecks of retired endpoints share
  this path (`old_bis=[]`).
- **Change-detection re-onboarding** is `bi/monitor.py::monitor()`, run by
  `bi-monitor.yml` (daily 21:01 UTC) — a *separate* workflow ~4h before the onboarding
  run in the same nightly cycle. On a confirmed change it closes the epoch and calls
  the same `reinit()` (`old_bis=epoch.border_inputs`, so survivors are re-probed
  first). It also retires stalled endpoints.
- **`reinit()` is the single onboarding primitive** for fresh onboards, rechecks, and
  change re-onboards — they differ only by `old_bis` and which workflow invokes them.
- **Cost**: every `Response` carries an actual `.cost` (`api.py:128`, via
  `compute_cost`), but `sample_prompts` returns only `(samples, n_errors)` and
  discards it. `phase_1a` discovery makes its own `OpenRouterClient` and also discards
  cost. `bi/costs.py` computes only *projected* monthly cost
  (`cost_per_request × cadence`), never actual spend. LT's `main.py` aggregates
  `response.cost` per endpoint into a stdout `summary` but never persists it.
- **Data roots diverge**: `config.data_dir = "website/data"` (LT:
  `website/data/{slug}/...`); `config.bi.data_dir = "data_bi"` (B3IT:
  `data_bi/{state,phase_2,phase_1,tokenizers,...}`). All paths are config-derived; no
  hardcoded `data_bi` literals in code. `bi/costs.py` already writes
  `website/data/bi_costs.json`, so B3IT has precedent under the website dir.
- **Notifications**: `notify.py` (pure stdlib SMTP via Gmail) + `notify-on-failure.yml`
  fire only on `workflow_run` failure. `notify.py` must keep running on system Python
  (the failure watcher runs before `uv sync`).

## Facts vs. derivations

Consistent with the BI/LT change-detection design: the ledger stores only **facts**
(money actually spent, with timestamps and counts). Today's and cumulative totals,
and the email contents, are **derived** by reading the ledger and the live run state;
nothing about the emails is persisted.

---

## PR 1 — Layout unification

Give LT and B3IT one shared data root with minimal risk to the live site: move B3IT
only; leave LT where it is.

- **Config**: `config.toml` `[bi] data_dir = "data_bi"` → `"website/data/bi"`. Every
  B3IT path (`state_dir`, `phase_2_dir`, `get_phase_1_dir`, tokenizers, etc.) derives
  from this, so the single change relocates the whole subtree.
- **Data move**: `git mv data_bi website/data/bi` (wholesale; ~274 MB of tracked
  blobs, renamed in place). Resulting layout: LT at `website/data/{slug}/...`, B3IT at
  `website/data/bi/{state,phase_2,...}`.
- **Generator**: `generate_site.py` iterates `DATA_DIR.iterdir()` treating each child
  as an endpoint. It already skips non-dirs (so `bi_costs.json` is fine), but `bi/` is
  a dir and would be walked as a fake endpoint. Add an explicit reserved-name skip
  (e.g. `RESERVED_DATA_DIRS = {"bi"}`, skipped in the enumeration loop). This also
  covers the deploy path, which runs `generate_site.py`.
- **Workflow commit paths**: `bi-monitor.yml`'s commit step does `git add data_bi` →
  change to `git add website/data` (covers the relocated B3IT data now, and the
  `website/data/spend/` ledger that bi-monitor will also write in PR 2; bi-monitor
  never touches LT data, so the broader add stages nothing extra). `update-endpoints.yml`
  and `run-main.yml` use `git add .` (already covers the new paths). Deploy publishes the generated site and
  the committed `website/data` JSON; it will now also carry `website/data/bi` data,
  which is harmless (nothing references it yet) and aligned with eventually surfacing
  B3IT on the site.
- **Unaffected**: root-level `endpoints_bi.yaml`, `endpoints_cache_bi.yaml`,
  `bi_selection.toml`, `strategies_test_reasoning.json` — none live under `data_bi`.

**Tests**: `generate_site` skips the reserved `bi/` dir (no phantom endpoint, no
crash); `config.bi` paths resolve under `website/data/bi`. Existing BI/LT tests pass
unchanged after the config + skip changes.

**Out of scope for this PR**: whether large committed artifacts (tokenizers,
phase_1, bi_prevalence, logprob_stats — research leftovers) should be git-ignored or
pruned. Noted as a separate cleanup; this PR only relocates.

---

## PR 2 — Universal actual-spend ledger

Capture the `response.cost` that's already computed for every request, for LT and
B3IT alike, into an append-only per-endpoint ledger. Depends on PR 1 (ledger lives in
the unified root).

### Storage

`website/data/spend/{slug}/{YYYY-MM}.jsonl` — one JSON object per line:

```
{"timestamp": "2026-06-22T00:55:13+00:00", "kind": "onboard", "cost": 0.0123,
 "n_queries": 412, "n_errors": 7}
```

- `slug = slugify(f"{model}#{provider}")` (same scheme as state/phase_2 dirs).
- `kind ∈ {lt, vetting, onboard, recheck, reinit, monitor}`. Onboarding-family =
  `{onboard, recheck, reinit}`; monitoring = `monitor`; LT = `lt`; vetting = `vetting`.
- One line per (endpoint, logical run-unit). Per-endpoint, per-month JSONL keeps each
  file small (≤ ~720 lines/month even for hourly LT), makes writes pure appends (no
  read-modify-write of a large file), and minimizes git-merge contention across the
  concurrent workflow pushes.
- Cumulative totals = sum `cost` over `website/data/spend/**/*.jsonl` filtered by
  `kind` (small files; reading all per run is fine). LT entries simply accumulate for
  future website use; nothing in PR 3 reads them.

There is one universal ledger, not per-method ledgers: each per-endpoint file holds
mixed-`kind` lines (`lt`, `monitor`, `onboard`, …). It lives at the neutral
`website/data/spend/` — a sibling of `bi/`, under neither method's subtree, so it
picks no side. The generator's reserved-skip set (PR 1) gains `spend` alongside `bi`.

### Capture mechanism

Two converging paths, both ending at one `append_entries` API, chosen per call site by
whichever is cleaner:

- **B3IT (cost otherwise discarded)** — a `Spend` accumulator (`cost`, `n_queries`,
  `n_errors`) held in a module-level `contextvars.ContextVar` in `spend.py`.
  `OpenRouterClient.query` adds the just-computed `response.cost` (and increments
  counts) to the active bucket if one is set; otherwise it is a no-op (LT main path,
  tests, ad-hoc scripts unaffected). The caller (`onboard_one`, `run_endpoint`,
  `vet_one`) sets a fresh bucket per endpoint/activity, runs the work, then reads and
  records it. Because the bucket is owned by the caller, a 3h-timeout
  `asyncio.wait_for` cancellation of `reinit()` still leaves the partial spend
  recorded. Because it is context-scoped (not tied to a client instance), it captures
  `phase_1a` discovery — which builds its own client — and every other query path
  (re-probe, reference collection, temperature gate, daily sampling).
- **LT (responses already in hand)** — `main.py` already aggregates per-endpoint
  `total_cost`/success/error in its `summary`; at run end it writes one `kind=lt` line
  per endpoint. No ContextVar needed where the responses are already collected.

`spend.py` exposes: the `Spend` accumulator + `ContextVar`, a context-manager/helper
to scope a bucket, `SpendEntry` (pydantic), `append_entries(slug, entries)` (atomic
append), and `cumulative_by_kind()` for the digest.

### Wiring

- `api.py`: in `query` (or `_make_request` where `cost` is computed), add to the
  active bucket if present.
- `main.py` (LT): write per-endpoint `kind=lt` lines at run end.
- `update_endpoints.py`: bucket around `vet_one` (`kind=vetting`); bucket around each
  `onboard_one` (`kind=onboard` for fresh, `kind=recheck` for rechecks).
- `monitor.py`: bucket around the daily sample in `run_endpoint` (`kind=monitor`), and
  a separate bucket around the change-triggered `reinit` (`kind=reinit`).
- Workflows already commit `website/data` (PR 1), so ledger files are persisted.

**Tests** (first): ContextVar capture across nested `asyncio` tasks (including a child
task created inside `sample_prompts`); partial spend survives `asyncio.wait_for`
cancellation; JSONL append round-trip; `cumulative_by_kind` aggregation across
multiple slug/month files; `query` is a no-op when no bucket is set.

---

## PR 3 — B3IT daily digest emails

Two independent emails, each built in-process at the end of its own daily B3IT
workflow and sent only when something notable happened. Depends on PR 2.

### Reports (in memory, not persisted)

- `update_endpoints_bi_lifecycle()` returns an `OnboardingReport`: rows of
  `{endpoint, outcome, n_bis, today_spend}` where `outcome ∈ {onboarded, timeout,
  no_bis, bad_temperature, recheck_resurrected, recheck_still_no_bis}`. The 3h timeout
  is captured in the existing `except asyncio.TimeoutError` branch (`outcome=timeout`).
- `monitor()` builds a `MonitorReport`: rows of `{endpoint, event, change_date,
  n_bis_after, today_spend}` where `event ∈ {change_detected, reonboarded,
  reonboard_no_bis, retired_stalled}`, plus today's total monitoring spend across all
  sampled endpoints.

`today_spend` for a row = the bucket recorded for that endpoint this run (PR 2).

### Digest + send (`bi/digest.py`)

- Refactor `notify.py` to expose a reusable `send_email(creds, subject, plain,
  html=None)` (via `EmailMessage.add_alternative` for the HTML part). `notify.py`
  stays pure stdlib so the failure watcher still runs on system Python; `bi/digest.py`
  (uv env) imports `send_email` and `load_creds_from_env` from it — single source of
  truth for SMTP.
- `digest.py` renders an HTML table + plain-text fallback from a report, pulling
  today's totals from the run and cumulative totals from `cumulative_by_kind()`
  (PR 2). **Gate**: send only if the report has ≥1 row; otherwise skip (quiet days
  produce no email — failures are still covered by the existing failure email). Spend
  is still logged to the ledger on quiet days, so the next email that does fire shows
  correct cumulative figures.
- The send happens as the last action of the python entrypoint. A send failure
  raises → surfaces via the existing failure email (loud, not silent). Missing SMTP
  secrets fail early, as in `notify.py` today.

### Email content

- **Onboarding email** (update-endpoints), subject e.g.
  `[trackllm] B3IT onboarding: 4 onboarded, 1 timed out, 2 no-BIs`: a table of the
  notable endpoints (onboarded / 3h-timeout / not-enough-BIs / recheck-resurrected /
  recheck-still-no-BIs / bad-temperature) with BI count and today's onboarding `$`;
  footer with today's and cumulative onboarding spend.
- **Monitoring email** (bi-monitor), subject e.g.
  `[trackllm] B3IT monitoring: 2 changes detected`: a table of changed / re-onboarded
  / re-onboard-no-BIs / retired-stalled endpoints with change date, BIs after, and
  re-onboard `$`; footer with today's and cumulative monitoring spend, and today's and
  cumulative reinit spend.

### Workflows

Add `GMAIL_USER`, `GMAIL_APP_PASSWORD`, `NOTIFY_EMAIL` env to the run step of
`update-endpoints.yml` and `bi-monitor.yml` (scheduled runs on `main` have secret
access).

**Tests** (first): report construction for each outcome/event type from mocked
lifecycle/monitor runs; the gate (no rows → no send); HTML + plain rendering snapshot;
`send_email` HTML/plain assembly.

---

## Rollout order

1. PR 1 (layout unification) — mechanical, validated by existing tests + the new skip.
2. PR 2 (spend ledger) — independently valuable; lands in the unified layout.
3. PR 3 (digest emails) — consumes the ledger and run outcomes.

## Out of scope (YAGNI)

- Website surfacing of actual spend (the ledger makes it possible later).
- LT emails (the ledger captures LT spend, but no LT digest is sent).
- Combining the two emails into one / reordering the two workflows.
- Pruning or git-ignoring large committed B3IT research artifacts (separate cleanup).
