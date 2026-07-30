# Implementation plan: every OpenRouter endpoint on the site, with a status

Spec: `docs/superpowers/specs/2026-07-30-all-endpoints-status-design.md` — read it first.

## Global Constraints

- Run everything with `uv` (`uv run --frozen pytest`, etc.). Never commit uv.lock churn.
- After editing code run `prek run --all-files`; commit with `--no-verify` (stale pre-commit hook).
- Filenames for outputs use `slugify` from `util.py`; endpoint slug is `slugify(f"{model}#{provider}")`.
- Tests first, then code to green. Succinct code, ~10% comments, no repeated blocks, no default argument values (single source of truth in config or a top-level constant).
- The site build (`make build`) must stay fully offline: it reads only committed files.
- Statuses are per-method (`lt`, `bi`) plus a derived `headline`. Fixed copy lives in ONE Python dict; templates/TS never invent status text.
- Free endpoint = advertised cost `[0, 0]` (prompt + completion both zero).
- `untrackable` headline = endpoint claims neither temperature nor logprobs+top_logprobs.
- Headline priority (first match wins): tracked → retired → untrackable → too_expensive → not_selected → errors_out → pending → free_excluded.
- bad_temperature copy must say: rejects/ignores the temperature parameter, T=0 sampling impossible, presumably to prevent distillation.

## Task 1: pipeline writers (catalog snapshot + LT failure cache)

Files: `src/trackllm_website/update_endpoints.py`, `src/trackllm_website/config.py`, new tests in `tests/`.

1. Extend `parse_model_endpoints` / `Endpoint` with `supports_logprobs: bool | None = None` stamped from endpoint-level `supported_parameters` (`"logprobs" in params and "top_logprobs" in params`; None when the field is absent).
2. New writer `save_endpoints_catalog(endpoints, path)` → `endpoints_catalog.yaml`, entries sorted by (model, provider): `{model, provider, cost: [in, out], created, supports_temperature, supports_logprobs, free}`. Written from the FULL unfiltered fetch: `get_endpoints` must expose the catalog before the free-endpoint and cost filters drop entries (add an `include_free`/raw hook or write the catalog inside `get_endpoints` before filtering — implementer's choice, keep one fetch for BI + catalog, do not add a third network sweep).
3. `test_endpoint_logprobs` failures currently vanish. Capture reason ("error: <api error>" or "returned N logprobs, expected M") and persist via new `endpoints_cache_lt.yaml`: `{failures: [{model, provider, reason, last_seen}]}`. An endpoint that later passes is removed from the cache. Follow the shape/IO patterns of `EndpointCache` (`bi/vetting.py`).
4. Wire both writers into `update_endpoints.main()` flow.
5. Tests: catalog writer round-trip incl. free endpoint + missing supported_parameters; failure cache add/clear/persist; parse stamping of supports_logprobs.
6. Generate the initial `endpoints_catalog.yaml` by running the real fetch once (network OK for this one-off; ~1 min) and commit it. Do NOT run the LT probe sweep (spends money); commit an empty `endpoints_cache_lt.yaml` (`failures: []`).

## Task 2: status resolver

Files: new `src/trackllm_website/generate_site/status.py`, new `tests/test_status.py`.

1. `EndpointStatus` (pydantic or dataclass, match neighboring style): `lt: str`, `bi: str`, `headline: str`, `lt_detail: str | None`, `bi_detail: str | None`.
2. `resolve_statuses(...)` — pure; inputs only from committed files (caller loads them): catalog entries, `endpoints_lt` list, set of LT slugs with observations (from `lt_scores` presence) and stalled flags (`ResultsStorage.is_stalled` results computed by caller), `endpoints_bi` list, BI `EndpointCache`, BI states (`load_all_states`), `SelectionPolicy`, LT failure cache. Output `dict[slug, EndpointStatus]` over union(catalog, previously-tracked). "Previously tracked" = has LT observations or a BI state file.
3. LT statuses: `tracked` (in endpoints_lt + observations), `stalled` (observations but dropped/stalled), `probe_failed` (in LT failure cache; detail = recorded reason), `no_logprobs` (supports_logprobs is False), `too_expensive` (sum(cost) ≥ config.api.max_cost_mtok), `free_excluded`, else `pending`.
4. BI statuses: `monitoring` / `retired:<reason>` (state files), `bad_temperature` / `too_expensive` / `liar` (cache buckets), `excluded` (policy exclude globs via `_matches_any`), `not_selected` (in endpoints_bi, no monitoring state), `free_excluded`, else `pending`. Note: monitoring/retired come from STATE FILES, never from re-running selection (that needs the network popularity feed).
5. `STATUS_COPY: dict[str, str]` — one sentence per status (see Global Constraints for bad_temperature wording). `headline_for(lt, bi)` implementing the priority chain; `errors_out` groups liar/probe_failed/retired:unreachable.
6. Tests: fixture catalog exercising EVERY lt/bi status, headline priority order (incl. grok case: lt tracked + bi too_expensive → tracked), historical-only endpoint, free endpoint, both-blocked → untrackable.

## Task 3: stamp statuses into generated JSON

Files: `generate_site/{__main__,tracked,overview,lt,b3it,model,org,provider}.py` as needed; tests.

1. Caller in `__main__.py`/`overview.py` loads the new files, computes stalled/observation sets, calls `resolve_statuses`, passes the dict down.
2. Fleet JSON (overview): every union endpoint becomes a row: slug, model, provider, org, headline, per-method statuses, one-line reason (copy + detail), and existing trace fields for tracked rows.
3. Endpoint JSON: add `status` object (both methods + copy + details). Emit endpoint JSON for every union endpoint (untracked ones get status + catalog metadata: pricing, claims, created; no series).
4. Model JSON for every catalog model (model page status summary, e.g. "0 of 7 endpoints trackable"); org/provider pages include untracked models/endpoints with headline badges. Reuse `naming.py` conventions; orgs derive from model prefix as today.
5. `tracked.py` semantics unchanged (charts only for endpoints with observations).
6. Tests: JSON emitters produce statuses; a gpt-5-like fixture model yields untrackable rows and a model page with the summary line.

## Task 4: front end

Files: `website/templates/{index,endpoint,model,org,provider}.html.j2`, `website/src/{overview,endpoint,model,components}.ts`, `website/style.css`.

1. Fleet table: status filter chips (tracked default; untrackable / too expensive / retired / errors / pending / free). Untracked rows: status badge + one-line reason in the trace column. Follow existing chips/table patterns in index template + overview.ts.
2. Search: matches model AND provider names across ALL rows regardless of active chip; highlight the matched substring (e.g. `<mark>`). Searching while a chip hides matches must still surface them (search overrides chip filter).
3. Endpoint page: status card at top (both methods + copy + detail); untracked pages render card + metadata, no chart section.
4. Model page: status summary line; per-endpoint badges. Org/provider pages: badges on untracked entries.
5. Style: badges/chips/mark styling consistent with existing design; check dark mode if the site has it (follow existing CSS custom properties).
6. Tests: extend `website/test/` bun tests: untracked endpoint page renders card and no chart; search matches "gpt-5" by model and "alibaba" by provider with highlighted `<mark>`; chip filtering counts.

## Task 5: integration, build, visual check

1. `make build` from a clean checkout of the branch; fix anything that breaks. Confirm page counts: ~1,100 endpoint pages, ~334 model pages.
2. Run full `uv run --frozen pytest` + `prek run --all-files` + bun tests.
3. Headless-chromium screenshots (see docs/memory recipe): overview with chips + a search for "gpt-5", one untrackable endpoint page (e.g. gpt-5.4@openai), one tracked endpoint page (regression), anthropic org page. Save to scratchpad, review for layout breakage.
4. Fix visual defects found; re-screenshot.
