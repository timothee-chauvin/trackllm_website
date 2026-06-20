# BI Selection: Recency, Popularity, Temperature Pre-filter, Deselection Grace

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. **All work happens in the worktree `~/.config/superpowers/worktrees/bi-selection-v2` on branch `feat/bi-selection-v2`** — every subagent must run from that directory.

**Goal:** Make BI flagship selection self-maintaining and usage-aware: select the newest N versions per family (by OpenRouter `created` date), add a popularity rule (top-N by token usage), pre-filter endpoints that self-declare no temperature support (cheap + avoids onboarding waste), and give a 1-month grace before delisting a model that drops out of any category.

**Architecture:** The OpenRouter `/models` API exposes `created` (release date) and `supported_parameters` per model; `/api/v1/datasets/rankings-daily` exposes token-usage rankings. We thread `created` onto each `Endpoint` (persisted), use `supported_parameters` transiently at vetting to skip temperature-unsupported models, add `latest_n` and a `popular` rule kind to the pure selection engine, and add a `deselected_since` timestamp to per-endpoint state so the lifecycle keeps a just-dropped model for `deselection_grace_days`.

**Tech Stack:** Python 3.13, uv, pydantic + pydantic-settings, orjson, requests/aiohttp, pytest.

**Conventions:** tests first; `uv run` everything; `prek run --all-files` after edits; no default argument values (config/constants are the single source of truth — pydantic optional model fields are fine); never silence errors; ~10% comments; never stage `uv.lock`.

**Builds on:** merged PR #2 (cost-based selection: `bi/vetting.py`, `bi/selection.py`, `bi/costs.py`, the lifecycle in `update_endpoints.py`).

## Decisions (from the user)
- `latest_n`: yes. Exclude `-fast` and `-search` variants; **keep** `-preview`.
- `popular` rule: yes (top-N by OpenRouter token usage).
- `temp:NO` pre-filter: yes (skip probing models whose `supported_parameters` omits `temperature`).
- Deselection grace: a model leaving popular/latest_n/any category keeps being monitored for **1 month** before delisting.
- Flagships keep the ceiling exemption.

## Validated facts (confirmed live this session)
- `/models` objects include `created` (unix ts), `canonical_slug`, and `supported_parameters` (list).
- Every model we *proved* ignores T=0 (gpt-5, gpt-5-mini/nano, o3-mini, gpt-5.5) reports `temp:NO` (temperature absent from `supported_parameters`); claude-opus-4.8 also reports `temp:NO`. So `temp:NO` is a reliable *skip* signal; `temp:YES` still needs the runtime gate.
- `/api/v1/datasets/rankings-daily` returns rows `{date, model_permaslug, total_tokens}`, 30-day window, top-50/day + an `other` row, auth via our API key. `model_permaslug` has a `-YYYYMMDD` date suffix and anthropic uses reversed word order (`anthropic/claude-4.7-opus-...` vs `/models` id `anthropic/claude-opus-4.7`) — map via `canonical_slug`.

## File structure
- `config.py`, `config.toml` — `Endpoint.created`; `[bi.popularity]`; `deselection_grace_days`.
- `update_endpoints.py` — `get_endpoints` captures `created`+`supported_parameters`; `update_endpoints_bi` temp pre-filter; `save_endpoints_bi` persists `created`; lifecycle: fetch popularity, thread it, deselection grace.
- `bi/popularity.py` (NEW) — rankings-daily client + permaslug→id mapping.
- `bi/selection.py` — `Rule.latest_n`, `popular` kind, `select_monitoring_targets(..., popular_models)`.
- `bi/state.py` — `EndpointBIState.deselected_since`.
- `bi/costs.py` — `build_cost_summary`/`preview` thread `popular_models`.
- `bi_selection.toml` — `latest_n` flagship globs, `popular` rule, updated `exclude`.

---

### Task 1: `Endpoint.created` + config keys + capture + persistence

**Files:** `config.toml`, `src/trackllm_website/config.py`, `src/trackllm_website/update_endpoints.py`, test `tests/test_bi_created.py`

- [ ] **Step 1: config.toml** — add `[bi.popularity]` and a grace key:

```toml
[bi.popularity]
top_n = 40

[bi.reinit]
# ... existing keys ...
deselection_grace_days = 30
```
(Add `deselection_grace_days = 30` inside the existing `[bi.reinit]` block.)

- [ ] **Step 2: config.py** — add to `Endpoint` after `cost_per_request`:

```python
    created: datetime | None = None  # model release date from OpenRouter /models
```
(Import `datetime` if not already: `from datetime import datetime`.) Add a `PopularityConfig(BaseModel)` with `top_n: int`; add `popularity: PopularityConfig` to `BIConfig`; add `deselection_grace_days: int` to `ReinitConfig`.

- [ ] **Step 3: failing test** `tests/test_bi_created.py`:

```python
from datetime import datetime, timezone
from trackllm_website.config import Endpoint
from trackllm_website.update_endpoints import save_endpoints_bi
import trackllm_website.update_endpoints as ue


def test_save_endpoints_persists_created(tmp_path, monkeypatch):
    monkeypatch.setattr(ue.config, "endpoints_yaml_path_bi", tmp_path / "endpoints_bi.yaml")
    e = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1),
                 cost_per_request=0.00001,
                 created=datetime(2026, 6, 16, tzinfo=timezone.utc))
    save_endpoints_bi([e])
    text = (tmp_path / "endpoints_bi.yaml").read_text()
    assert "created" in text and "2026-06-16" in text
```

- [ ] **Step 4: run → fail.** `uv run pytest tests/test_bi_created.py -v`

- [ ] **Step 5: implement.**
- In `save_endpoints_bi` (update_endpoints.py), add `"created"` to each dumped entry: `"created": e.created.isoformat() if e.created else None`.
- In `get_endpoints` / `get_model_endpoints`: the `/models` list already gives model ids; capture `created` (unix ts → `datetime.fromtimestamp(c, tz=timezone.utc)`) into a `model_id -> created` map, and set `created=` when constructing each `Endpoint`. (Read the current `get_endpoints`/`get_model_endpoints` to thread the map through.)
- pydantic-settings loads `endpoints_bi.yaml` into `config.endpoints_bi`; `Endpoint.created: datetime | None` parses an ISO string automatically.

- [ ] **Step 6: run → pass; config check.**
`uv run pytest tests/test_bi_created.py -v` then
`uv run python -c "from trackllm_website.config import config; print(config.bi.popularity.top_n, config.bi.reinit.deselection_grace_days)"` → `40 30`

- [ ] **Step 7: prek + full suite + commit.**
```bash
uv run pytest tests/ -q && prek run --all-files
git add config.toml src/trackllm_website/config.py src/trackllm_website/update_endpoints.py tests/test_bi_created.py
git commit -m "capture and persist model created date; popularity + grace config"
```

---

### Task 2: `temp:NO` pre-filter in vetting

**Files:** `src/trackllm_website/config.py` (Endpoint), `src/trackllm_website/update_endpoints.py`, test `tests/test_bi_temp_prefilter.py`

Capture `supported_parameters` at catalog pull; route models that omit `temperature` straight to `bad_temperature` without probing.

- [ ] **Step 1: Endpoint transient field.** Add to `Endpoint`:
```python
    supports_temperature: bool | None = None  # from /models supported_parameters; transient (not persisted)
```
Do NOT add it to `save_endpoints_bi`'s dump (it's transient, recomputed each refresh).

- [ ] **Step 2: failing test** `tests/test_bi_temp_prefilter.py`:

```python
from trackllm_website.config import Endpoint
from trackllm_website.update_endpoints import partition_temperature


def ep(model, supports):
    return Endpoint(api="openrouter", model=model, provider="p", cost=(1, 1),
                    supports_temperature=supports)


def test_partition_skips_temp_unsupported():
    eps = [ep("a", True), ep("b", False), ep("c", None)]
    probe, skip = partition_temperature(eps)
    assert [e.model for e in probe] == ["a", "c"]   # None => probe (unknown)
    assert [e.model for e in skip] == ["b"]          # explicit False => skip
```

- [ ] **Step 3: run → fail.**

- [ ] **Step 4: implement.**
- In `get_endpoints`/`get_model_endpoints`: set `supports_temperature = "temperature" in (model_obj.get("supported_parameters") or [])` when building each Endpoint (thread the `supported_parameters` from the `/models` object alongside `created`).
- Add the pure helper to update_endpoints.py:
```python
def partition_temperature(endpoints: list[Endpoint]) -> tuple[list[Endpoint], list[Endpoint]]:
    """(to_probe, to_skip): skip endpoints that explicitly declare no temperature
    support (supports_temperature is False); probe unknown (None) and True."""
    probe = [e for e in endpoints if e.supports_temperature is not False]
    skip = [e for e in endpoints if e.supports_temperature is False]
    return probe, skip
```
- In `update_endpoints_bi`: after building the to-vet list (and before the concurrent vetting), call `partition_temperature`; add each skipped endpoint to `cache.add_bad_temperature(e)` (with a `logger.info` count), and only vet the `probe` list. This must run *before* the strategy-resolution/vetting fan-out so we never spend on temp:NO models.

- [ ] **Step 5: run → pass; prek; full suite; commit.**
```bash
uv run pytest tests/test_bi_temp_prefilter.py -v && uv run pytest tests/ -q && prek run --all-files
git add src/trackllm_website/config.py src/trackllm_website/update_endpoints.py tests/test_bi_temp_prefilter.py
git commit -m "vetting: skip temperature-unsupported models (cheap bad_temperature pre-filter)"
```

---

### Task 3: `bi/popularity.py` — rankings-daily client + permaslug mapping

**Files:** `src/trackllm_website/bi/popularity.py` (NEW), test `tests/test_bi_popularity.py`

- [ ] **Step 1: VERIFY the mapping first** (one-off inspection, not committed). Run:
```bash
uv run python -c "
import requests
from trackllm_website.config import config
h={'Authorization':f'Bearer {config.openrouter_api_key}'}
models=requests.get('https://openrouter.ai/api/v1/models',headers=h).json()['data']
canon={m['canonical_slug']:m['id'] for m in models}
rk=requests.get('https://openrouter.ai/api/v1/datasets/rankings-daily',headers=h).json()['data']
import re
for r in rk[:8]:
    ps=r['model_permaslug']; base=re.sub(r'-\d{8}$','',ps)
    print(ps,'-> canon?',canon.get(base),'  base=',base)
"
```
Determine the exact rule: strip trailing `-YYYYMMDD`, then match against `canonical_slug` → id. If `canonical_slug` doesn't match, fall back to matching the stripped permaslug directly against model `id`s. **Implement whichever the inspection shows works**; the test below encodes the rule you confirm.

- [ ] **Step 2: failing test** `tests/test_bi_popularity.py` (pure mapping + aggregation, no network):

```python
from trackllm_website.bi.popularity import aggregate_rankings, map_to_model_ids


def test_aggregate_and_map():
    rows = [
        {"date": "2026-06-16", "model_permaslug": "deepseek/deepseek-v4-flash-20260423", "total_tokens": "100"},
        {"date": "2026-06-15", "model_permaslug": "deepseek/deepseek-v4-flash-20260423", "total_tokens": "50"},
        {"date": "2026-06-16", "model_permaslug": "tencent/hy3-preview-20260421", "total_tokens": "120"},
        {"date": "2026-06-16", "model_permaslug": "other", "total_tokens": "999"},
    ]
    ranked = aggregate_rankings(rows)  # [(permaslug, tokens)] desc, 'other' dropped
    assert ranked[0][0] == "deepseek/deepseek-v4-flash-20260423"  # 150 > 120
    assert all(p != "other" for p, _ in ranked)

    canonical = {"deepseek/deepseek-v4-flash": "deepseek/deepseek-v4-flash",
                 "tencent/hy3-preview": "tencent/hy3-preview"}
    ids = map_to_model_ids([p for p, _ in ranked], canonical)
    assert ids == ["deepseek/deepseek-v4-flash", "tencent/hy3-preview"]
```

- [ ] **Step 3: run → fail.**

- [ ] **Step 4: implement `bi/popularity.py`** (adapt the mapping to step-1 findings):

```python
"""OpenRouter token-usage rankings: which models are popular.

Source: GET /api/v1/datasets/rankings-daily (top-50 models/day by tokens + an
'other' row, ~30-day window). model_permaslug carries a -YYYYMMDD suffix and a
provider-specific word order, so map to /models ids via canonical_slug.
"""

import re
from collections import Counter

import requests

from trackllm_website.config import config

RANKINGS_URL = "https://openrouter.ai/api/v1/datasets/rankings-daily"
MODELS_URL = "https://openrouter.ai/api/v1/models"
_DATE_SUFFIX = re.compile(r"-\d{8}$")


def aggregate_rankings(rows: list[dict]) -> list[tuple[str, int]]:
    """Sum total_tokens per permaslug over the window, drop 'other', sort desc."""
    agg: Counter[str] = Counter()
    for r in rows:
        slug = r["model_permaslug"]
        if slug == "other":
            continue
        agg[slug] += int(r["total_tokens"])
    return agg.most_common()


def map_to_model_ids(permaslugs: list[str], canonical_to_id: dict[str, str]) -> list[str]:
    """Map ranking permaslugs (date-suffixed) to /models ids, preserving order,
    skipping ones with no match. De-dupes (a model can appear under versioned slugs)."""
    out: list[str] = []
    seen: set[str] = set()
    for ps in permaslugs:
        base = _DATE_SUFFIX.sub("", ps)
        mid = canonical_to_id.get(base)
        if mid and mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


def fetch_popular_models(top_n: int) -> list[str]:
    """Return the top_n most-used model ids (popularity order). Network call."""
    headers = {"Authorization": f"Bearer {config.openrouter_api_key}"}
    rows = requests.get(RANKINGS_URL, headers=headers, timeout=30).json()["data"]
    models = requests.get(MODELS_URL, headers=headers, timeout=30).json()["data"]
    canonical_to_id = {m["canonical_slug"]: m["id"] for m in models}
    ranked = aggregate_rankings(rows)
    ids = map_to_model_ids([p for p, _ in ranked], canonical_to_id)
    return ids[:top_n]
```

- [ ] **Step 5: run → pass.** `uv run pytest tests/test_bi_popularity.py -v`

- [ ] **Step 6: live sanity (not committed).** `uv run python -c "from trackllm_website.bi.popularity import fetch_popular_models; print(fetch_popular_models(15))"` — confirm it returns real model ids (deepseek-v4-flash, tencent/hy3-preview, etc.). Report the list.

- [ ] **Step 7: prek + full suite + commit.**
```bash
uv run pytest tests/ -q && prek run --all-files
git add src/trackllm_website/bi/popularity.py tests/test_bi_popularity.py
git commit -m "add OpenRouter popularity (rankings-daily) client + permaslug mapping"
```

---

### Task 4: `Rule.latest_n` + engine (created-based)

**Files:** `src/trackllm_website/bi/selection.py`, test `tests/test_bi_selection_latest_n.py`

- [ ] **Step 1: failing tests:**

```python
from datetime import datetime, timezone
from trackllm_website.bi.selection import Rule, SelectionPolicy, select_monitoring_targets
from trackllm_website.config import Endpoint


def ep(model, provider, cpr, created):
    return Endpoint(api="openrouter", model=model, provider=provider, cost=(1, 1),
                    cost_per_request=cpr,
                    created=datetime(2026, created, 1, tzinfo=timezone.utc))


def test_latest_n_keeps_newest_per_pattern():
    pol = SelectionPolicy(budget_per_month=100, max_endpoint_cost=100, exclude=[],
        rules=[Rule(name="flagships", kind="models", patterns=["z-ai/glm-*"],
                    providers_per_model=1, latest_n=2, flagship=True)])
    cands = [ep("z-ai/glm-5.2", "p", 0.00001, 6), ep("z-ai/glm-5.1", "p", 0.00001, 4),
             ep("z-ai/glm-5", "p", 0.00001, 2), ep("z-ai/glm-4.7", "p", 0.00001, 1)]
    sel, _ = select_monitoring_targets(cands, pol, [])
    assert sorted(e.model for e in sel) == ["z-ai/glm-5.1", "z-ai/glm-5.2"]


def test_latest_n_cheapest_provider_per_kept_model():
    pol = SelectionPolicy(budget_per_month=100, max_endpoint_cost=100, exclude=[],
        rules=[Rule(name="f", kind="models", patterns=["m/a"], providers_per_model=1,
                    latest_n=1, flagship=True)])
    cands = [ep("m/a", "cheap", 0.00001, 6), ep("m/a", "pricey", 0.0001, 6)]
    sel, _ = select_monitoring_targets(cands, pol, [])
    assert [e.provider for e in sel] == ["cheap"]
```

- [ ] **Step 2: run → fail** (Rule has no `latest_n`; `select_monitoring_targets` takes 2 args).

- [ ] **Step 3: implement.**
- Add `latest_n: int | None = None` to `Rule`.
- Add the `popular_models: list[str]` parameter to `select_monitoring_targets` now (used in Task 5; pass `[]` here). Signature: `def select_monitoring_targets(candidates, policy, popular_models):`.
- In the `kind == "models"` branch, BEFORE the existing per-pattern model iteration: when `rule.latest_n is not None`, restrict the matched `model_keys` for each pattern to the newest `latest_n` by `created` (treat `created is None` as oldest). Implement per-pattern (each pattern is its own family):
```python
        if rule.kind == "models":
            stop = False
            for pattern in rule.patterns:
                pat_models = [m for m in by_model if _matches_any(by_model[m][0], [pattern])]
                if rule.latest_n is not None:
                    pat_models.sort(
                        key=lambda m: by_model[m][0].created or datetime.min.replace(tzinfo=timezone.utc),
                        reverse=True,
                    )
                    pat_models = pat_models[: rule.latest_n]
                # within the selected family models, cheapest-first then cheapest provider
                pat_models.sort(key=lambda m: (monthly_cost(by_model[m][0]), m))
                for m in pat_models:
                    eps = by_model[m]
                    n = len(eps) if rule.providers_per_model == "all" else rule.providers_per_model
                    for e in eps[:n]:
                        ... (existing add / max_monthly_cost / budget-stop logic) ...
                if stop:
                    break
```
**IMPORTANT:** Preserve the existing engine invariants from PR #2 — the post-loop flagship-warning / non-flagship-`ValueError` budget checks, the `stop`-flag budget break (not `return`), total-order tie-breaks, and dedup via `add()`. Restructuring the models branch to iterate per-pattern (instead of the old "all matched models" set) is the change; do not regress the budget/flagship logic. Add `from datetime import datetime, timezone` import.
- Update the existing callers of `select_monitoring_targets` (in `bi/costs.py` and `update_endpoints.py` and existing tests) to pass a third arg `[]` for now (Task 5/7 wire real popularity).

- [ ] **Step 4: run latest_n tests + the full existing selection suite (must stay green).**
`uv run pytest tests/test_bi_selection_latest_n.py tests/test_bi_selection_engine.py -v`

- [ ] **Step 5: prek + full suite + commit.**
```bash
uv run pytest tests/ -q && prek run --all-files
git add src/trackllm_website/bi/selection.py src/trackllm_website/bi/costs.py src/trackllm_website/update_endpoints.py tests/
git commit -m "selection: latest_n keeps newest versions per pattern by created date"
```

---

### Task 5: `popular` rule kind + engine

**Files:** `src/trackllm_website/bi/selection.py`, test `tests/test_bi_selection_popular.py`

- [ ] **Step 1: failing tests:**

```python
from trackllm_website.bi.selection import Rule, SelectionPolicy, select_monitoring_targets
from trackllm_website.config import Endpoint


def ep(model, provider, cpr):
    return Endpoint(api="openrouter", model=model, provider=provider, cost=(1, 1),
                    cost_per_request=cpr)


def test_popular_selects_top_n_present_in_candidates():
    pol = SelectionPolicy(budget_per_month=100, max_endpoint_cost=100, exclude=[],
        rules=[Rule(name="popular", kind="popular", patterns=[], providers_per_model=1)])
    cands = [ep("m/a", "cheap", 0.00001), ep("m/a", "pricey", 0.0001), ep("m/b", "p", 0.00001)]
    # popularity order: m/b most popular, m/a next, m/c absent from candidates
    sel, breakdown = select_monitoring_targets(cands, pol, ["m/b", "m/a", "m/c"])
    assert set(e.model for e in sel) == {"m/a", "m/b"}
    assert breakdown[next(e for e in sel if e.model == "m/a")] == "popular"
    assert [e.provider for e in sel if e.model == "m/a"] == ["cheap"]  # cheapest provider


def test_popular_respects_max_monthly_cost():
    pol = SelectionPolicy(budget_per_month=100, max_endpoint_cost=100, exclude=[],
        rules=[Rule(name="popular", kind="popular", patterns=[], providers_per_model=1,
                    max_monthly_cost=0.10)])  # 0.10/mo => cpr <= ~1.67e-5
    cands = [ep("m/cheap", "p", 0.00001), ep("m/pricey", "p", 0.00005)]
    sel, _ = select_monitoring_targets(cands, pol, ["m/pricey", "m/cheap"])
    assert [e.model for e in sel] == ["m/cheap"]
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement.** Extend `Rule.kind` Literal to include `"popular"`. In `select_monitoring_targets`, add a branch for `rule.kind == "popular"`: iterate `popular_models` in order; for each that is a candidate model (`m in by_model`), take its providers (`providers_per_model`, cheapest first), apply `max_monthly_cost` skip and the non-flagship budget-stop, `add(e, rule.name)`. Mirror the models-branch structure (stop flag, not return).

- [ ] **Step 4: run popular tests + full selection suite.**

- [ ] **Step 5: prek + full suite + commit.**
```bash
uv run pytest tests/ -q && prek run --all-files
git add src/trackllm_website/bi/selection.py tests/test_bi_selection_popular.py
git commit -m "selection: add popular rule (top-N by token usage)"
```

---

### Task 6: deselection grace (1 month)

**Files:** `src/trackllm_website/bi/state.py`, `src/trackllm_website/update_endpoints.py`, test `tests/test_bi_grace.py`

- [ ] **Step 1: state field.** Add to `EndpointBIState`: `deselected_since: datetime | None = None`. Confirm save/load round-trips it (pydantic `model_dump(mode="json")` already handles datetime; `load` via `model_validate`).

- [ ] **Step 2: failing test** `tests/test_bi_grace.py`:

```python
from datetime import datetime, timedelta, timezone
from trackllm_website.bi.state import EndpointBIState
from trackllm_website.config import Endpoint
from trackllm_website.update_endpoints import should_delist

NOW = datetime(2026, 6, 17, tzinfo=timezone.utc)


def state(model, deselected_since):
    return EndpointBIState(
        endpoint=Endpoint(api="openrouter", model=model, provider="p", cost=(1, 1)),
        status="monitoring", epochs=[], deselected_since=deselected_since)


def test_grace_logic():
    grace = 30
    assert should_delist(state("a", None), NOW, grace) is False           # just fell out this run
    assert should_delist(state("b", NOW - timedelta(days=10)), NOW, grace) is False  # within grace
    assert should_delist(state("c", NOW - timedelta(days=31)), NOW, grace) is True   # past grace
```

- [ ] **Step 3: run → fail.**

- [ ] **Step 4: implement.**
- `should_delist(state, now, grace_days)` pure helper in update_endpoints.py:
```python
def should_delist(state: EndpointBIState, now: datetime, grace_days: int) -> bool:
    """A monitoring endpoint that's no longer selected delists only after the grace
    period since it dropped out. Caller sets deselected_since when it leaves the set."""
    if state.deselected_since is None:
        return False
    return now - state.deselected_since >= timedelta(days=grace_days)
```
- In `select_lifecycle_actions`: the `delist` list becomes monitoring states **not in the selected set** for which `should_delist(...)` is True (i.e. past grace). (It needs `now` and `config.bi.reinit.deselection_grace_days` — pass `now`; read grace from config.)
- In the executor `update_endpoints_bi_lifecycle`, before computing actions: for each monitoring state, set `deselected_since = now` if it's NOT in the selected set and `deselected_since is None`; clear `deselected_since = None` if it IS in the selected set; `state.save(...)` those whose flag changed. Then `select_lifecycle_actions` uses the (now-updated) `deselected_since` to decide delist. The delist executor closes the epoch (`gap`) and retires as before.

- [ ] **Step 5: run grace test + lifecycle tests; prek; full suite; commit.**
```bash
uv run pytest tests/test_bi_grace.py tests/test_bi_lifecycle.py -v && uv run pytest tests/ -q && prek run --all-files
git add src/trackllm_website/bi/state.py src/trackllm_website/update_endpoints.py tests/test_bi_grace.py
git commit -m "lifecycle: 1-month grace before delisting a deselected model"
```

---

### Task 7: `bi_selection.toml` rewrite + wire popularity through lifecycle & costs

**Files:** `bi_selection.toml`, `src/trackllm_website/update_endpoints.py`, `src/trackllm_website/bi/costs.py`, test updates

- [ ] **Step 1: rewrite `bi_selection.toml`.**
- `exclude`: keep `*image*`, `*-fast`, `*customtools*`; **add** `*search*`; **add** `openrouter/*` (OpenRouter stealth/rotating aliases like `owl-alpha` that surface in popularity but aren't stable endpoints — defense-in-depth for the popular rule). Ensure no `-preview` exclude exists (it doesn't). Final exclude (confirm): `["*image*", "*-fast", "*search*", "*customtools*", "openrouter/*"]`.
- Rewrite the `flagships` rule to use `latest_n = 2` with family globs:
```toml
[[rule]]
name = "flagships"
kind = "models"
latest_n = 2
patterns = [
    "anthropic/claude-opus-*", "anthropic/claude-sonnet-*", "anthropic/claude-haiku-*",
    "anthropic/claude-fable-*",
    "openai/gpt-4o", "openai/gpt-4o-mini",
    "google/gemini-*-flash*",
    "deepseek/deepseek-v*", "deepseek/deepseek-chat*",
    "qwen/qwen3*", "z-ai/glm-*", "moonshotai/kimi-*",
    "x-ai/grok-*", "minimax/minimax-m*",
    "meta-llama/llama-4*", "mistralai/mistral-large*",
]
providers_per_model = 1
flagship = true
```
(Note: `openai/gpt-4o`/`gpt-4o-mini` are exact, not globs, so `latest_n` is a no-op for them — that's intentional, it avoids the `gpt-4o*`→search-preview trap. The `*search*` exclude also guards it.)
- Add a `popular` rule:
```toml
[[rule]]
name = "popular"
kind = "popular"
patterns = []
providers_per_model = 1
max_monthly_cost = 0.25
```
- Keep `provider-coverage` and `long-tail`. Drop the old hand-pinned `corroboration` patterns or update them to globs (your judgment — corroboration with `providers_per_model="all"` + `max_monthly_cost=0.10` on `["deepseek/deepseek-v*", "z-ai/glm-*", "moonshotai/kimi-*"]` is reasonable).

- [ ] **Step 2: thread popularity into the lifecycle.** In `update_endpoints_bi_lifecycle`: fetch `popular_models = fetch_popular_models(config.bi.popularity.top_n)` (from `bi/popularity`), pass it as the 3rd arg to `select_monitoring_targets(candidates, policy, popular_models)`. Wrap the fetch in try/except → `[]` on failure (popularity is best-effort; a rankings API hiccup shouldn't abort the daily run — log a warning).

- [ ] **Step 3: thread popularity into costs.** In `bi/costs.py`: `build_cost_summary(candidates, policy, popular_models)` gains the param and passes it through; `preview()` and `write_cost_summary()` fetch `fetch_popular_models(config.bi.popularity.top_n)` (best-effort, `[]` on failure) and pass it. Update the existing `build_cost_summary` test to pass `[]`.

- [ ] **Step 4: validate the policy loads + engine runs.**
`uv run python -c "from trackllm_website.bi.selection import load_policy; from trackllm_website.config import config, root; p=load_policy(root/config.bi.selection_path); print([r.name for r in p.rules], [r.latest_n for r in p.rules])"`

- [ ] **Step 5: prek + full suite + commit.**
```bash
uv run pytest tests/ -q && prek run --all-files
git add bi_selection.toml src/trackllm_website/update_endpoints.py src/trackllm_website/bi/costs.py tests/
git commit -m "selection policy: latest_n flagship globs + popular rule; wire popularity through"
```

---

### Task 8: integration check + PR readiness

**Files:** test `tests/test_bi_selection_integration.py`

- [ ] **Step 1: integration test** exercising latest_n + popular + exclude together on synthetic candidates (with `created` + a popularity list), asserting: newest-2 per family kept, a popular non-flagship model selected, an `*-fast`/`*search*` model excluded, and a flagship over the ceiling still selected. (Write it concretely against the engine; ~1 test function, real `SelectionPolicy` mirroring `bi_selection.toml`'s rule shapes.)

- [ ] **Step 2: full suite + prek.** `uv run pytest tests/ -q && prek run --all-files`

- [ ] **Step 3: live dry-preview is the USER's post-merge step** (it needs a fresh vetting to populate `created` + costs, which costs money and is now cheaper thanks to the temp:NO pre-filter). Do NOT run it here. Note in the final report that after merge the user runs: `update_endpoints_bi()` (fresh vet, now temp-pre-filtered) then `python -m trackllm_website.bi.costs preview`.

- [ ] **Step 4: commit.**
```bash
git add tests/test_bi_selection_integration.py
git commit -m "add latest_n + popular + exclude integration test"
```

---

## Self-review

- **Spec coverage:** `created` capture+persist (T1); temp:NO pre-filter (T2); popularity client (T3); `latest_n` (T4); `popular` rule (T5); deselection grace (T6); policy rewrite incl. exclude `-fast`/`-search`, keep `-preview`, + popularity wiring (T7); integration (T8). All four user decisions covered.
- **Engine-invariant risk (T4/T5):** restructuring the `models` branch and adding a `popular` branch must preserve PR #2's post-loop budget checks, stop-flag (not return), tie-breaks, and dedup. T4/T5 steps call this out; the existing `tests/test_bi_selection_engine.py` (kept green at each step) is the guard.
- **Signature change ripple:** `select_monitoring_targets` gains a required `popular_models` arg in T4 — every caller (costs.py, update_endpoints.py, tests) updated in T4, real values wired in T7.
- **`created`-missing candidates:** sort as oldest (T4) — carried-forward/legacy entries without `created` simply rank last under `latest_n`, never crash.
- **Best-effort popularity:** rankings fetch failures degrade to `[]` (T7), never abort the daily run.
- **No placeholder steps**; every code step shows the code.
