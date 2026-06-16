# Cost-Based BI Endpoint Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the "onboard every vetted endpoint" model with a budget-driven selection: vetting measures real $/request, a single legible policy file (`bi_selection.toml`) decides which endpoints to monitor within a monthly budget, phase 1 excludes endpoints that ignore T=0, and the backend emits cost/spend data for a website costs page.

**Architecture:** Cost management lives in exactly one place — `bi_selection.toml` (budget, per-rule caps, an explicit `max_endpoint_cost` ceiling). Vetting (`bi/vetting.py`) measures each endpoint's cost-per-request with its resolved query strategy and sorts the catalog into buckets (candidate / liar / too_expensive). A pure rule engine (`bi/selection.py`) turns the candidate pool + budget into the monitored set. Phase 1 gains a conditional temperature gate (high T=0 border-input prevalence triggers a T=0-vs-T=1 differential check; if temperature is a no-op, the endpoint is cached `bad_temperature`). The lifecycle onboards from the selected set instead of the full catalog.

**Tech Stack:** Python 3.13, uv, pydantic + pydantic-settings, orjson, tomllib (stdlib), pytest, asyncio.

**Conventions (CLAUDE.md / project):** tests first; `uv run` everything; `prek run --all-files` after edits; `slugify` from `util.py` for filenames; no default argument values (config/constants are the single source of truth); never silence errors; ~10% comments; never stage `uv.lock` (it carries an unrelated pending change).

**Builds on:** merged PR #1 (BI epoch state, detection, monitor, reinit, lifecycle, LT events). Branch from current `main`.

## Scope

In scope: cost-based vetting + bucketed cache, `bi_selection.toml` policy, selection rule engine, dropping the `$30/Mtok` cap for BI, the conditional temperature gate, lifecycle integration, cost-data emission (JSON), and the one-off resumption run.

Out of scope (deferred to the website plan): the costs *page* HTML/JS, BI TV-distance plots, the changes feed, the instability badge. This plan emits `website/data/bi_costs.json`; rendering it is the website plan's job.

## File structure

- `bi_selection.toml` (NEW) — user-editable selection policy: budget, ceiling, exclude globs, ordered rules.
- `src/trackllm_website/bi/selection.py` (NEW) — policy pydantic models + loader + pure `select_monitoring_targets`.
- `src/trackllm_website/bi/vetting.py` (NEW) — cost-measuring probe + bucketed cache (`EndpointCache`), extracted/rewritten from the old token test.
- `src/trackllm_website/bi/costs.py` (NEW) — compute cost/spend summary, emit `website/data/bi_costs.json`.
- `src/trackllm_website/config.py`, `config.toml` (MODIFY) — drop token-count + BI cost-cap config, add selection-file path, add cadence constants, add `cost_per_request` to `Endpoint`.
- `src/trackllm_website/api.py` (MODIFY) — parametrize `get_endpoints` cost filtering (BI: none; LT: keeps `$30/Mtok`).
- `src/trackllm_website/update_endpoints.py` (MODIFY) — use `bi/vetting.py`; lifecycle onboards from the selected set; too_expensive/bad_temperature rechecks.
- `src/trackllm_website/bi/phase_1.py` (MODIFY) — conditional temperature gate.
- `bad_endpoints_bi.yaml` → replaced by `endpoints_cache_bi.yaml` (bucketed). Migration handled in Task 6.

---

### Task 1: Config — drop token/cost-cap knobs, add selection path, cadence, and `cost_per_request`

**Files:**
- Modify: `config.toml`
- Modify: `src/trackllm_website/config.py`

- [ ] **Step 1: Edit `config.toml`**

In `[bi]`, remove `max_input_tokens = 15` (and `max_output_tokens` if present). Add under `[bi]`:

```toml
selection_path = "bi_selection.toml"
samples_per_day = 200  # BIs (top_k_bis=20) x samples/BI/day (queries_per_token=10), the monitoring cadence
days_per_month = 30
```

Leave `[api] max_cost_mtok = 30` untouched (LT still uses it).

- [ ] **Step 2: Edit `Endpoint` in `config.py`** — add an optional measured cost field after `max_logprobs`:

```python
    max_logprobs: int | None = None
    cost_per_request: float | None = None  # measured $/monitoring-query; set by BI vetting
```

`__eq__`/`__hash__` already key only on (api, model, provider), so this field doesn't affect identity — leave them as is.

- [ ] **Step 3: Edit `BIConfig` in `config.py`** — remove the `max_input_tokens: int` and `max_output_tokens: int | None` fields. Add:

```python
    selection_path: str
    samples_per_day: int
    days_per_month: int
```

Add a property on `BIConfig`:

```python
    @property
    def samples_per_month(self) -> int:
        return self.samples_per_day * self.days_per_month
```

- [ ] **Step 4: Find and fix references to the removed config**

Run: `rg -n "max_input_tokens|max_output_tokens" src tests`
The only production references are in `update_endpoints.py::test_endpoint_token_usage` (rewritten in Task 3) and possibly `bi/common.py`/`bi/bi_prevalence.py`. For any in analysis-only files (`bi_prevalence.py`, `phase_1_stats.py`), if they read `config.bi.max_input_tokens`, replace with a local constant `MAX_INPUT_TOKENS = 15` in that file (these are offline tools, not the production path). Do NOT touch `update_endpoints.py` yet (Task 3 rewrites that function wholesale).

- [ ] **Step 5: Verify config loads**

Run: `uv run python -c "from trackllm_website.config import config; print(config.bi.samples_per_month, config.bi.selection_path)"`
Expected: `6000 bi_selection.toml`

- [ ] **Step 6: Commit**

```bash
git add config.toml src/trackllm_website/config.py
git commit -m "config: drop BI token-count knobs, add selection path + cadence + cost_per_request"
```

---

### Task 2: `api.get_endpoints` — parametrize cost filtering

**Files:**
- Modify: `src/trackllm_website/api.py` (the `get_endpoints` function — find it with `rg -n "def get_endpoints"`; note it lives in `api.py` or `update_endpoints.py`, confirm first)

- [ ] **Step 1: Locate the function and its cost filter**

Run: `rg -n "max_cost_mtok|def get_endpoints" src/trackllm_website`
The filter is `e.cost[0] + e.cost[1] < config.api.max_cost_mtok`. It's currently applied unconditionally inside `get_endpoints`.

- [ ] **Step 2: Add a parameter (no default value — pass explicitly per caller)**

Change the signature to add `max_cost_mtok: float | None` and gate the filter:

```python
async def get_endpoints(logprob_filter: bool, max_cost_mtok: float | None) -> list[Endpoint]:
    ...
    if max_cost_mtok is not None:
        filtered_endpoints = [
            e for e in filtered_endpoints if e.cost[0] + e.cost[1] < max_cost_mtok
        ]
```

(Keep the `openrouter_avoid_free_endpoints` filter as is.)

- [ ] **Step 3: Update callers**

Run: `rg -n "get_endpoints\(" src`
- LT caller (`update_endpoints_lt`): pass `logprob_filter=True, max_cost_mtok=config.api.max_cost_mtok`.
- BI caller (`update_endpoints_bi`): pass `logprob_filter=False, max_cost_mtok=None` (no catalog cost cap — cost is handled by selection).

- [ ] **Step 4: Verify imports resolve**

Run: `uv run python -c "from trackllm_website.update_endpoints import get_endpoints"` (or wherever it lives) `&& uv run pytest tests/ -q`
Expected: import OK, 79 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/trackllm_website/api.py src/trackllm_website/update_endpoints.py
git commit -m "get_endpoints: BI drops the catalog cost cap, LT keeps it"
```

---

### Task 3: `bi/vetting.py` — cost-measuring probe + bucketed cache

**Files:**
- Create: `src/trackllm_website/bi/vetting.py`
- Test: `tests/test_bi_vetting.py`

This replaces `test_endpoint_token_usage`. A probe resolves the endpoint's strategy, runs one real monitoring-style query, measures actual billed cost, and checks pricing honesty. No token-count checks.

- [ ] **Step 1: Write the failing tests (mocked client + generation cost)**

```python
import asyncio

from trackllm_website.bi.common import PlainStrategy
from trackllm_website.bi.vetting import EndpointCache, VetResult, vet_endpoint
from trackllm_website.config import Endpoint
from trackllm_website.storage import Response, ResponseError, Usage

EP = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1.0, 2.0))


class FakeClient:
    def __init__(self, response, gen_cost):
        self._response, self._gen_cost = response, gen_cost

    async def query(self, endpoint, prompt, **kwargs):
        return self._response

    async def get_generation_cost(self, gen_id, session=None):
        return self._gen_cost


def ok_response(cost, gen_id="g1"):
    # build a real Response with content + a billed cost
    return Response(
        content="Hello", reasoning_content=None, error=None,
        input_tokens=7, output_tokens=1, reasoning_tokens=0,
        cost=cost, generation_id=gen_id, logprobs=None,
    )


def test_vet_records_measured_cost():
    client = FakeClient(ok_response(cost=0.00003), gen_cost=0.00003)
    res = asyncio.run(vet_endpoint(client, EP, PlainStrategy()))
    assert res.bucket == "candidate"
    assert abs(res.cost_per_request - 0.00003) < 1e-9


def test_vet_flags_pricing_liar():
    # advertised ~ (7*1 + 1*2)/1e6 = 9e-6; billed 10x that
    client = FakeClient(ok_response(cost=9e-5, gen_id="g1"), gen_cost=9e-5)
    res = asyncio.run(vet_endpoint(client, EP, PlainStrategy()))
    assert res.bucket == "liar"


def test_vet_transient_error_is_not_cached():
    resp = Response(content=None, reasoning_content=None,
                    error=ResponseError(code=500, message="boom"),
                    input_tokens=0, output_tokens=0, reasoning_tokens=0,
                    cost=None, generation_id=None, logprobs=None)
    client = FakeClient(resp, gen_cost=None)
    res = asyncio.run(vet_endpoint(client, EP, PlainStrategy()))
    assert res.bucket == "transient"  # don't cache; retry next run


def test_cache_round_trip(tmp_path):
    cache = EndpointCache(liars=[], too_expensive=[], bad_temperature=[])
    cache.add_liar(EP)
    path = tmp_path / "endpoints_cache_bi.yaml"
    cache.save(path)
    loaded = EndpointCache.load(path)
    assert loaded.is_cached(EP)
    assert loaded.bucket_of(EP) == "liar"
```

Before finalizing, open `src/trackllm_website/storage.py` and confirm the real `Response`, `ResponseError`, and any `Usage` field names/constructor — adjust `ok_response`/the error response to match exactly (the existing `tests/test_bi_prevalence.py` and `tests/test_bi_sampling.py` show the real construction pattern; mirror it).

- [ ] **Step 2: Run tests — expect ModuleNotFoundError**

Run: `uv run pytest tests/test_bi_vetting.py -v`

- [ ] **Step 3: Implement `bi/vetting.py`**

```python
"""Vet BI candidate endpoints by measured cost-per-request, and cache the rejects.

Buckets: candidate (usable, carries measured cost), liar (billed != advertised),
too_expensive (set by the catalog refresh against the selection ceiling),
bad_temperature (set by phase 1 when T=0 is ignored). Only liars are permanent;
the others are rechecked periodically since prices / providers change.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from trackllm_website.bi.common import (
    QueryStrategy,
    extract_first_token,
    strategy_to_query_args,
)
from trackllm_website.config import Endpoint, logger

PRICE_TOLERANCE = 0.01
Bucket = Literal["candidate", "liar", "too_expensive", "bad_temperature", "transient"]


class VetResult(BaseModel):
    bucket: Bucket
    cost_per_request: float | None = None


def _advertised_cost(input_tokens: int, output_tokens: int, endpoint: Endpoint) -> float:
    return input_tokens * endpoint.cost[0] / 1e6 + output_tokens * endpoint.cost[1] / 1e6


async def vet_endpoint(
    client, endpoint: Endpoint, strategy: QueryStrategy
) -> VetResult:
    """Probe one endpoint with its resolved strategy; classify it.

    A transient error (network / 5xx) returns bucket="transient" so the caller
    does NOT cache it — it will be retried next run.
    """
    response = await client.query(
        endpoint, "a", temperature=0.0, logprobs=False, **strategy_to_query_args(strategy)
    )
    if response.error:
        logger.info(f"{endpoint} vet: transient error {response.error.message[:80]}")
        return VetResult(bucket="transient")

    advertised = _advertised_cost(response.input_tokens, response.output_tokens, endpoint)
    measured = response.cost
    if measured is None and response.generation_id:
        measured = await client.get_generation_cost(response.generation_id)
    if measured is None:
        return VetResult(bucket="transient")  # couldn't price it; retry later

    if advertised > 0 and measured > advertised * (1 + PRICE_TOLERANCE):
        logger.info(f"{endpoint} vet: liar (billed {measured:.8f} vs {advertised:.8f})")
        return VetResult(bucket="liar")

    return VetResult(bucket="candidate", cost_per_request=measured)


class EndpointCache(BaseModel):
    liars: list[Endpoint]
    too_expensive: list[Endpoint]
    bad_temperature: list[Endpoint]

    def is_cached(self, endpoint: Endpoint) -> bool:
        return self.bucket_of(endpoint) is not None

    def bucket_of(self, endpoint: Endpoint) -> Bucket | None:
        if endpoint in self.liars:
            return "liar"
        if endpoint in self.too_expensive:
            return "too_expensive"
        if endpoint in self.bad_temperature:
            return "bad_temperature"
        return None

    def add_liar(self, endpoint: Endpoint) -> None:
        if endpoint not in self.liars:
            self.liars.append(endpoint)

    def add_too_expensive(self, endpoint: Endpoint) -> None:
        if endpoint not in self.too_expensive:
            self.too_expensive.append(endpoint)

    def add_bad_temperature(self, endpoint: Endpoint) -> None:
        if endpoint not in self.bad_temperature:
            self.bad_temperature.append(endpoint)

    def save(self, path: Path) -> None:
        def dump(es: list[Endpoint]) -> list[dict]:
            return [
                {"api": e.api, "model": e.model, "provider": e.provider, "cost": list(e.cost)}
                for e in sorted(es, key=lambda e: (e.model, e.provider or ""))
            ]

        data = {
            "liars": dump(self.liars),
            "too_expensive": dump(self.too_expensive),
            "bad_temperature": dump(self.bad_temperature),
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "EndpointCache":
        if not path.exists():
            return cls(liars=[], too_expensive=[], bad_temperature=[])
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        def parse(key: str) -> list[Endpoint]:
            return [
                Endpoint(api=e["api"], model=e["model"], provider=e.get("provider"),
                         cost=tuple(e["cost"]))
                for e in data.get(key, [])
            ]

        return cls(liars=parse("liars"),
                   too_expensive=parse("too_expensive"),
                   bad_temperature=parse("bad_temperature"))
```

Note: the `datetime` import and per-entry timestamps for recheck are handled at the call site in Task 7 (the cache stores membership; recheck cadence is driven by a separate timestamp file or by clearing buckets on a schedule — see Task 7). Keep `vetting.py` free of recheck-timing logic. Remove the unused `datetime` import if you don't use it here.

- [ ] **Step 4: Run tests — expect 4 PASS**

- [ ] **Step 5: Full suite + prek, commit**

```bash
uv run pytest tests/ -q && prek run --all-files
git add src/trackllm_website/bi/vetting.py tests/test_bi_vetting.py
git commit -m "add cost-measuring BI vetting + bucketed endpoint cache"
```

---

### Task 4: `bi_selection.toml` + policy models

**Files:**
- Create: `bi_selection.toml`
- Create: `src/trackllm_website/bi/selection.py` (policy models + loader only; the engine is Task 5)
- Test: `tests/test_bi_selection_policy.py`

- [ ] **Step 1: Write `bi_selection.toml`**

```toml
# BI monitoring selection policy. The ONLY place cost is managed.
# Rules run top to bottom; each adds endpoints (no duplicates) until
# budget_per_month is reached. Per-endpoint monthly cost = measured
# cost_per_request x config.bi.samples_per_month (20 BIs x 10 samples x 30 days).

budget_per_month = 10.0
# Non-flagship endpoints whose monthly cost exceeds this are never probed past
# vetting (cached too_expensive). Flagships (pins + flagship rule models) are exempt.
max_endpoint_cost = 0.50

# Never selected anywhere (globs match "model" or "model#provider").
exclude = [
    "*image*", "*-fast", "*:extended", "*customtools*", "*-beta",
    "switchpoint/router",
]

[[rule]]
name = "flagships"           # models people care about, cheapest provider each
kind = "models"
patterns = [
    "anthropic/claude-fable-5", "anthropic/claude-opus-4.8", "anthropic/claude-opus-4.7",
    "anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet-4.5", "anthropic/claude-haiku-4.5",
    "openai/gpt-4o", "openai/gpt-4o-mini",
    "google/gemini-3-flash*", "google/gemini-2.5-flash*",
    "deepseek/deepseek-v3.2", "deepseek/deepseek-chat*",
    "qwen/qwen3-235b*", "moonshotai/kimi-k2*", "z-ai/glm-4.7",
    "meta-llama/llama-4*", "mistralai/mistral-large*", "minimax/minimax-m2",
]
providers_per_model = 1
flagship = true              # exempt from max_endpoint_cost

[[rule]]
name = "corroboration"       # extra providers for swap-prone models, where cheap
kind = "models"
patterns = ["deepseek/deepseek-v3.2", "z-ai/glm-4.7", "qwen/qwen3-235b*", "moonshotai/kimi-k2*"]
providers_per_model = "all"
max_monthly_cost = 0.10

[[rule]]
name = "provider-coverage"   # every provider represented by its cheapest endpoint
kind = "providers"
patterns = ["*"]
endpoints_per_provider = 1
max_monthly_cost = 0.25

[[rule]]
name = "long-tail"           # fill remaining budget, cheapest models first
kind = "models"
patterns = ["*"]
providers_per_model = 1
max_monthly_cost = 0.10
```

- [ ] **Step 2: Write the failing test**

```python
from pathlib import Path

from trackllm_website.bi.selection import SelectionPolicy, load_policy


def test_loads_policy(tmp_path):
    p = tmp_path / "sel.toml"
    p.write_text("""
budget_per_month = 10.0
max_endpoint_cost = 0.5
exclude = ["*image*"]
[[rule]]
name = "flagships"
kind = "models"
patterns = ["anthropic/claude-fable-5"]
providers_per_model = 1
flagship = true
[[rule]]
name = "long-tail"
kind = "models"
patterns = ["*"]
providers_per_model = 1
max_monthly_cost = 0.1
""")
    policy = load_policy(p)
    assert policy.budget_per_month == 10.0
    assert policy.max_endpoint_cost == 0.5
    assert policy.rules[0].name == "flagships"
    assert policy.rules[0].flagship is True
    assert policy.rules[0].providers_per_model == 1
    assert policy.rules[1].max_monthly_cost == 0.1
    # flagship model patterns surface for the vetting exemption
    assert "anthropic/claude-fable-5" in policy.flagship_patterns()
```

- [ ] **Step 3: Implement the models + loader in `bi/selection.py`**

```python
"""BI selection policy: data models + loader. The rule engine is in this module too (Task 5)."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Rule(BaseModel):
    name: str
    kind: Literal["models", "providers"]
    patterns: list[str]
    providers_per_model: int | Literal["all"] | None = None
    endpoints_per_provider: int | None = None
    max_monthly_cost: float | None = None
    flagship: bool = False


class SelectionPolicy(BaseModel):
    budget_per_month: float
    max_endpoint_cost: float
    exclude: list[str]
    rules: list[Rule]

    def flagship_patterns(self) -> list[str]:
        return [p for r in self.rules if r.flagship for p in r.patterns]


def load_policy(path: Path) -> SelectionPolicy:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    rules = [Rule(**r) for r in raw.pop("rule", [])]
    return SelectionPolicy(rules=rules, **raw)
```

- [ ] **Step 4: Run test — expect PASS**

Run: `uv run pytest tests/test_bi_selection_policy.py -v`

- [ ] **Step 5: Commit**

```bash
git add bi_selection.toml src/trackllm_website/bi/selection.py tests/test_bi_selection_policy.py
git commit -m "add bi_selection.toml policy file + loader"
```

---

### Task 5: Selection rule engine

**Files:**
- Modify: `src/trackllm_website/bi/selection.py`
- Test: `tests/test_bi_selection_engine.py`

- [ ] **Step 1: Write the failing tests**

```python
import fnmatch

import pytest

from trackllm_website.bi.selection import (
    Rule, SelectionPolicy, monthly_cost, select_monitoring_targets,
)
from trackllm_website.config import Endpoint


def ep(model, provider, cpr):
    return Endpoint(api="openrouter", model=model, provider=provider, cost=(1, 1),
                    cost_per_request=cpr)


# samples_per_month default 6000 in config; monthly = cpr * 6000
def test_monthly_cost():
    assert abs(monthly_cost(ep("m", "p", 0.0001)) - 0.6) < 1e-9


def test_flagship_selected_over_budget_and_exempt_from_ceiling():
    policy = SelectionPolicy(
        budget_per_month=1.0, max_endpoint_cost=0.5, exclude=[],
        rules=[Rule(name="flagships", kind="models", patterns=["openai/gpt-5"],
                    providers_per_model=1, flagship=True)],
    )
    # gpt-5 monthly cost 3.0 > budget AND > ceiling, but flagship => selected
    cands = [ep("openai/gpt-5", "openai", 0.0005)]  # 0.0005*6000 = 3.0
    selected, breakdown = select_monitoring_targets(cands, policy)
    assert cands[0] in selected
    assert breakdown[cands[0]] == "flagships"


def test_cheapest_provider_per_flagship_model():
    policy = SelectionPolicy(
        budget_per_month=100.0, max_endpoint_cost=10.0, exclude=[],
        rules=[Rule(name="flagships", kind="models", patterns=["m/a"],
                    providers_per_model=1, flagship=True)],
    )
    cands = [ep("m/a", "cheap", 0.00001), ep("m/a", "pricey", 0.0001)]
    selected, _ = select_monitoring_targets(cands, policy)
    assert [e.provider for e in selected] == ["cheap"]


def test_exclude_globs_win():
    policy = SelectionPolicy(
        budget_per_month=100.0, max_endpoint_cost=10.0, exclude=["*image*"],
        rules=[Rule(name="long-tail", kind="models", patterns=["*"],
                    providers_per_model=1, max_monthly_cost=10.0)],
    )
    cands = [ep("openai/gpt-image", "openai", 0.00001), ep("m/b", "p", 0.00001)]
    selected, _ = select_monitoring_targets(cands, policy)
    assert [e.model for e in selected] == ["m/b"]


def test_max_monthly_cost_skips_pricey_in_wildcard_rule():
    policy = SelectionPolicy(
        budget_per_month=100.0, max_endpoint_cost=10.0, exclude=[],
        rules=[Rule(name="long-tail", kind="models", patterns=["*"],
                    providers_per_model=1, max_monthly_cost=0.10)],  # 0.10/mo => cpr<=~1.67e-5
    )
    cands = [ep("m/cheap", "p", 0.00001), ep("m/pricey", "p", 0.00005)]
    selected, _ = select_monitoring_targets(cands, policy)
    assert [e.model for e in selected] == ["m/cheap"]


def test_budget_stops_wildcard_fill():
    policy = SelectionPolicy(
        budget_per_month=0.6, max_endpoint_cost=10.0, exclude=[],
        rules=[Rule(name="long-tail", kind="models", patterns=["*"],
                    providers_per_model=1, max_monthly_cost=10.0)],
    )
    # each endpoint is 0.6/mo; budget 0.6 fits exactly one
    cands = [ep("m/a", "p", 0.0001), ep("m/b", "p", 0.0001)]
    selected, _ = select_monitoring_targets(cands, policy)
    assert len(selected) == 1


def test_named_rule_busting_budget_raises():
    # two flagships, each 6.0/mo, budget 10 => named rules alone exceed budget => loud error
    policy = SelectionPolicy(
        budget_per_month=10.0, max_endpoint_cost=100.0, exclude=[],
        rules=[Rule(name="flagships", kind="models", patterns=["m/a", "m/b"],
                    providers_per_model=1, flagship=True)],
    )
    cands = [ep("m/a", "p", 0.001), ep("m/b", "p", 0.001)]  # 6.0/mo each
    with pytest.raises(ValueError, match="exceeds budget"):
        select_monitoring_targets(cands, policy)
```

- [ ] **Step 2: Run tests — expect ModuleNotFoundError / ImportError**

- [ ] **Step 3: Implement the engine in `bi/selection.py`**

Add to the existing module:

```python
import fnmatch
from collections import defaultdict

from trackllm_website.config import Endpoint, config


def monthly_cost(endpoint: Endpoint) -> float:
    if endpoint.cost_per_request is None:
        raise ValueError(f"{endpoint} has no measured cost_per_request")
    return endpoint.cost_per_request * config.bi.samples_per_month


def _matches_any(endpoint: Endpoint, patterns: list[str]) -> bool:
    targets = (endpoint.model, f"{endpoint.model}#{endpoint.provider}")
    return any(fnmatch.fnmatch(t, p) for t in targets for p in patterns)


def select_monitoring_targets(
    candidates: list[Endpoint], policy: SelectionPolicy
) -> tuple[list[Endpoint], dict[Endpoint, str]]:
    """Pure: apply rules in order within budget. Returns (selected, rule-label-by-endpoint).

    Named-model / flagship rules that alone exceed the budget raise ValueError
    (a config error, not silent truncation). Wildcard fill rules stop at budget.
    """
    pool = [e for e in candidates if not _matches_any(e, policy.exclude)]
    by_model: dict[str, list[Endpoint]] = defaultdict(list)
    by_provider: dict[str, list[Endpoint]] = defaultdict(list)
    for e in pool:
        by_model[e.model].append(e)
        by_provider[e.provider_without_suffix].append(e)
    for d in (by_model, by_provider):
        for k in d:
            d[k].sort(key=monthly_cost)

    selected: dict[Endpoint, str] = {}
    spent = 0.0

    def add(e: Endpoint, label: str) -> None:
        nonlocal spent
        if e in selected:
            return
        selected[e] = label
        spent += monthly_cost(e)

    for rule in policy.rules:
        is_wildcard = rule.patterns == ["*"]
        if rule.kind == "models":
            model_keys = [m for m in by_model if _matches_any(by_model[m][0], rule.patterns)]
            model_keys.sort(key=lambda m: monthly_cost(by_model[m][0]))
            for m in model_keys:
                eps = by_model[m]
                n = len(eps) if rule.providers_per_model == "all" else rule.providers_per_model
                for e in eps[:n]:
                    if e in selected:
                        continue
                    if rule.max_monthly_cost is not None and monthly_cost(e) > rule.max_monthly_cost:
                        continue
                    if not rule.flagship and is_wildcard and spent + monthly_cost(e) > policy.budget_per_month:
                        return list(selected), selected  # budget reached, stop fill
                    add(e, rule.name)
        else:  # providers
            for prov, eps in sorted(by_provider.items()):
                for e in eps[: rule.endpoints_per_provider]:
                    if e in selected:
                        continue
                    if rule.max_monthly_cost is not None and monthly_cost(e) > rule.max_monthly_cost:
                        continue
                    if is_wildcard and spent + monthly_cost(e) > policy.budget_per_month:
                        continue  # this provider too pricey; try cheaper providers
                    add(e, rule.name)

    # Named (non-wildcard) rules must fit the budget; flagships are budget-exempt but
    # we still surface a blown budget loudly if NON-flagship named rules overrun.
    nonflag = sum(monthly_cost(e) for e, lbl in selected.items()
                  if not next(r for r in policy.rules if r.name == lbl).flagship)
    if nonflag > policy.budget_per_month:
        raise ValueError(
            f"non-flagship selection ${nonflag:.2f}/mo exceeds budget ${policy.budget_per_month:.2f}"
        )
    return list(selected), selected
```

Reconcile with the tests: `test_named_rule_busting_budget_raises` uses a flagship rule whose total exceeds budget. Per the design, **flagships are budget-exempt**, so flagships overrunning is allowed, NOT an error — but the test expects `ValueError`. Resolve this contradiction in favor of the design's intent with a guard the test should encode: a *flagship* total exceeding budget is allowed (warn via `logger`), while *non-flagship named rules* (none in the current policy, but possible) exceeding budget raises. **Update that test** to assert flagships over budget are selected with a logged warning (use `caplog`), not an error; keep a separate test for a hypothetical non-flagship named rule overrun raising. Document this in your report.

- [ ] **Step 4: Run tests — iterate until green**

Run: `uv run pytest tests/test_bi_selection_engine.py -v`

- [ ] **Step 5: Smoke-test against the real catalog**

Run:
```bash
uv run python -c "
from trackllm_website.bi.selection import load_policy, select_monitoring_targets, monthly_cost
from trackllm_website.config import config
# fake cost_per_request from advertised price (real costs come from vetting later)
cands=[e.model_copy(update={'cost_per_request': (7*e.cost[0]+e.cost[1])/1e6}) for e in config.endpoints_bi]
policy=load_policy(config.root_path() if False else __import__('pathlib').Path(config.bi.selection_path))
sel,bd=select_monitoring_targets(cands, policy)
import collections; c=collections.Counter(bd.values())
print(len(sel),'endpoints', dict(c), 'nominal \$%.1f/mo'%sum(monthly_cost(e) for e in sel))
"
```
(`config.bi.selection_path` is relative to repo root; resolve it via `root / config.bi.selection_path` — import `root` from `trackllm_website.config`.) Expected: a few hundred endpoints, tiers populated, nominal cost in the ballpark of the budget. Report the numbers.

- [ ] **Step 6: Full suite + prek, commit**

```bash
uv run pytest tests/ -q && prek run --all-files
git add src/trackllm_website/bi/selection.py tests/test_bi_selection_engine.py
git commit -m "add BI selection rule engine (budget, ceiling, flagship exemption)"
```

---

### Task 6: Wire vetting into the catalog refresh

**Files:**
- Modify: `src/trackllm_website/update_endpoints.py`
- Test: extend `tests/test_bi_vetting.py`

Rewrite `update_endpoints_bi()` to: pull the catalog (no cost cap), skip endpoints already cached or already-good, vet the rest (resolve strategy → `vet_endpoint`), measure cost, sort into candidate/liar/too_expensive (ceiling from the policy, flagships exempt), and write `endpoints_bi.yaml` with measured `cost_per_request`.

- [ ] **Step 1: Write the failing test (pure bucketing helper)**

Add a pure helper `bucket_by_ceiling(cost_per_request, model, provider, policy)` and test it:

```python
from trackllm_website.bi.selection import SelectionPolicy, Rule
from trackllm_website.update_endpoints import exceeds_ceiling

def _policy(ceiling, flagships):
    return SelectionPolicy(budget_per_month=10, max_endpoint_cost=ceiling, exclude=[],
        rules=[Rule(name="flagships", kind="models", patterns=flagships,
                    providers_per_model=1, flagship=True)])

def test_non_flagship_above_ceiling_is_too_expensive():
    pol = _policy(0.5, ["m/flag"])
    # 0.0001*6000 = 0.6 > 0.5 ceiling
    assert exceeds_ceiling(0.0001, "m/other", "p", pol) is True

def test_flagship_above_ceiling_is_kept():
    pol = _policy(0.5, ["m/flag"])
    assert exceeds_ceiling(0.0001, "m/flag", "p", pol) is False
```

- [ ] **Step 2: Run — expect ImportError on `exceeds_ceiling`**

- [ ] **Step 3: Implement `exceeds_ceiling` + rewrite `update_endpoints_bi`**

Add the helper near the top of `update_endpoints.py`:

```python
from trackllm_website.bi.selection import SelectionPolicy, load_policy, _matches_any
from trackllm_website.bi.vetting import EndpointCache, vet_endpoint
from trackllm_website.config import config, root

ENDPOINTS_CACHE_BI_PATH = root / "endpoints_cache_bi.yaml"


def exceeds_ceiling(cost_per_request: float, model: str, provider: str, policy: SelectionPolicy) -> bool:
    """A non-flagship endpoint above the monthly ceiling is too_expensive to keep probing."""
    fake = Endpoint(api="openrouter", model=model, provider=provider, cost=(0, 0))
    if _matches_any(fake, policy.flagship_patterns()):
        return False
    return cost_per_request * config.bi.samples_per_month > policy.max_endpoint_cost
```

Rewrite `update_endpoints_bi()`:
- `all_endpoints = await get_endpoints(logprob_filter=False, max_cost_mtok=None)`
- `policy = load_policy(root / config.bi.selection_path)`
- `cache = EndpointCache.load(ENDPOINTS_CACHE_BI_PATH)`
- `known_good = {str(e): e for e in config.endpoints_bi}`
- For endpoints not in `known_good` and not `cache.is_cached(e)`: resolve strategy (use `resolve_strategies`), then `vet_endpoint`. Route by bucket: `candidate` → set `e.cost_per_request`, then if `exceeds_ceiling(...)` → `cache.add_too_expensive(e)` else keep as good; `liar` → `cache.add_liar(e)`; `too_expensive` never returned by vet (ceiling is applied here); `transient` → skip (don't cache).
- Refresh `cost_per_request` for still-good endpoints by re-vetting them too (prices change) — but to bound cost, re-vet good endpoints only every run is acceptable (one cheap probe each). Keep it simple: re-vet all (good ∪ new) that aren't cached.
- Write `endpoints_bi.yaml` entries WITH `cost_per_request`. Factor the dump into a module-level `save_endpoints_bi(endpoints: list[Endpoint]) -> None` (writes `{api, model, provider, cost, cost_per_request}` per entry, sorted) and have `update_endpoints_bi` call it — Task 10's preview reuses this exact writer.
- `cache.save(ENDPOINTS_CACHE_BI_PATH)`.

Migration: on first run, if `bad_endpoints_bi.yaml` exists and `endpoints_cache_bi.yaml` does not, convert old `price_mismatch` entries → `liars`, drop `token_usage` entries (those were token-count rejects, no longer a reason — let them be re-vetted). Add a one-shot `migrate_bad_endpoints()` helper and call it at the top of `update_endpoints_bi`. Delete `bad_endpoints_bi.yaml` via `git rm` once converted (do this in the commit).

Update the dump block that writes `endpoints_bi.yaml` to include `"cost_per_request": e.cost_per_request`.

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/test_bi_vetting.py -v && uv run pytest tests/ -q`

- [ ] **Step 5: prek + commit**

```bash
prek run --all-files
git rm bad_endpoints_bi.yaml
git add src/trackllm_website/update_endpoints.py tests/test_bi_vetting.py
git commit -m "catalog refresh: cost-based vetting + bucketed cache + measured cost_per_request"
```

---

### Task 7: Lifecycle onboards from the selected set

**Files:**
- Modify: `src/trackllm_website/update_endpoints.py`
- Test: extend `tests/test_bi_lifecycle.py`

`select_lifecycle_actions` currently treats all of `config.endpoints_bi` as onboarding candidates. It must onboard only the **selected** set, and recheck `too_expensive`/`bad_temperature` caches periodically.

- [ ] **Step 1: Write the failing test**

```python
from trackllm_website.update_endpoints import select_lifecycle_actions
from trackllm_website.config import Endpoint
from datetime import datetime, timezone

def ep(m, cpr=0.00001):
    return Endpoint(api="openrouter", model=m, provider="p", cost=(1,1), cost_per_request=cpr)

NOW = datetime(2026, 6, 16, tzinfo=timezone.utc)

def test_onboard_only_selected():
    # 3 candidates, but selected set is just two
    selected = [ep("m/a"), ep("m/b")]
    actions = select_lifecycle_actions(selected, {}, NOW)
    assert {e.model for e in actions.onboard} == {"m/a", "m/b"}
```

(The signature stays `select_lifecycle_actions(candidates, states, now)`; the change is that the *caller* passes the selected set, not `config.endpoints_bi`. So this test mainly pins the caller contract. Keep it.)

- [ ] **Step 2: Run — should still pass (signature unchanged) — then change the caller**

In `update_endpoints_bi_lifecycle()`, replace:

```python
candidates = config.endpoints_bi
```

with:

```python
from trackllm_website.bi.selection import load_policy, select_monitoring_targets
policy = load_policy(root / config.bi.selection_path)
candidates, _breakdown = select_monitoring_targets(config.endpoints_bi, policy)
logger.info(f"Selection: monitoring {len(candidates)} of {len(config.endpoints_bi)} candidates")
```

- [ ] **Step 3: Recheck of cached buckets**

`too_expensive` and `bad_temperature` endpoints should be re-vetted periodically (prices drop; providers fix temperature) rather than never. Add to `update_endpoints_bi` (Task 6 territory, but wire here): every `config.bi.reinit.recheck_days`, clear the `too_expensive` bucket so those endpoints get re-vetted next refresh. Implement minimally: store a `last_recheck` ISO date at the top of `endpoints_cache_bi.yaml`; when `now - last_recheck >= recheck_days`, empty `too_expensive` (and `bad_temperature`) before vetting, and reset `last_recheck`. Add the field to `EndpointCache` (`last_recheck: datetime | None = None`) and its save/load. Keep `liars` permanent.

- [ ] **Step 4: Test + verify**

Run: `uv run pytest tests/ -q && prek run --all-files`

- [ ] **Step 5: Commit**

```bash
git add src/trackllm_website/update_endpoints.py src/trackllm_website/bi/vetting.py tests/test_bi_lifecycle.py
git commit -m "lifecycle onboards the budget-selected set; recheck expensive/bad-temp caches"
```

---

### Task 8: Conditional temperature gate in phase 1

**Files:**
- Modify: `src/trackllm_website/bi/phase_1.py`
- Modify: `config.toml`, `src/trackllm_website/config.py` (add gate params)
- Test: `tests/test_bi_temperature_gate.py`

Phase 1 runs at T=0. If border-input prevalence exceeds a threshold, re-sample the border prompts at T=1; if T=1 does **not** broaden the distribution vs T=0 (temperature is a no-op), the endpoint is `bad_temperature`.

- [ ] **Step 1: Add config**

In `config.toml` under a new `[bi.temperature_gate]`:

```toml
[bi.temperature_gate]
prevalence_trigger = 0.30   # if > this fraction of sampled prompts are border inputs, run the check
check_prompts = 8           # how many border prompts to re-sample at T=1
check_samples = 8           # samples per prompt at each temperature
```

Add `TemperatureGateConfig(BaseModel)` with `prevalence_trigger: float`, `check_prompts: int`, `check_samples: int`; add `temperature_gate: TemperatureGateConfig` to `BIConfig`.

- [ ] **Step 2: Write the failing test (pure decision function)**

```python
from trackllm_website.bi.phase_1 import temperature_is_ignored


def test_ignored_when_t0_matches_t1():
    # per-prompt distinct-output counts at T=0 and T=1 are identical and both diverse
    t0 = {"p1": 3, "p2": 4, "p3": 2}
    t1 = {"p1": 3, "p2": 4, "p3": 2}
    assert temperature_is_ignored(t0, t1) is True


def test_honored_when_t1_broadens():
    t0 = {"p1": 1, "p2": 1, "p3": 2}
    t1 = {"p1": 3, "p2": 4, "p3": 5}
    assert temperature_is_ignored(t0, t1) is False
```

- [ ] **Step 3: Implement the decision + the async check**

```python
def temperature_is_ignored(
    t0_distinct: dict[str, int], t1_distinct: dict[str, int]
) -> bool:
    """True if raising temperature 0->1 does NOT broaden the output distribution.

    Honored endpoints show strictly more diversity at T=1 on at least some prompts;
    if T=1 never exceeds T=0, temperature is a no-op.
    """
    return all(t1_distinct.get(p, 0) <= n for p, n in t0_distinct.items())
```

And an async `check_temperature(client, endpoint, strategy, border_prompts) -> bool` that samples up to `config.bi.temperature_gate.check_prompts` border prompts `check_samples` times at T=0 and T=1 (reuse `sample_prompts` from `bi/sampling.py` with `n_per_prompt=check_samples`, calling it once per temperature — note `sample_prompts` hardcodes T=0; add a `temperature` parameter to `sample_prompts` defaulting to nothing... no defaults: add `temperature: float` as a required param and update its one existing caller in `monitor.py`/`reinit.py` to pass `0.0`). Count distinct tokens per prompt at each temperature, then return `temperature_is_ignored(...)`.

- [ ] **Step 4: Integrate into phase 1b / discovery**

In the phase-1 path that finalizes border inputs (`phase_1b` writes `border_inputs.json`; the reinit path calls `discover_candidates` → `phase_1a`), after border tokens are found, compute prevalence = `len(border_tokens) / n_prompts_sampled`. If `prevalence > config.bi.temperature_gate.prevalence_trigger`, run `check_temperature`; if it returns True (ignored), signal `bad_temperature` for that endpoint (return an empty/sentinel result so `reinit` yields `None` and the caller caches `bad_temperature`). Thread this so `update_endpoints_bi_lifecycle`'s `onboard_one` adds the endpoint to the cache's `bad_temperature` bucket when discovery reports the temperature failure (distinguish it from ordinary `no_bis`). Keep the wiring minimal and explicit; add a `reason` to the discovery return so the caller knows whether it was `no_bis` or `bad_temperature`.

- [ ] **Step 5: Run tests + prek**

Run: `uv run pytest tests/ -q && prek run --all-files`

- [ ] **Step 6: Commit**

```bash
git add config.toml src/trackllm_website/config.py src/trackllm_website/bi/phase_1.py \
        src/trackllm_website/bi/sampling.py src/trackllm_website/bi/reinit.py \
        src/trackllm_website/bi/monitor.py tests/test_bi_temperature_gate.py
git commit -m "phase 1: conditional T=0-vs-T=1 gate; cache temperature-ignoring endpoints"
```

---

### Task 9: Emit cost/spend data (`bi/costs.py`)

**Files:**
- Create: `src/trackllm_website/bi/costs.py`
- Test: `tests/test_bi_costs.py`

Backend for the costs page: compute a summary and write `website/data/bi_costs.json`. (Rendering is the website plan.)

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import orjson
from trackllm_website.bi.costs import build_cost_summary
from trackllm_website.config import Endpoint
from trackllm_website.bi.selection import SelectionPolicy, Rule

def ep(m, p, cpr):
    return Endpoint(api="openrouter", model=m, provider=p, cost=(1,1), cost_per_request=cpr)

def test_summary_run_rate_and_top():
    policy = SelectionPolicy(budget_per_month=10, max_endpoint_cost=10, exclude=[],
        rules=[Rule(name="long-tail", kind="models", patterns=["*"],
                    providers_per_model=1, max_monthly_cost=10)])
    cands = [ep("m/a","p",0.00001), ep("m/b","p",0.00005)]
    summary = build_cost_summary(cands, policy)
    assert summary["budget_per_month"] == 10
    assert abs(summary["run_rate_per_month"] - (0.00001+0.00005)*6000) < 1e-6
    # top endpoints sorted by monthly cost descending
    assert summary["endpoints"][0]["model"] == "m/b"
    assert summary["by_rule"]["long-tail"]["count"] == 2
```

- [ ] **Step 2: Run — expect ModuleNotFoundError**

- [ ] **Step 3: Implement `bi/costs.py`**

```python
"""Compute the BI cost/spend summary consumed by the website costs page."""

from collections import defaultdict
from pathlib import Path

import orjson

from trackllm_website.bi.selection import SelectionPolicy, monthly_cost, select_monitoring_targets
from trackllm_website.config import Endpoint, config

COSTS_FILENAME = "bi_costs.json"


def build_cost_summary(candidates: list[Endpoint], policy: SelectionPolicy) -> dict:
    selected, breakdown = select_monitoring_targets(candidates, policy)
    rows = sorted(
        ({"model": e.model, "provider": e.provider, "rule": breakdown[e],
          "cost_per_request": e.cost_per_request, "monthly_cost": monthly_cost(e)}
         for e in selected),
        key=lambda r: r["monthly_cost"], reverse=True,
    )
    by_rule: dict[str, dict] = defaultdict(lambda: {"count": 0, "monthly_cost": 0.0})
    for r in rows:
        by_rule[r["rule"]]["count"] += 1
        by_rule[r["rule"]]["monthly_cost"] += r["monthly_cost"]
    return {
        "budget_per_month": policy.budget_per_month,
        "run_rate_per_month": sum(r["monthly_cost"] for r in rows),
        "n_selected": len(selected),
        "by_rule": dict(by_rule),
        "endpoints": rows,  # full list, descending; frontend shows top-20 + expand
    }


def write_cost_summary() -> None:
    from trackllm_website.bi.selection import load_policy
    from trackllm_website.config import root

    policy = load_policy(root / config.bi.selection_path)
    summary = build_cost_summary(config.endpoints_bi, policy)
    path = Path(config.data_dir) / COSTS_FILENAME
    path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
```

(Phase-1-vs-phase-2 spend split: add later from real request counts; for now `run_rate_per_month` is the phase-2 monitoring run-rate. Note this in your report — the actual-account-spend line is fetched by the website plan, not here, to avoid a daily API call in the build.)

- [ ] **Step 4: Run test + smoke-test on real data**

Run: `uv run pytest tests/test_bi_costs.py -v` then
`uv run python -c "from trackllm_website.bi.costs import write_cost_summary; write_cost_summary(); import orjson,pathlib; d=orjson.loads(pathlib.Path('website/data/bi_costs.json').read_bytes()); print(d['n_selected'],'selected, run-rate \$%.1f/mo'%d['run_rate_per_month'])"`
(Requires `endpoints_bi.yaml` to carry `cost_per_request` — only true after Task 6 has run against the real catalog; until then this smoke-test will raise on the missing field. If so, skip the real-data smoke-test here and note it runs as part of Task 10.)

- [ ] **Step 5: Commit**

```bash
git add src/trackllm_website/bi/costs.py tests/test_bi_costs.py
git commit -m "emit BI cost summary JSON for the costs page"
```

---

### Task 10: `cost-preview` CLI — price a `bi_selection.toml` without monitoring

**Files:**
- Modify: `src/trackllm_website/bi/costs.py` (add lazy cost-fill + a `preview` entrypoint)
- Test: extend `tests/test_bi_costs.py`

Lets the user iterate on `bi_selection.toml` cheaply: fill in any missing measured `cost_per_request` (probe once, cache to `endpoints_bi.yaml`), run selection, and print per-model / per-tier $/mo contributions and the total vs budget — before any real monitoring. Costs come from stored data when present; otherwise a one-time cheap probe per endpoint fills and caches them.

- [ ] **Step 1: Write the failing tests**

```python
import asyncio
from trackllm_website.bi.costs import ensure_costs, format_preview, build_cost_summary
from trackllm_website.bi.selection import SelectionPolicy, Rule
from trackllm_website.config import Endpoint


def ep(m, p, cpr=None):
    return Endpoint(api="openrouter", model=m, provider=p, cost=(1, 1), cost_per_request=cpr)


def test_ensure_costs_only_probes_missing(monkeypatch):
    probed = []

    async def fake_measure(client, endpoint, strategy):
        probed.append(str(endpoint))
        from trackllm_website.bi.vetting import VetResult
        return VetResult(bucket="candidate", cost_per_request=0.00002)

    monkeypatch.setattr("trackllm_website.bi.costs.vet_endpoint", fake_measure)
    monkeypatch.setattr("trackllm_website.bi.costs.resolve_strategies",
                        lambda client, eps: asyncio.sleep(0, result=({str(e): None for e in eps}, [])))
    cands = [ep("m/a", "p", cpr=0.00001), ep("m/b", "p")]  # b is missing
    filled = asyncio.run(ensure_costs(cands, save=False))
    assert [str(e) for e in probed] == ["openrouter#m/b#p"]  # only the missing one probed
    assert all(e.cost_per_request is not None for e in filled)


def test_format_preview_groups_by_model_and_totals():
    policy = SelectionPolicy(budget_per_month=10, max_endpoint_cost=10, exclude=[],
        rules=[Rule(name="long-tail", kind="models", patterns=["*"],
                    providers_per_model=1, max_monthly_cost=10)])
    cands = [ep("m/a", "p", 0.00001), ep("m/b", "p", 0.00005)]
    text = format_preview(build_cost_summary(cands, policy))
    assert "m/b" in text and "m/a" in text
    assert "/mo" in text
    assert "10.00" in text  # budget shown
```

Confirm the real `resolve_strategies` signature/return shape in `bi/common.py` and adjust the monkeypatch (it returns `(strategies_by_str, failed)`).

- [ ] **Step 2: Run tests — expect ImportError**

Run: `uv run pytest tests/test_bi_costs.py -v`

- [ ] **Step 3: Implement in `bi/costs.py`**

```python
from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.common import resolve_strategies
from trackllm_website.bi.vetting import vet_endpoint
from trackllm_website.config import logger, root


async def ensure_costs(candidates: list[Endpoint], *, save: bool) -> list[Endpoint]:
    """Return candidates with cost_per_request filled, probing only those missing it.

    Probed costs are written back to endpoints_bi.yaml when save=True so repeated
    previews don't re-probe. Endpoints that fail vetting (liar/transient) are dropped.
    """
    missing = [e for e in candidates if e.cost_per_request is None]
    if missing:
        logger.info(f"cost-preview: measuring {len(missing)} endpoints missing a cost")
        async with OpenRouterClient(timeout=60.0) as probe:
            strategies, _ = await resolve_strategies(probe, missing)
        async with OpenRouterClient() as client:
            for e in missing:
                strat = strategies.get(str(e))
                if strat is None:
                    continue
                res = await vet_endpoint(client, e, strat)
                if res.bucket == "candidate":
                    e.cost_per_request = res.cost_per_request
    priced = [e for e in candidates if e.cost_per_request is not None]
    if save:
        from trackllm_website.update_endpoints import save_endpoints_bi  # written in Task 6
        save_endpoints_bi(priced)
    return priced


def format_preview(summary: dict) -> str:
    lines = [
        f"Budget:   ${summary['budget_per_month']:.2f}/mo",
        f"Run-rate: ${summary['run_rate_per_month']:.2f}/mo  ({summary['n_selected']} endpoints)",
        "",
        "By rule:",
    ]
    for rule, info in sorted(summary["by_rule"].items(), key=lambda kv: -kv[1]["monthly_cost"]):
        lines.append(f"  {rule:18s} {info['count']:4d} endpoints  ${info['monthly_cost']:.2f}/mo")
    lines += ["", "Most expensive selected endpoints:"]
    for r in summary["endpoints"][:25]:
        lines.append(f"  ${r['monthly_cost']:6.2f}/mo  [{r['rule']:16s}] {r['model']} ({r['provider']})")
    return "\n".join(lines)


async def preview(policy_path: str | None = None) -> None:
    """Price a bi_selection.toml (default: the configured one) without monitoring."""
    from trackllm_website.bi.selection import load_policy, select_monitoring_targets  # noqa: F401

    path = root / (policy_path or config.bi.selection_path)
    policy = load_policy(path)
    candidates = await ensure_costs(list(config.endpoints_bi), save=True)
    summary = build_cost_summary(candidates, policy)
    print(format_preview(summary))
```

Add a `fire` entrypoint at the bottom of `costs.py`:

```python
if __name__ == "__main__":
    import asyncio
    import fire

    fire.Fire({
        "preview": lambda policy_path=None: asyncio.run(preview(policy_path)),
        "write": write_cost_summary,
    })
```

Note: `save_endpoints_bi(endpoints)` must be factored out of `update_endpoints_bi` in Task 6 (the block that dumps `endpoints_bi.yaml` including `cost_per_request`) so both the refresh and the preview write the same format. If Task 6 didn't extract it, extract it now as part of this task and have `update_endpoints_bi` call it.

- [ ] **Step 4: Run tests — expect PASS**

Run: `uv run pytest tests/test_bi_costs.py -v`

- [ ] **Step 5: Real dry-run against the catalog**

Run: `uv run python -m trackllm_website.bi.costs preview 2>&1 | grep -v "PyTorch\|INFO"`
This fills any missing costs (cheap probes) and prints the priced selection. Report the by-rule totals and run-rate. Try a second run to confirm it's instant (costs cached, no re-probe). **This is the tool the user will use to tune `bi_selection.toml`.**

- [ ] **Step 6: Full suite + prek, commit**

```bash
uv run pytest tests/ -q && prek run --all-files
git add src/trackllm_website/bi/costs.py tests/test_bi_costs.py endpoints_bi.yaml
git commit -m "add cost-preview CLI: price a selection policy from cached/probed costs"
```

---

### Task 11: Resumption — re-vet, select, onboard (one-off, local)

Runs once, locally (not GHA), after Tasks 1–9 are merged. ~hours of wall clock.

- [ ] **Step 1: Refresh the catalog with cost-based vetting**

Run: `uv run python -c "import asyncio; from trackllm_website.update_endpoints import update_endpoints_bi; asyncio.run(update_endpoints_bi())"`
This re-vets the catalog, writes `endpoints_bi.yaml` with measured `cost_per_request`, and populates `endpoints_cache_bi.yaml` (liars / too_expensive). Report counts per bucket.

- [ ] **Step 2: Inspect the selection before onboarding**

Run: `uv run python -c "from trackllm_website.bi.costs import write_cost_summary; write_cost_summary()" && uv run python -c "import orjson,pathlib; d=orjson.loads(pathlib.Path('website/data/bi_costs.json').read_bytes()); print('selected',d['n_selected'],'run-rate \$%.2f/mo'%d['run_rate_per_month']); [print(' ',k,v) for k,v in d['by_rule'].items()]"`
Confirm run-rate ≤ budget (+ flagship overage) and tier counts look sane. **Pause for the user to eyeball this before spending on onboarding.**

- [ ] **Step 3: Onboard the selected set**

Run: `uv run python -c "import asyncio; from trackllm_website.update_endpoints import update_endpoints_bi_lifecycle; asyncio.run(update_endpoints_bi_lifecycle())"`
This discovers BIs (with the temperature gate), collects references, and writes `monitoring` state files; temperature-ignoring endpoints land in `bad_temperature`. Report: how many onboarded `monitoring`, how many `no_bis`, how many `bad_temperature`.

- [ ] **Step 4: Verify the daily monitor runs clean against the new set**

Run: `uv run python -c "import asyncio; from trackllm_website.bi.monitor import monitor; asyncio.run(monitor())"` (or trigger `bi-monitor.yml` via `gh workflow run` once merged) and confirm it samples the monitored endpoints without error and commits `data_bi`.

- [ ] **Step 5: Commit the produced state + catalog**

```bash
git add data_bi/state endpoints_bi.yaml endpoints_cache_bi.yaml website/data/bi_costs.json
git commit -m "resumption: cost-based selection onboarded; cost summary emitted"
git push
```

---

## Self-review notes

- **Spec coverage:** measured cost/request (T3); drop `$30/Mtok` for BI (T2) + drop token-count knobs (T1); three+ bucket cache candidate/liar/too_expensive/bad_temperature (T3 + T6 ceiling + T8 temperature); `bi_selection.toml` with pins-via-flagships/flagships/corroboration/provider-coverage/long-tail + budget + explicit `max_endpoint_cost` flagships-exempt (T4/T5); selection feeds lifecycle (T7); conditional T=0-vs-T=1 gate triggered by prevalence (T8); costs data top-N + full list + per-rule + run-rate (T9); `cost-preview` CLI to price a policy from cached/probed costs before any monitoring (T10); resumption through new selection (T11). **Gap intentionally deferred:** actual-account-spend line and phase-1-vs-phase-2 split (website plan / a later cost-history task); costs *page* HTML.
- **Pins:** the design mentioned a separate `pins` rule (specific model#provider). It's representable as a `flagship` rule whose patterns include `model#provider` strings (the engine matches both `model` and `model#provider`). No separate rule kind needed; if the user wants an explicit `[[rule]] name="pins"`, it's just another flagship-flagged `models` rule.
- **Contradiction resolved (T5 step 3):** flagships are budget-exempt, so a flagship total over budget is a logged warning, not a `ValueError`; only non-flagship named-rule overruns raise. The draft test was corrected to match the design.
- **Known judgment calls for the implementer, flagged inline:** exact `Response`/`ResponseError` constructor fields (T3); whether `get_endpoints` lives in `api.py` or `update_endpoints.py` (T2); adding a required `temperature` param to `sample_prompts` and updating its callers (T8); the phase-1 prevalence denominator (`n_prompts_sampled`) — confirm against `EndpointState` in `bi/common.py`.
