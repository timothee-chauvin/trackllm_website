# BI/LT Production Change Detection (Backend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved design (docs/superpowers/specs/2026-06-11-bi-lt-change-detection-design.md): BI epoch state, adaptive change detection, daily monitoring with re-init, endpoint lifecycle, and LT score/event automation. Website display is a separate follow-up plan.

**Architecture:** State files store only facts (samples, references, actions); all detection output is derived by pure functions in `bi/detection.py`. A daily `bi/monitor.py` samples BIs, applies the adaptive rule, and triggers hybrid re-init via `bi/reinit.py`. Lifecycle (onboard/retire/resurrect) extends `update_endpoints.py`. LT change events get a stable append-only log.

**Tech Stack:** Python 3.13, uv, pydantic (+pydantic-settings), orjson, pytest, asyncio + aiolimiter, GitHub Actions.

**Conventions (from CLAUDE.md / user prefs):** tests first; `uv run` everything; `prek run --all-files` after edits; `slugify` for output filenames; no default argument values (config or module constants are the single source of truth); never silence errors.

---

### Task 1: Config — new detection/re-init parameters

**Files:**
- Modify: `config.toml`
- Modify: `src/trackllm_website/config.py`

- [ ] **Step 1: Add new sections/values to `config.toml`**

Replace the `[bi.phase_1]` value `target_border_inputs = 30` with `50`, replace `[bi.phase_2]` value `queries_per_token = 20` with `10`, and append:

```toml
[bi.detection]
window = 14
exclusion = 4
min_baseline = 5
sigma_k = 4.0
abs_delta = 0.2
persistence = 3
cooldown = 10
instability_window = 14
instability_threshold = 0.4

[bi.reinit]
reprobe_samples = 10
reference_samples = 100
top_k_bis = 20
min_bis = 10
stall_days = 7
recheck_days = 14
max_onboard_per_run = 10
```

- [ ] **Step 2: Add the pydantic models in `config.py`**

After `PrevalenceConfig`:

```python
class DetectionConfig(BaseModel):
    window: int
    exclusion: int
    min_baseline: int
    sigma_k: float
    abs_delta: float
    persistence: int
    cooldown: int
    instability_window: int
    instability_threshold: float


class ReinitConfig(BaseModel):
    reprobe_samples: int
    reference_samples: int
    top_k_bis: int
    min_bis: int
    stall_days: int
    recheck_days: int
    max_onboard_per_run: int
```

In `BIConfig`, add fields `detection: DetectionConfig` and `reinit: ReinitConfig`, plus:

```python
    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"
```

- [ ] **Step 3: Verify config loads**

Run: `uv run python -c "from trackllm_website.config import config; print(config.bi.detection.sigma_k, config.bi.reinit.top_k_bis, config.bi.state_dir)"`
Expected: `4.0 20 data_bi/state`

- [ ] **Step 4: Commit**

```bash
git add config.toml src/trackllm_website/config.py
git commit -m "add BI detection and reinit config sections"
```

---

### Task 2: Real-data test fixtures

**Files:**
- Create: `tests/fixtures/phase_2/...` (generated)
- Create: `tests/fixtures/make_fixtures.py`

- [ ] **Step 1: Write the fixture extraction script**

`tests/fixtures/make_fixtures.py` — extracts 4 representative endpoints from real data (clean change, unstable, stable, change-then-death), trimmed to keep fixtures small. The reference batch keeps all samples; other days keep 10.

```python
"""One-off: extract real-data fixtures for tests. Run from repo root."""

import random
from pathlib import Path

import orjson

from trackllm_website.bi.analyze import load_phase2_results
from trackllm_website.config import config

FIXTURE_DIR = Path("tests/fixtures/phase_2")
SLUGS = [
    "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8",  # clean change 2026-01-24
    "qwen2fqwen3-235b-a22b-250723wandb2fbf16",  # unstable (TV~0.47 from day 2)
    "openai2fgpt-4o-mini23azure",  # stable throughout
    "mistralai2fmistral-7b-instruct-v0.323together",  # change 2026-01-30 then death
]
MAX_PROMPTS = 20
MAX_SAMPLES = 10
LAST_DAY = "2026-03-15"


def main() -> None:
    rng = random.Random(0)
    for slug in SLUGS:
        results = load_phase2_results(config.bi.phase_2_dir / slug)
        prompts = sorted(results)[:MAX_PROMPTS]
        ref_ts = min(ts for p in prompts for ts in results[p])
        out = {}
        for p in prompts:
            out[p] = {}
            for ts, samples in results[p].items():
                if ts[:10] > LAST_DAY:
                    continue
                if ts != ref_ts and len(samples) > MAX_SAMPLES:
                    samples = rng.sample(samples, MAX_SAMPLES)
                out[p][ts] = samples
        dest = FIXTURE_DIR / slug
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "data.json").write_bytes(orjson.dumps(out))
        print(slug, sum(len(b) for b in out.values()), "batches")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate and sanity-check the fixtures**

Run: `uv run python tests/fixtures/make_fixtures.py && du -sh tests/fixtures/phase_2`
Expected: 4 lines with batch counts, total size ≲ 2 MB.

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures
git commit -m "add real-data phase 2 fixtures for BI tests"
```

---

### Task 3: `bi/state.py` — epoch state model

**Files:**
- Create: `src/trackllm_website/bi/state.py`
- Test: `tests/test_bi_state.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime, timezone

from trackllm_website.bi.state import Epoch, EndpointBIState, RetiredInfo
from trackllm_website.config import Endpoint


def make_state() -> EndpointBIState:
    return EndpointBIState(
        endpoint=Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 2)),
        status="monitoring",
        epochs=[
            Epoch(
                start=datetime(2026, 1, 14, tzinfo=timezone.utc),
                border_inputs=["a", "b"],
                reference={"a": [["2026-01-14T00:00:00+00:00", "tok"]]},
            )
        ],
    )


def test_round_trip(tmp_path):
    state = make_state()
    state.save(tmp_path)
    loaded = EndpointBIState.load(tmp_path / f"{state.slug}.json")
    assert loaded == state


def test_current_epoch_open_and_closed():
    state = make_state()
    assert state.current_epoch is state.epochs[0]
    state.epochs[0].end = datetime(2026, 2, 1, tzinfo=timezone.utc)
    state.epochs[0].end_reason = "change_detected"
    assert state.current_epoch is None


def test_retired_requires_info():
    state = make_state()
    state.status = "retired"
    state.retired = RetiredInfo(
        reason="stalled",
        since=datetime(2026, 2, 1, tzinfo=timezone.utc),
        last_recheck=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    assert state.retired.reason == "stalled"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bi_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'trackllm_website.bi.state'`

- [ ] **Step 3: Implement `bi/state.py`**

```python
"""Per-endpoint BI monitoring state: epochs of (border inputs, reference samples).

State files record only facts (samples, references, actions taken). Anything
the detection algorithm computes is derived elsewhere and never persisted.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal

import orjson
from pydantic import BaseModel

from trackllm_website.config import Endpoint, config
from trackllm_website.util import slugify

ReferenceSamples = dict[str, list[tuple[str, str]]]  # prompt -> [(timestamp, token)]


class Epoch(BaseModel):
    start: datetime
    border_inputs: list[str]
    reference: ReferenceSamples
    end: datetime | None = None
    end_reason: Literal["change_detected", "stalled", "gap"] | None = None
    change_date: datetime | None = None
    params: dict | None = None  # detection params in force when the epoch closed


class RetiredInfo(BaseModel):
    reason: Literal["stalled", "no_bis", "delisted"]
    since: datetime
    last_recheck: datetime


class EndpointBIState(BaseModel):
    endpoint: Endpoint
    status: Literal["monitoring", "retired"]
    retired: RetiredInfo | None = None
    epochs: list[Epoch]

    @property
    def slug(self) -> str:
        return slugify(f"{self.endpoint.model}#{self.endpoint.provider}")

    @property
    def current_epoch(self) -> Epoch | None:
        if self.epochs and self.epochs[-1].end is None:
            return self.epochs[-1]
        return None

    def save(self, state_dir: Path) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / f"{self.slug}.json"
        path.write_bytes(
            orjson.dumps(self.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
        )

    @classmethod
    def load(cls, path: Path) -> "EndpointBIState":
        return cls.model_validate(orjson.loads(path.read_bytes()))


def load_all_states(state_dir: Path) -> dict[str, EndpointBIState]:
    if not state_dir.exists():
        return {}
    return {
        p.stem: EndpointBIState.load(p) for p in sorted(state_dir.glob("*.json"))
    }
```

(Callers pass `config.bi.state_dir` explicitly — no default arguments.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bi_state.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/trackllm_website/bi/state.py tests/test_bi_state.py
git commit -m "add BI epoch state model"
```

---

### Task 4: `bi/detection.py` — pure detection functions

**Files:**
- Create: `src/trackllm_website/bi/detection.py`
- Modify: `src/trackllm_website/bi/adaptive_rule.py` (delegate to detection.py)
- Test: `tests/test_bi_detection.py`

- [ ] **Step 1: Write the failing tests (real-data fixtures)**

```python
from collections import Counter
from pathlib import Path

import orjson
import pytest

from trackllm_website.bi.detection import (
    adaptive_transitions,
    epoch_tv_series,
    is_unstable,
    select_top_bis,
)

FIXTURES = Path("tests/fixtures/phase_2")


def load_fixture(slug: str) -> dict:
    return orjson.loads((FIXTURES / slug / "data.json").read_bytes())


def reference_from_first_batch(results: dict) -> dict[str, list]:
    ref_ts = min(ts for batches in results.values() for ts in batches)
    return {p: b[ref_ts] for p, b in results.items() if ref_ts in b and b[ref_ts]}


def series(slug: str):
    results = load_fixture(slug)
    return epoch_tv_series(reference_from_first_batch(results), results)


def test_detects_hyperbolic_deepseek_change():
    tv = series("deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8")
    events = adaptive_transitions(tv)
    assert [e[:10] for e in events] == ["2026-01-24"]


def test_stable_endpoint_no_events():
    tv = series("openai2fgpt-4o-mini23azure")
    assert adaptive_transitions(tv) == []
    assert not is_unstable(tv)


def test_unstable_endpoint_flagged_not_fired():
    tv = series("qwen2fqwen3-235b-a22b-250723wandb2fbf16")
    assert adaptive_transitions(tv) == []
    assert is_unstable(tv)


def test_select_top_bis_by_balance():
    reference = {
        "balanced": [["t", "a"], ["t", "b"], ["t", "a"], ["t", "b"]],
        "skewed": [["t", "a"], ["t", "a"], ["t", "a"], ["t", "b"]],
        "dirac": [["t", "a"], ["t", "a"], ["t", "a"], ["t", "a"]],
    }
    assert select_top_bis(reference, 2) == ["balanced", "skewed"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bi_detection.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `bi/detection.py`**

Move `adaptive_transitions` from `bi/adaptive_rule.py`, with parameters read from `config.bi.detection` instead of module constants; add the rest:

```python
"""Pure change-detection functions on BI sampling data (no I/O, no state)."""

import statistics
from collections import Counter

from trackllm_website.bi.analyze import compute_tv_distance, get_distribution
from trackllm_website.bi.phase_2 import Timestamp
from trackllm_website.config import config


def epoch_tv_series(
    reference: dict[str, list], results: dict
) -> list[tuple[Timestamp, float]]:
    """Daily mean TV of each batch vs the epoch reference, for reference prompts.

    `results` is the phase 2 dict {prompt: {timestamp: [(ts, token), ...]}}.
    Days at or before the reference batch are excluded.
    """
    ref_dists = {p: get_distribution(samples) for p, samples in reference.items()}
    ref_day = max(ts[:10] for batches in results.values() for ts in batches
                  if any(s == samples for p2, samples in reference.items()
                         for s in [batches.get(ts)])) if False else None
    all_ts = sorted({ts for p in ref_dists for ts in results.get(p, {})})
    if not all_ts:
        return []
    ref_ts = all_ts[0]
    out: list[tuple[Timestamp, float]] = []
    for ts in all_ts[1:]:
        tvs = []
        for p, ref_dist in ref_dists.items():
            samples = results.get(p, {}).get(ts)
            if not samples:
                continue
            tv = compute_tv_distance(ref_dist, get_distribution(samples))
            if tv is not None:
                tvs.append(tv)
        if tvs:
            out.append((ts, statistics.mean(tvs)))
    return out


def adaptive_transitions(
    tv_over_time: list[tuple[Timestamp, float]],
) -> list[Timestamp]:
    """Sustained deviations from a trailing baseline (see design doc).

    A day deviates when |tv - baseline_mean| exceeds both abs_delta and
    sigma_k * baseline_std; an event fires after `persistence` consecutive
    deviating days, dated at the first one (onset). The baseline excludes the
    most recent `exclusion` days.
    """
    d = config.bi.detection
    timestamps = [ts for ts, _ in tv_over_time]
    vals = [v for _, v in tv_over_time]
    events: list[Timestamp] = []
    streak = 0
    last_event_idx: int | None = None

    for i in range(len(vals)):
        baseline = vals[max(0, i - d.exclusion - d.window) : i - d.exclusion]
        if len(baseline) < d.min_baseline:
            continue
        mean = statistics.mean(baseline)
        std = statistics.stdev(baseline)
        dev = abs(vals[i] - mean)
        if dev > d.abs_delta and dev > d.sigma_k * std:
            streak += 1
        else:
            streak = 0
        if streak == d.persistence:
            onset = i - d.persistence + 1
            if last_event_idx is None or onset - last_event_idx >= d.cooldown:
                events.append(timestamps[onset])
                last_event_idx = onset
    return events


def is_unstable(tv_over_time: list[tuple[Timestamp, float]]) -> bool:
    """Median TV over the trailing window exceeds the instability threshold."""
    d = config.bi.detection
    if not tv_over_time:
        return False
    tail = [v for _, v in tv_over_time[-d.instability_window :]]
    return statistics.median(tail) >= d.instability_threshold


def balance_score(dist: Counter) -> float:
    counts = sorted(dist.values(), reverse=True)
    return counts[1] / counts[0] if len(counts) > 1 else 0.0


def select_top_bis(reference: dict[str, list], k: int) -> list[str]:
    """Top-k prompts by top-2 balance (p2/p1) of their reference distribution."""
    scored = sorted(
        reference,
        key=lambda p: balance_score(get_distribution(reference[p])),
        reverse=True,
    )
    return scored[:k]
```

Note: remove the dead `ref_day = ... if False else None` line — it must not appear in the final file (shown here only to flag that no reference-day heuristic beyond "first timestamp" is wanted).

In `bi/adaptive_rule.py`: delete `adaptive_transitions` and the module constants `WINDOW/EXCLUSION/MIN_BASELINE/SIGMA_K/ABS_DELTA/PERSISTENCE/COOLDOWN`, import `adaptive_transitions` from `trackllm_website.bi.detection`.

- [ ] **Step 4: Run tests; iterate on the fixture expectations only if the dates differ by ±1 day** (fixture subsampling can shift onset by one day; the hyperbolic event must stay in Jan 23–25)

Run: `uv run pytest tests/test_bi_detection.py -v`
Expected: 4 PASS

- [ ] **Step 5: Run the full suite and prek**

Run: `uv run pytest -q && prek run --all-files`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/trackllm_website/bi/detection.py src/trackllm_website/bi/adaptive_rule.py tests/test_bi_detection.py
git commit -m "add pure BI detection module (adaptive rule, instability, BI ranking)"
```

---

### Task 5: Migration — synthesize epoch 0 from existing data

**Files:**
- Create: `src/trackllm_website/bi/migrate_state.py`
- Test: `tests/test_bi_migrate.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import orjson

from trackllm_website.bi.migrate_state import migrate_endpoint
from trackllm_website.config import Endpoint

FIXTURES = Path("tests/fixtures/phase_2")


def test_migrate_builds_closed_gap_epoch():
    slug = "mistralai2fmistral-7b-instruct-v0.323together"
    results = orjson.loads((FIXTURES / slug / "data.json").read_bytes())
    endpoint = Endpoint(
        api="openrouter", model="mistralai/mistral-7b-instruct-v0.3",
        provider="together", cost=(0.2, 0.2),
    )
    state = migrate_endpoint(endpoint, results)
    assert state.status == "retired"  # all history ends pre-resumption
    [epoch] = state.epochs
    assert epoch.end_reason == "gap"
    assert epoch.start.date().isoformat() == "2026-01-14"
    # end = last day with any successful sample
    assert epoch.end.date().isoformat() == "2026-02-25"
    assert set(epoch.reference) == set(epoch.border_inputs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bi_migrate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `bi/migrate_state.py`**

```python
"""One-off migration: build epoch-0 state files from existing phase 2 data.

Every historical endpoint gets a single closed epoch (end_reason="gap") whose
reference is the first batch, ending at its last day with a successful sample.
Status is retired(stalled) for all: resumption re-onboards live endpoints
fresh (design: January references are stale after the outage).
"""

from datetime import datetime, timezone

import fire

from trackllm_website.bi.analyze import load_phase2_results
from trackllm_website.bi.state import EndpointBIState, Epoch, RetiredInfo
from trackllm_website.config import config, logger
from trackllm_website.util import endpoint_from_slug


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def migrate_endpoint(endpoint, results) -> EndpointBIState:
    all_ts = sorted({ts for b in results.values() for ts in b})
    ref_ts = all_ts[0]
    reference = {p: b[ref_ts] for p, b in results.items() if b.get(ref_ts)}
    last_success = max(
        (ts for p, b in results.items() for ts, samples in b.items() if samples),
        default=ref_ts,
    )
    now = datetime.now(tz=timezone.utc)
    return EndpointBIState(
        endpoint=endpoint,
        status="retired",
        retired=RetiredInfo(reason="stalled", since=_parse(last_success), last_recheck=now),
        epochs=[
            Epoch(
                start=_parse(ref_ts),
                border_inputs=sorted(reference),
                reference=reference,
                end=_parse(last_success),
                end_reason="gap",
            )
        ],
    )


def migrate() -> None:
    n = 0
    for d in sorted(config.bi.phase_2_dir.iterdir()):
        if not d.is_dir():
            continue
        results = load_phase2_results(d)
        if not results:
            logger.warning(f"{d.name}: no results, skipping")
            continue
        state = migrate_endpoint(endpoint_from_slug(d.name), results)
        state.save(config.bi.state_dir)
        n += 1
    logger.info(f"Migrated {n} endpoints to {config.bi.state_dir}")


if __name__ == "__main__":
    fire.Fire(migrate)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bi_migrate.py -v`
Expected: PASS

- [ ] **Step 5: Run the migration on real data and inspect one state file**

Run: `uv run python -m trackllm_website.bi.migrate_state && uv run python -c "
from trackllm_website.bi.state import EndpointBIState
from trackllm_website.config import config
s = EndpointBIState.load(config.bi.state_dir / 'openai2fgpt-4o-mini23azure.json')
print(s.status, s.epochs[0].end_reason, len(s.epochs[0].border_inputs))"`
Expected: `Migrated 53 endpoints`, then `retired gap <n_bis>`

- [ ] **Step 6: Commit (code, tests, and generated state files)**

```bash
git add src/trackllm_website/bi/migrate_state.py tests/test_bi_migrate.py data_bi/state
git commit -m "migrate existing phase 2 history to epoch state files"
```

---

### Task 6: `bi/sampling.py` — shared async sampler

**Files:**
- Create: `src/trackllm_website/bi/sampling.py`
- Test: `tests/test_bi_sampling.py`

Used by the monitor (daily batches), re-probe, and reference collection. Wraps the query/strategy/rate-limit machinery of `phase_2.py` into one reusable coroutine, decoupled from monthly files.

- [ ] **Step 1: Write the failing test (fake client)**

```python
import asyncio

from trackllm_website.bi.common import PlainStrategy
from trackllm_website.bi.sampling import sample_prompts
from trackllm_website.config import Endpoint


class FakeResponse:
    def __init__(self, content):
        self.error = None
        self.content = content
        self.reasoning = None


class FakeClient:
    def __init__(self, answers):
        self.answers = answers  # prompt -> iterator of tokens

    async def query(self, endpoint, prompt, **kwargs):
        return FakeResponse(next(self.answers[prompt]))


def test_sample_prompts_collects_n_per_prompt():
    endpoint = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))
    client = FakeClient({"a": iter("xyxyx"), "b": iter("zzzzz")})
    samples, n_errors = asyncio.run(
        sample_prompts(client, endpoint, PlainStrategy(), ["a", "b"], 3)
    )
    assert n_errors == 0
    assert [tok for _, tok in samples["a"]] == ["x", "y", "x"]
    assert len(samples["b"]) == 3
```

(Adjust `FakeResponse` attributes to whatever `extract_first_token` reads — check `bi/common.py` while implementing; the fake must satisfy it.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bi_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `bi/sampling.py`**

```python
"""Reusable BI sampler: query an endpoint's prompts n times each at T=0."""

import asyncio
from datetime import datetime, timezone

from aiolimiter import AsyncLimiter

from trackllm_website.bi.common import (
    QueryStrategy,
    extract_first_token,
    strategy_to_query_args,
)
from trackllm_website.config import Endpoint, config, logger


async def sample_prompts(
    client,
    endpoint: Endpoint,
    strategy: QueryStrategy,
    prompts: list[str],
    n_per_prompt: int,
) -> tuple[dict[str, list[tuple[str, str]]], int]:
    """Returns ({prompt: [(timestamp, token), ...]}, n_errors).

    Respects the phase 2 rate limits from config. Empty responses count as
    successes with no sample (consistent with phase_2.py).
    """
    cfg = config.bi.phase_2
    limiter = AsyncLimiter(cfg.requests_per_second_per_endpoint, 1)
    semaphore = asyncio.Semaphore(cfg.max_concurrent_requests_per_endpoint)
    samples: dict[str, list[tuple[str, str]]] = {p: [] for p in prompts}
    n_errors = 0

    async def one(prompt: str) -> None:
        nonlocal n_errors
        for i in range(n_per_prompt):
            async with semaphore:
                await limiter.acquire()
                ts = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
                response = await client.query(
                    endpoint,
                    prompt,
                    temperature=0.0,
                    logprobs=False,
                    **strategy_to_query_args(strategy),
                )
            if response.error:
                logger.warning(f"Error for {endpoint}: {prompt!r}: {response.error.message}")
                n_errors += 1
                continue
            tok = extract_first_token(response)
            if tok:
                samples[prompt].append((ts, tok))
            if i < n_per_prompt - 1:
                await asyncio.sleep(cfg.request_delay_seconds)

    await asyncio.gather(*(one(p) for p in prompts))
    return samples, n_errors
```

(While implementing, check `client.query` and `extract_first_token` signatures in `api.py` / `bi/common.py` and align; the test's fake mirrors the real response shape.)

- [ ] **Step 4: Make the test fast** — the fake hits `request_delay_seconds = 5.0`; monkeypatch it in the test:

```python
def test_sample_prompts_collects_n_per_prompt(monkeypatch):
    from trackllm_website.config import config
    monkeypatch.setattr(config.bi.phase_2, "request_delay_seconds", 0.0)
    ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_bi_sampling.py -v`
Expected: PASS in < 5 s

- [ ] **Step 6: Commit**

```bash
git add src/trackllm_website/bi/sampling.py tests/test_bi_sampling.py
git commit -m "add reusable async BI sampler"
```

---

### Task 7: Refactor `phase_1a` to take an explicit endpoint list

**Files:**
- Modify: `src/trackllm_website/bi/phase_1.py`

- [ ] **Step 1: Change the signature**

`async def phase_1a(temperature: float, base_dir: Path | None = None)` becomes:

```python
async def phase_1a(
    endpoints: list[Endpoint], temperature: float, base_dir: Path | None
) -> None:
```

Inside, delete `endpoints = config.endpoints_bi_phase_1`. Update the `__main__` block:

```python
if __name__ == "__main__":
    TEMPERATURE = 0.0
    asyncio.run(phase_1a(config.endpoints_bi_phase_1, TEMPERATURE, None))
```

Search for other callers: `rg -n "phase_1a\(" src tests` and update each the same way.

- [ ] **Step 2: Verify imports still resolve**

Run: `uv run python -c "from trackllm_website.bi.phase_1 import phase_1a" && uv run pytest -q`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add src/trackllm_website/bi/phase_1.py
git commit -m "make phase_1a take an explicit endpoint list"
```

---

### Task 8: `bi/reinit.py` — hybrid re-init / onboarding

**Files:**
- Create: `src/trackllm_website/bi/reinit.py`
- Test: `tests/test_bi_reinit.py`

- [ ] **Step 1: Write the failing tests (mocked sampler/discovery)**

```python
import asyncio
from datetime import datetime, timezone

import pytest

from trackllm_website.bi import reinit as reinit_mod
from trackllm_website.config import Endpoint

ENDPOINT = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))
NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)


def fake_sampler(distributions):
    """distributions: prompt -> list of tokens to cycle through."""
    async def sample(client, endpoint, strategy, prompts, n):
        return (
            {p: [(NOW.isoformat(), distributions[p][i % len(distributions[p])])
                 for i in range(n)] for p in prompts},
            0,
        )
    return sample


def test_reinit_keeps_survivors_and_ranks(monkeypatch):
    # old BI "dead" collapsed to one token; "alive" still has two
    monkeypatch.setattr(
        reinit_mod, "sample_prompts",
        fake_sampler({"alive": ["a", "b"], "dead": ["a"], "new1": ["x", "y"]}),
    )
    async def fake_discover(endpoint, exclude):
        return ["new1"]
    monkeypatch.setattr(reinit_mod, "discover_candidates", fake_discover)
    monkeypatch.setattr(reinit_mod.config.bi.reinit, "top_k_bis", 2)
    monkeypatch.setattr(reinit_mod.config.bi.reinit, "min_bis", 2)

    epoch = asyncio.run(reinit_mod.reinit(None, None, ENDPOINT, ["alive", "dead"], NOW))
    assert epoch is not None
    assert sorted(epoch.border_inputs) == ["alive", "new1"]
    assert set(epoch.reference) == {"alive", "new1"}


def test_reinit_returns_none_below_min_bis(monkeypatch):
    monkeypatch.setattr(reinit_mod, "sample_prompts", fake_sampler({"dead": ["a"]}))
    async def fake_discover(endpoint, exclude):
        return []
    monkeypatch.setattr(reinit_mod, "discover_candidates", fake_discover)
    assert asyncio.run(reinit_mod.reinit(None, None, ENDPOINT, ["dead"], NOW)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bi_reinit.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `bi/reinit.py`**

```python
"""Hybrid re-initialization: re-probe old BIs, top up via phase 1, rank, keep top-k.

Also the onboarding path for new endpoints (old_bis=[]).
"""

from datetime import datetime

from trackllm_website.bi.common import QueryStrategy
from trackllm_website.bi.detection import select_top_bis
from trackllm_website.bi.phase_1 import phase_1a
from trackllm_website.bi.sampling import sample_prompts
from trackllm_website.bi.state import Epoch
from trackllm_website.config import Endpoint, config, logger


async def discover_candidates(endpoint: Endpoint, exclude: list[str]) -> list[str]:
    """Run phase 1a discovery for one endpoint, returning new BI candidates."""
    import tempfile
    from pathlib import Path

    from trackllm_website.bi.phase_1 import Phase1EndpointState  # for reading results

    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp)
        await phase_1a([endpoint], 0.0, base_dir)
        # phase 1a writes per-endpoint results under base_dir; collect border tokens
        from trackllm_website.bi.analyze import load_border_inputs  # noqa: F401
        import orjson

        results_dir = config.bi.get_phase_1_dir(0.0, base_dir)
        candidates: list[str] = []
        for f in results_dir.glob("*.json"):
            data = orjson.loads(f.read_bytes())
            for prompts_dict in data.values():
                for prompt, outputs in prompts_dict.items():
                    if len(set(outputs)) >= 2 and prompt not in exclude:
                        candidates.append(prompt)
        return candidates


async def reinit(
    client,
    strategy: QueryStrategy,
    endpoint: Endpoint,
    old_bis: list[str],
    now: datetime,
) -> Epoch | None:
    """Returns the new epoch, or None if fewer than min_bis BIs were found."""
    r = config.bi.reinit

    survivors: list[str] = []
    if old_bis:
        reprobe, _ = await sample_prompts(client, endpoint, strategy, old_bis, r.reprobe_samples)
        survivors = [p for p, s in reprobe.items() if len({tok for _, tok in s}) > 1]
        logger.info(f"{endpoint}: {len(survivors)}/{len(old_bis)} BIs survived re-probe")

    candidates = survivors
    if len(candidates) < r.top_k_bis:
        candidates = candidates + await discover_candidates(endpoint, exclude=candidates)

    if not candidates:
        return None

    reference, _ = await sample_prompts(
        client, endpoint, strategy, candidates, r.reference_samples
    )
    reference = {p: s for p, s in reference.items() if s}
    keep = select_top_bis(reference, r.top_k_bis)
    if len(keep) < r.min_bis:
        logger.warning(f"{endpoint}: only {len(keep)} BIs after re-init, below min {r.min_bis}")
        return None
    return Epoch(
        start=now,
        border_inputs=keep,
        reference={p: reference[p] for p in keep},
    )
```

(`discover_candidates` parses phase 1a's on-disk format — verify the exact result layout in `bi/common.py::EndpointState` while implementing and adjust the parsing; the structure is `{token_count: {prompt: [outputs]}}` per the cost code in `bi/analyze.py::compute_phase1_cost`. The `import ... # noqa` line is scaffolding and must not survive — remove unused imports before committing.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bi_reinit.py -v`
Expected: 2 PASS

- [ ] **Step 5: Run full suite + prek, commit**

```bash
uv run pytest -q && prek run --all-files
git add src/trackllm_website/bi/reinit.py tests/test_bi_reinit.py
git commit -m "add hybrid re-init / onboarding"
```

---

### Task 9: `bi/monitor.py` — daily monitor

**Files:**
- Create: `src/trackllm_website/bi/monitor.py`
- Test: `tests/test_bi_monitor.py`

Design: pure decision function `decide(state, results, today) -> Decision` separated from I/O, so tests run on fixtures without network.

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime, timezone
from pathlib import Path

import orjson

from trackllm_website.bi.monitor import Decision, decide
from trackllm_website.bi.migrate_state import migrate_endpoint
from trackllm_website.config import Endpoint

FIXTURES = Path("tests/fixtures/phase_2")
ENDPOINT = Endpoint(api="openrouter", model="m/x", provider="p", cost=(1, 1))


def open_state_from_fixture(slug: str):
    results = orjson.loads((FIXTURES / slug / "data.json").read_bytes())
    state = migrate_endpoint(ENDPOINT, results)
    state.status = "monitoring"
    state.retired = None
    epoch = state.epochs[0]
    epoch.end = None
    epoch.end_reason = None
    return state, results


def test_change_detected_closes_epoch():
    state, results = open_state_from_fixture(
        "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8"
    )
    decision = decide(state, results, datetime(2026, 2, 15, tzinfo=timezone.utc))
    assert decision.action == "reinit"
    assert decision.change_date.date().isoformat() == "2026-01-24"


def test_stable_endpoint_no_action():
    state, results = open_state_from_fixture("openai2fgpt-4o-mini23azure")
    decision = decide(state, results, datetime(2026, 2, 15, tzinfo=timezone.utc))
    assert decision.action == "none"
    assert decision.unstable is False


def test_stalled_endpoint_retired():
    # mistral-7b together: all queries error after 2026-02-25
    state, results = open_state_from_fixture(
        "mistralai2fmistral-7b-instruct-v0.323together"
    )
    decision = decide(state, results, datetime(2026, 3, 10, tzinfo=timezone.utc))
    assert decision.action == "retire_stalled"
```

Note: the hyperbolic fixture changes on Jan 24 and the rule fires once; `decide` reports the *latest unhandled* event so the monitor re-inits exactly once.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bi_monitor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `bi/monitor.py`**

```python
"""Daily BI monitor: sample border inputs, detect changes, trigger re-init."""

import asyncio
from datetime import datetime, timezone
from typing import Literal

import fire
from pydantic import BaseModel

from trackllm_website.api import OpenRouterClient
from trackllm_website.bi.analyze import load_phase2_results
from trackllm_website.bi.common import resolve_strategies
from trackllm_website.bi.detection import adaptive_transitions, epoch_tv_series, is_unstable
from trackllm_website.bi.phase_2 import get_output_path, load_existing_results, save_results
from trackllm_website.bi.reinit import reinit
from trackllm_website.bi.sampling import sample_prompts
from trackllm_website.bi.state import EndpointBIState, load_all_states
from trackllm_website.config import config, logger


class Decision(BaseModel):
    action: Literal["none", "reinit", "retire_stalled"]
    change_date: datetime | None = None
    unstable: bool = False


def _day_has_samples(results: dict, day: str) -> bool | None:
    """True/False if the day was queried with/without successes, None if not queried."""
    queried = False
    for batches in results.values():
        for ts, samples in batches.items():
            if ts[:10] == day:
                queried = True
                if samples:
                    return True
    return False if queried else None


def decide(state: EndpointBIState, results: dict, now: datetime) -> Decision:
    epoch = state.current_epoch
    if epoch is None:
        return Decision(action="none")

    epoch_results = {
        p: {ts: s for ts, s in results.get(p, {}).items()
            if ts >= epoch.start.isoformat()}
        for p in epoch.border_inputs
    }

    # Stall: the most recent stall_days queried days all had zero successes
    recent_days = sorted(
        {ts[:10] for b in epoch_results.values() for ts in b}, reverse=True
    )[: config.bi.reinit.stall_days]
    if len(recent_days) >= config.bi.reinit.stall_days and all(
        _day_has_samples(epoch_results, day) is False for day in recent_days
    ):
        return Decision(action="retire_stalled")

    tv = epoch_tv_series(epoch.reference, epoch_results)
    events = adaptive_transitions(tv)
    if events:
        return Decision(
            action="reinit",
            change_date=datetime.fromisoformat(events[-1]),
            unstable=is_unstable(tv),
        )
    return Decision(action="none", unstable=is_unstable(tv))


async def run_endpoint(client, strategy, state: EndpointBIState, now: datetime) -> None:
    epoch = state.current_epoch
    assert epoch is not None

    samples, _ = await sample_prompts(
        client, state.endpoint, strategy, epoch.border_inputs,
        config.bi.phase_2.queries_per_token,
    )
    path = get_output_path(state.endpoint, now.strftime("%Y-%m"))
    existing = load_existing_results(path)
    batch_key = now.replace(microsecond=0).isoformat()
    for prompt, prompt_samples in samples.items():
        existing.setdefault(prompt, {})[batch_key] = prompt_samples
    save_results(path, existing)

    endpoint_dir = config.bi.phase_2_dir / state.slug
    results = load_phase2_results(endpoint_dir)
    decision = decide(state, results, now)

    if decision.action == "retire_stalled":
        from trackllm_website.bi.state import RetiredInfo

        epoch.end = now
        epoch.end_reason = "stalled"
        state.status = "retired"
        state.retired = RetiredInfo(reason="stalled", since=now, last_recheck=now)
        logger.warning(f"{state.endpoint}: retired (stalled)")
    elif decision.action == "reinit":
        epoch.end = now
        epoch.end_reason = "change_detected"
        epoch.change_date = decision.change_date
        epoch.params = config.bi.detection.model_dump()
        logger.warning(f"{state.endpoint}: change detected (onset {decision.change_date})")
        new_epoch = await reinit(client, strategy, state.endpoint, epoch.border_inputs, now)
        if new_epoch is None:
            from trackllm_website.bi.state import RetiredInfo

            state.status = "retired"
            state.retired = RetiredInfo(reason="no_bis", since=now, last_recheck=now)
        else:
            state.epochs.append(new_epoch)
    state.save(config.bi.state_dir)


async def monitor() -> None:
    states = load_all_states(config.bi.state_dir)
    monitoring = [s for s in states.values() if s.status == "monitoring"]
    logger.info(f"Monitoring {len(monitoring)} endpoints")
    now = datetime.now(tz=timezone.utc)

    async with OpenRouterClient(timeout=60.0) as probe_client:
        strategies, failed = await resolve_strategies(
            probe_client, [s.endpoint for s in monitoring]
        )
    client = OpenRouterClient()
    try:
        await asyncio.gather(*(
            run_endpoint(client, strategies[str(s.endpoint)], s, now)
            for s in monitoring if str(s.endpoint) in strategies
        ))
    finally:
        await client.close()


if __name__ == "__main__":
    fire.Fire(monitor)
```

Important detail: `decide` must not re-fire on an event already handled — after a re-init the new epoch only sees data from its own start, so this is structural; no event bookkeeping needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bi_monitor.py -v`
Expected: 3 PASS

- [ ] **Step 5: Full suite + prek, commit**

```bash
uv run pytest -q && prek run --all-files
git add src/trackllm_website/bi/monitor.py tests/test_bi_monitor.py
git commit -m "add daily BI monitor"
```

---

### Task 10: Lifecycle — onboarding, delisting, resurrection

**Files:**
- Modify: `src/trackllm_website/update_endpoints.py`
- Test: `tests/test_bi_lifecycle.py`

- [ ] **Step 1: Write the failing tests (pure selection logic)**

```python
from datetime import datetime, timedelta, timezone

from trackllm_website.bi.state import EndpointBIState, Epoch, RetiredInfo
from trackllm_website.config import Endpoint
from trackllm_website.update_endpoints import select_lifecycle_actions

NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)


def ep(model):
    return Endpoint(api="openrouter", model=model, provider="p", cost=(1, 1))


def retired_state(model, reason, last_recheck):
    return EndpointBIState(
        endpoint=ep(model), status="retired",
        retired=RetiredInfo(reason=reason, since=NOW - timedelta(days=60),
                            last_recheck=last_recheck),
        epochs=[],
    )


def test_new_candidates_onboarded_up_to_cap():
    candidates = [ep(f"m/{i}") for i in range(20)]
    actions = select_lifecycle_actions(candidates, {}, NOW)
    assert len(actions.onboard) == 10  # max_onboard_per_run


def test_recheck_due_only_after_interval():
    states = {
        "due": retired_state("m/due", "stalled", NOW - timedelta(days=20)),
        "recent": retired_state("m/recent", "stalled", NOW - timedelta(days=2)),
    }
    candidates = [ep("m/due"), ep("m/recent")]
    actions = select_lifecycle_actions(candidates, states, NOW)
    assert [s.endpoint.model for s in actions.recheck] == ["m/due"]


def test_delisted_when_absent_from_catalog():
    states = {"gone": EndpointBIState(endpoint=ep("m/gone"), status="monitoring", epochs=[])}
    actions = select_lifecycle_actions([], states, NOW)
    assert [s.endpoint.model for s in actions.delist] == ["m/gone"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bi_lifecycle.py -v`
Expected: FAIL with `ImportError: cannot import name 'select_lifecycle_actions'`

- [ ] **Step 3: Implement in `update_endpoints.py`**

```python
class LifecycleActions(BaseModel):
    onboard: list[Endpoint]
    recheck: list[EndpointBIState]
    delist: list[EndpointBIState]


def select_lifecycle_actions(
    candidates: list[Endpoint],
    states: dict[str, EndpointBIState],
    now: datetime,
) -> LifecycleActions:
    """Pure selection of lifecycle actions; the caller performs them."""
    r = config.bi.reinit
    known = {s.slug for s in states.values()}
    candidate_set = set(candidates)

    onboard = [
        e for e in candidates
        if slugify(f"{e.model}#{e.provider}") not in known
    ][: r.max_onboard_per_run]

    recheck = [
        s for s in states.values()
        if s.status == "retired"
        and s.endpoint in candidate_set
        and now - s.retired.last_recheck >= timedelta(days=r.recheck_days)
    ]

    delist = [
        s for s in states.values()
        if s.status == "monitoring" and s.endpoint not in candidate_set
    ]
    return LifecycleActions(onboard=onboard, recheck=recheck, delist=delist)


async def update_endpoints_bi_lifecycle():
    """Onboard new candidates, delist vanished endpoints, re-check retired ones."""
    states = load_all_states(config.bi.state_dir)
    candidates = config.endpoints_bi
    now = datetime.now(tz=timezone.utc)
    actions = select_lifecycle_actions(candidates, states, now)
    logger.info(
        f"BI lifecycle: {len(actions.onboard)} to onboard, "
        f"{len(actions.recheck)} to re-check, {len(actions.delist)} to delist"
    )

    for state in actions.delist:
        epoch = state.current_epoch
        if epoch is not None:
            epoch.end = now
            epoch.end_reason = "gap"
        state.status = "retired"
        state.retired = RetiredInfo(reason="delisted", since=now, last_recheck=now)
        state.save(config.bi.state_dir)

    to_init = [(e, []) for e in actions.onboard] + [
        (s.endpoint, []) for s in actions.recheck
    ]
    if not to_init:
        return
    async with OpenRouterClient(timeout=60.0) as probe_client:
        strategies, _ = await resolve_strategies(probe_client, [e for e, _ in to_init])
    client = OpenRouterClient()
    try:
        for endpoint, old_bis in to_init:
            if str(endpoint) not in strategies:
                continue
            epoch = await reinit(client, strategies[str(endpoint)], endpoint, old_bis, now)
            slug = slugify(f"{endpoint.model}#{endpoint.provider}")
            state = states.get(slug) or EndpointBIState(
                endpoint=endpoint, status="monitoring", epochs=[]
            )
            if epoch is None:
                state.status = "retired"
                state.retired = RetiredInfo(reason="no_bis", since=now, last_recheck=now)
            else:
                state.status = "monitoring"
                state.retired = None
                state.epochs.append(epoch)
            state.save(config.bi.state_dir)
    finally:
        await client.close()
```

Add the needed imports (`datetime`, `timedelta`, `timezone`, `EndpointBIState`, `RetiredInfo`, `load_all_states`, `reinit`, `resolve_strategies`, `slugify`) and call `update_endpoints_bi_lifecycle()` from `main()` after `update_endpoints_bi()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bi_lifecycle.py -v`
Expected: 3 PASS

- [ ] **Step 5: Full suite + prek, commit**

```bash
uv run pytest -q && prek run --all-files
git add src/trackllm_website/update_endpoints.py tests/test_bi_lifecycle.py
git commit -m "add BI endpoint lifecycle (onboard, delist, resurrect)"
```

---

### Task 11: Remove `endpoints_bi_phase_1.yaml`

**Files:**
- Delete: `endpoints_bi_phase_1.yaml`
- Modify: `src/trackllm_website/config.py`, `src/trackllm_website/bi/phase_1.py`, `src/trackllm_website/bi/phase_2.py`, `src/trackllm_website/util.py`, `src/trackllm_website/bi/analyze.py`

- [ ] **Step 1: Find all references**

Run: `rg -n "endpoints_bi_phase_1|bi_phase_1" src tests`

- [ ] **Step 2: Replace each use**

- `config.py`: remove `endpoints_yaml_path_bi_phase_1`, its entry in `yaml_file=[...]`, and the `endpoints_bi_phase_1` field.
- `phase_1.py` `__main__`: use `config.endpoints_bi` (discovery candidates come from the vetted list now).
- `phase_2.py`: `endpoints = config.endpoints_bi_phase_1` → derive from state files: `[s.endpoint for s in load_all_states(config.bi.state_dir).values() if s.status == "monitoring"]` (phase_2.py stays usable standalone; the monitor is the scheduled path).
- `util.py::endpoint_from_slug`: drop `config.endpoints_bi_phase_1` from the candidate list.
- `analyze.py::compute_phase1_cost`: use `config.endpoints_bi`.
- Delete the file: `git rm endpoints_bi_phase_1.yaml`

- [ ] **Step 3: Verify**

Run: `uv run pytest -q && uv run python -c "from trackllm_website.config import config; print(len(config.endpoints_bi))" && prek run --all-files`
Expected: tests pass, endpoint count prints

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "retire endpoints_bi_phase_1.yaml; monitored set lives in state files"
```

---

### Task 12: LT change-event log

**Files:**
- Create: `src/trackllm_website/lt_events.py`
- Modify: `src/trackllm_website/lt_scores.py` (call event update from compute_all/compute_latest)
- Test: `tests/test_lt_events.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime, timezone

from trackllm_website.lt_events import LTChangeEvent, merge_events
from trackllm_website.lt_scores import ChangePoint

NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)
DATES = [datetime(2026, 1, 1, tzinfo=timezone.utc)] * 300  # placeholder index→date map


def test_new_change_appended():
    events = merge_events("slug", [], [ChangePoint(index=150, sigma=14.0)], DATES, NOW)
    assert len(events) == 1
    assert events[0].first_detected == NOW


def test_recomputed_change_near_existing_is_same_event():
    existing = [LTChangeEvent(endpoint="slug", index=150, date=DATES[150],
                              sigma=14.0, first_detected=NOW)]
    events = merge_events(
        "slug", existing, [ChangePoint(index=160, sigma=15.0)], DATES, NOW
    )
    assert len(events) == 1
    assert events[0].first_detected == NOW  # original detection date kept
    assert events[0].sigma == 15.0  # magnitude refreshed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lt_events.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `lt_events.py`**

```python
"""Append-only LT change-event log with stable first-detected dates."""

from datetime import datetime
from pathlib import Path

import orjson
from pydantic import BaseModel

from trackllm_website.lt_scores import PEAK_DISTANCE, ChangePoint

EVENTS_FILENAME = "lt_changes.json"


class LTChangeEvent(BaseModel):
    endpoint: str
    index: int
    date: datetime
    sigma: float
    first_detected: datetime


def merge_events(
    slug: str,
    existing: list[LTChangeEvent],
    changes: list[ChangePoint],
    dates: list[datetime],
    now: datetime,
) -> list[LTChangeEvent]:
    """A recomputed change within PEAK_DISTANCE indices of an existing event is
    the same event: keep first_detected, refresh index/date/sigma."""
    merged = list(existing)
    for cp in changes:
        match = next(
            (e for e in merged if abs(e.index - cp.index) <= PEAK_DISTANCE), None
        )
        if match is not None:
            match.index = cp.index
            match.date = dates[cp.index]
            match.sigma = cp.sigma
        else:
            merged.append(
                LTChangeEvent(
                    endpoint=slug, index=cp.index, date=dates[cp.index],
                    sigma=cp.sigma, first_detected=now,
                )
            )
    return merged


def load_events(path: Path) -> dict[str, list[LTChangeEvent]]:
    if not path.exists():
        return {}
    raw = orjson.loads(path.read_bytes())
    return {
        slug: [LTChangeEvent.model_validate(e) for e in events]
        for slug, events in raw.items()
    }


def save_events(path: Path, events: dict[str, list[LTChangeEvent]]) -> None:
    path.write_bytes(
        orjson.dumps(
            {s: [e.model_dump(mode="json") for e in evts] for s, evts in events.items()},
            option=orjson.OPT_INDENT_2,
        )
    )
```

In `lt_scores.py`, after `_save_scores(...)` in both `compute_all` and `compute_latest`, update the shared events file:

```python
from datetime import timezone

from trackllm_website.lt_events import EVENTS_FILENAME, load_events, merge_events, save_events

# inside the per-endpoint loop, after _save_scores:
        events_path = data_dir / EVENTS_FILENAME
        all_events = load_events(events_path)
        all_events[endpoint_dir.name] = merge_events(
            endpoint_dir.name,
            all_events.get(endpoint_dir.name, []),
            scores.changes,
            scores.dates,
            datetime.now(tz=timezone.utc),
        )
        save_events(events_path, all_events)
```

(Hoist the load/save outside the loop in each function so the file is read once and written once per run.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lt_events.py -v`
Expected: 2 PASS

- [ ] **Step 5: Run scoring end-to-end on real data**

Run: `uv run python src/trackllm_website/lt_scores.py all && uv run python -c "
from pathlib import Path; import orjson
d = orjson.loads(Path('website/data/lt_changes.json').read_bytes())
print(len(d), 'endpoints with events:', sum(len(v) for v in d.values()), 'events')"`
Expected: counts print without error

- [ ] **Step 6: Commit**

```bash
git add src/trackllm_website/lt_events.py src/trackllm_website/lt_scores.py tests/test_lt_events.py
git commit -m "add stable LT change-event log"
```

---

### Task 13: Workflows

**Files:**
- Create: `.github/workflows/bi-monitor.yml`
- Delete: `.github/workflows/bi-phase-2.yml`
- Modify: `.github/workflows/run-main.yml`, `.github/workflows/update-endpoints.yml`

- [ ] **Step 1: Create `bi-monitor.yml`** (modeled on bi-phase-2.yml, running from main)

```yaml
name: BI daily monitor

on:
  schedule:
    - cron: '01 21 * * *'
  workflow_dispatch:

permissions:
  contents: write

jobs:
  bi-monitor:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.13'
    - uses: astral-sh/setup-uv@v4
    - run: uv sync
    - name: Run BI monitor
      env:
        OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
      run: uv run python -m trackllm_website.bi.monitor
    - name: Commit results
      run: |
        git config user.name "github-actions[bot]"
        git config user.email "github-actions[bot]@users.noreply.github.com"
        git add data_bi
        git diff --staged --quiet || git commit -m "[bot] daily BI monitor run"
        git pull --rebase && git push
```

(Mirror the exact commit/push idiom used at the end of the existing `run-main.yml` — read it first and copy its conflict-handling.)

- [ ] **Step 2: Delete the retired workflow**

```bash
git rm .github/workflows/bi-phase-2.yml
```

- [ ] **Step 3: Add LT scoring to `run-main.yml`**

After the step that runs `main.py` and before the commit step, insert:

```yaml
    - name: Compute LT scores
      run: uv run python src/trackllm_website/lt_scores.py latest
```

Confirm the commit step `git add`s `website/data` (it already commits collected data there; lt_scores.json and lt_changes.json land in the same tree).

- [ ] **Step 4: Confirm `update-endpoints.yml` picks up the lifecycle**

It runs `update_endpoints.py` `main()`, which now includes `update_endpoints_bi_lifecycle()`. Verify the workflow's commit step adds `data_bi` (state files change on onboarding); if it only adds the yaml files, extend it with `git add data_bi`.

- [ ] **Step 5: Validate workflow syntax**

Run: `uv run python -c "import yaml; [yaml.safe_load(open(f)) for f in ['.github/workflows/bi-monitor.yml', '.github/workflows/run-main.yml', '.github/workflows/update-endpoints.yml']]; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add .github/workflows
git commit -m "schedule daily BI monitor from main; compute LT scores hourly"
```

---

### Task 14: Resumption — re-onboard live endpoints

This runs once, after everything above is merged and the daily workflows are green.

- [ ] **Step 1: Trigger the lifecycle manually** (all 53 historical endpoints are `retired` after migration; live candidates get re-onboarded through the capped daily lifecycle, ~10/day, or run it repeatedly):

Run: `uv run python -c "
import asyncio
from trackllm_website.update_endpoints import update_endpoints_bi_lifecycle
asyncio.run(update_endpoints_bi_lifecycle())"`

Repeat until `0 to onboard, 0 to re-check`. Expected over a few runs: ~23 recoverable endpoints back to `monitoring` with fresh epoch 1; truly delisted ones become `retired(no_bis)` or stay retired.

- [ ] **Step 2: Verify the daily monitor runs clean**

Run: `gh workflow run bi-monitor.yml && sleep 600 && gh run list --workflow bi-monitor.yml --limit 1`
Expected: conclusion `success`, a `[bot] daily BI monitor run` commit touching `data_bi/phase_2` and `data_bi/state`.

- [ ] **Step 3: Commit any state produced locally**

```bash
git add data_bi/state endpoints_bi.yaml bad_endpoints_bi.yaml
git commit -m "re-onboard live BI endpoints after outage"
git push
```

---

## Self-review notes

- **Spec coverage**: state model (T3), facts-vs-derived (T3/T4/T9 — no detector state persisted), adaptive rule + instability (T4), phase 1b top-k (T4 `select_top_bis` + T8), hybrid re-init (T8), daily monitor + stall (T9), lifecycle incl. resurrection + delist + cap (T10), endpoints_bi_phase_1 retirement (T11), LT scoring + stable events (T12–13), workflows from main (T13), migration + resumption (T5, T14). **Not covered here by design**: website display (BI plots, changes feed, instability badge) — separate follow-up plan.
- **Known judgment calls for the implementer**: exact `extract_first_token` response shape (T6), phase 1a on-disk format parsing (T8), commit/push idiom in workflows (T13). Each is flagged inline with where to look.
- Fixture-based date assertions may shift ±1 day from subsampling; tests pin the week, implementer adjusts the exact day after first run (T4 step 4 note).
