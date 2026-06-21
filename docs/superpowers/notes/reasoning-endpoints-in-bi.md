# Note: how BI handles reasoning endpoints

Correcting a misconception: **reasoning endpoints ARE supported by BI.** They are
not "unusable." They only cost more, because their chain-of-thought tokens are
billed as output. The genuinely-unusable case is unrelated: endpoints that ignore
temperature (`temp:NO`) cannot produce the T=0 behavior BI relies on, so those are
excluded.

| Endpoint kind | BI status | Why |
| --- | --- | --- |
| Reasoning (emits CoT before the answer) | **Kept** | A working query strategy is discovered per endpoint; cost is higher because reasoning tokens are billed as output |
| `temp:NO` (ignores temperature) | **Excluded** | T=0 doesn't pin the output, so "border inputs" would be fake with no detection power |

## Why reasoning endpoints work

BI's signal is the first answer-token distribution at T=0: sample a prompt several
times, and if two distinct top tokens both appear, the prompt sits on a border.
Reasoning models emit a CoT trace before the visible answer, so a naive "request 1
output token" probe returns empty content (this is the exact failure
`bi/test_reasoning.py` was written to characterize — see its module docstring,
`test_reasoning.py:1-5`). The codebase solves this with a per-endpoint **query
strategy** plus a first-token extractor that spans the reasoning trace.

### 1. Per-endpoint strategy resolution

Three strategies are modeled (`bi/common.py:30-45`):

- `PlainStrategy` — non-reasoning models; no special args.
- `ReasoningDisabledStrategy` — sends `reasoning={"effort": "none"}`
  (`bi/common.py:52-53`).
- `ReasoningBudgetStrategy(budget)` — lets the model reason to completion within a
  token budget, requesting `output_tokens = budget + 1` and
  `reasoning={"max_tokens": budget}` (`bi/common.py:54-58`).

`discover_strategy` (`bi/common.py:110-171`) probes each endpoint in order: plain,
then `effort=none`, then escalating reasoning budgets `1, 2, 4, …` up to
`config.bi.probe.max_budget` (2048 in `config.toml`). The first strategy that
returns a usable first token wins. On success at a budget, it stores
`ReasoningBudgetStrategy(budget=budget * 2)` — double the discovered budget for
headroom (`bi/common.py:167`, and the doubling round-trips through
`_raw_to_strategy` / `_strategy_to_raw`, `bi/common.py:90-107`).

`resolve_strategies` (`bi/common.py:174-236`) runs this across all endpoints
concurrently and caches results to `config.bi.probe.strategies_path`, so probing is
a one-time cost per endpoint.

### 2. Getting a comparable answer-token distribution after reasoning

The key is `extract_first_token` (`bi/common.py:61-70`): it concatenates
`reasoning_content` and `content`, splits on whitespace, and returns the first
token. So for a reasoning model the "first token" is the first token of the CoT
trace, and for a plain model it's the first token of the answer — both reduced to
the same comparable signal. The API layer parses `reasoning` and `reasoning_tokens`
out of the OpenRouter response (`api.py:125-137`, `api.py:168-181`), and the sampler
calls `strategy_to_query_args(strategy)` on every query so the right reasoning args
are always attached (`bi/sampling.py:41-47`; T=0 detection sampling in
`bi/common.py:780-825`).

### 3. T=0 and top-2 comparison still hold

Temperature is passed through unchanged on every query (`bi/sampling.py:41-47`,
`bi/common.py:786-792`); the strategy only governs the reasoning args, never the
temperature. The border test is still "does T=0 sampling yield >=2 distinct first
tokens" — applied identically to the reasoning-trace-aware first token. So the top-2
tie detection works after a reasoning trace.

## The real exclusion: `temp:NO`, not reasoning

Two distinct mechanisms keep temperature-ignoring endpoints out:

1. **Catalog pre-filter** — models whose OpenRouter `supported_parameters` omits
   `temperature` are routed straight to the `bad_temperature` bucket without even
   probing (`update_endpoints.py:301-348`, logged as "temp:NO"). The
   `supports_temperature` flag is stamped during catalog refresh
   (`update_endpoints.py:108-188`).
2. **Onboarding temperature gate** — for endpoints that pass the pre-filter, if
   phase-1 discovery shows a suspiciously high border-input prevalence (above
   `config.bi.temperature_gate.prevalence_trigger`, 0.30), `reinit` runs
   `check_temperature` (`bi/reinit.py:111-122`, `bi/phase_1.py:180-212`). It
   re-samples border prompts at T=0 and T=1; if raising temperature does **not**
   broaden the output distribution (`temperature_is_ignored`,
   `bi/phase_1.py:169-177`), the endpoint is reported `bad_temperature` and cached
   out — not retired as a normal `no_bis`.

Note this gate fires precisely on *reasoning models that also ignore temperature*
(`bi/phase_1.py:188-195`): the reasoning trace is fine, but a T=0-ignoring model
produces fake borders. It is the temperature behavior, not the reasoning, that
disqualifies them. Both `too_expensive` and `bad_temperature` rejects are
periodically re-vetted (`vetting.py:147-156`, `update_endpoints.py:327-335`), since
providers fix temperature over time.

## Cost implications

Reasoning tokens are billed as output. `vet_endpoint` measures the *actual* billed
cost via `get_generation_cost` rather than trusting token math, and the expected
cost (`response.cost`) already includes reasoning tokens (`vetting.py:29-63`,
`api.py:128`). The status log surfaces `Avg Tok (in/out/reas)` per endpoint so the
reasoning overhead is visible (`bi/common.py:580`, `bi/common.py:616-622`), and
totals track `total_reasoning_tokens` (`bi/common.py:808`). The BI-quality summary
even reports yields separately for plain vs `+reasoning` budgets
(`bi/common.py:676-697`).

The most expensive single path is the onboarding temperature gate on a reasoning
endpoint: up to `check_prompts * check_samples * 2` queries, each billing the full
reasoning budget as output — by design, since that's exactly the kind of endpoint
(reasoning model that might ignore temperature) it needs to rule out
(`bi/phase_1.py:192-195`).

## Edge cases the code actually reveals

- **Hidden reasoning.** If a budget strategy yields a first token but the endpoint
  returns no `reasoning_content` (reasoning is billed but never exposed),
  `discover_strategy` rejects it with a `"hidden reasoning"` error and
  `resolve_strategies` caches it as `{"skip": "hidden reasoning"}`
  (`bi/common.py:163-167`, `bi/common.py:223-224`). Such endpoints are dropped — not
  because reasoning is unsupported, but because the trace can't be observed.
- **Transient errors aren't cached.** Probe failures with HTTP 429/0 short-circuit
  without poisoning the cache (`bi/common.py:119`, `bi/common.py:137-138`); vetting
  returns `transient` for un-priceable responses so they're retried later
  (`vetting.py:46-57`).
- **Cache compatibility.** The persisted budget is the discovered (un-doubled)
  value, kept compatible with `test_reasoning.py`'s strategy file
  (`bi/common.py:99-107`).

Ambiguity worth flagging: BI treats the first whitespace-delimited token of the
concatenated CoT+answer as its signal. For reasoning models this is effectively the
first CoT token, which is a reasonable and consistent proxy but is not literally the
"first post-reasoning answer token." The code does not isolate the answer's first
token separately; it relies on the concatenation in `extract_first_token`. That is
the actual mechanism, stated here without embellishment.
