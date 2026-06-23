"""Test whether border inputs generalize across models.

For each pair of endpoints that share tested tokens, builds a 2x2 contingency
table (BI for A × BI for B) and runs a Cochran-Mantel-Haenszel test of
independence, stratified by endpoint pair category:

- Same model, different provider
- Same tokenizer, different model
- Both unknown tokenizer, different model (uses fallback token pool)
- Different tokenizer
"""

import os
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import numpy as np
import orjson
from scipy.stats import chi2
from statsmodels.stats.contingency_tables import StratifiedTable

from trackllm_website.config import config


class PairCategory(Enum):
    SAME_MODEL = auto()
    DIFFERENT_MODEL = auto()


CATEGORY_LABELS = {
    PairCategory.SAME_MODEL: "Same model, diff provider",
    PairCategory.DIFFERENT_MODEL: "Different model",
}

MIN_SHARED_TOKENS = 100

# Models that should be treated as the same for independence analysis
MODEL_EQUIVALENCES: list[set[str]] = [
    {"mistralai/ministral-3b", "mistralai/ministral-3b-2512"},
    {"mistralai/ministral-8b", "mistralai/ministral-8b-2512"},
    {
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-v3.2-speciale",
        "deepseek/deepseek-v3.2-exp",
    },
    {"google/gemini-2.5-flash-lite-preview-09-2025", "google/gemini-2.5-flash-lite"},
    {"openai/gpt-4o-mini", "openai/gpt-4o-mini-2024-07-18"},
    {"qwen/qwen3-30b-a3b", "qwen/qwen3-30b-a3b-instruct-2507"},
]

_MODEL_CANONICAL: dict[str, str] = {}
for _group in MODEL_EQUIVALENCES:
    _canonical = min(_group)
    for _model in _group:
        _MODEL_CANONICAL[_model] = _canonical


@dataclass
class CategoryResult:
    tables: list[np.ndarray] = field(default_factory=list)
    endpoints: set[str] = field(default_factory=set)

    @property
    def n_models(self) -> int:
        return len({_get_model(e) for e in self.endpoints})


def _get_model(fname: str) -> str:
    name = fname.removesuffix(".json")
    model = name.split("23")[0].replace("2f", "/")
    return _MODEL_CANONICAL.get(model, model)


def load_endpoint_bis(data_dir: Path) -> dict[str, tuple[set[str], set[str]]]:
    """Returns {filename: (tested_prompts, border_inputs)}."""
    result = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(data_dir / fname, "rb") as f:
            d = orjson.loads(f.read())
        tested = set()
        bis = set()
        for prompt_dict in d.values():
            for prompt, outputs in prompt_dict.items():
                tested.add(prompt)
                if len(set(outputs)) >= 2:
                    bis.add(prompt)
        result[fname] = (tested, bis)
    return result


def classify_pair(model_i: str, model_j: str) -> PairCategory:
    if model_i == model_j:
        return PairCategory.SAME_MODEL
    return PairCategory.DIFFERENT_MODEL


def _longest_common_substring(a: str, b: str) -> int:
    best = 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best


def build_contingency_tables(
    endpoint_bis: dict[str, tuple[set[str], set[str]]],
    show_lcs: bool,
) -> dict[PairCategory, CategoryResult]:
    results: dict[PairCategory, CategoryResult] = {
        c: CategoryResult() for c in PairCategory
    }
    fnames = sorted(endpoint_bis)
    cross_model_lcs: list[tuple[int, str, str]] = []

    for i in range(len(fnames)):
        tested_i, bis_i = endpoint_bis[fnames[i]]
        model_i = _get_model(fnames[i])

        for j in range(i + 1, len(fnames)):
            tested_j, bis_j = endpoint_bis[fnames[j]]
            shared = tested_i & tested_j
            if len(shared) < MIN_SHARED_TOKENS:
                continue

            bi_i = bis_i & shared
            bi_j = bis_j & shared
            a = len(bi_i & bi_j)
            b = len(bi_i - bi_j)
            c = len(bi_j - bi_i)
            d = len(shared) - a - b - c

            model_j = _get_model(fnames[j])
            cat = classify_pair(model_i, model_j)
            results[cat].tables.append(np.array([[a, b], [c, d]]))
            results[cat].endpoints.update([fnames[i], fnames[j]])

            if show_lcs and cat == PairCategory.DIFFERENT_MODEL:
                lcs = _longest_common_substring(model_i, model_j)
                cross_model_lcs.append((lcs, model_i, model_j))

    if show_lcs:
        # Check if models in "different model" may actually be the same;
        # if so, update MODEL_EQUIVALENCES and re-run.
        cross_model_lcs.sort(key=lambda x: -x[0])
        seen = set()
        print("\nTop cross-model pairs by longest common substring:")
        for lcs, mi, mj in cross_model_lcs:
            pair = (min(mi, mj), max(mi, mj))
            if pair in seen:
                continue
            seen.add(pair)
            if len(seen) > 50:
                break
            print(f"  LCS={lcs:3d}  {mi}  vs  {mj}")

    return results


@dataclass
class CMHResult:
    odds_ratio: float
    ci_low: float
    ci_high: float
    p_value: float


def compute_cmh(tables: list[np.ndarray]) -> CMHResult | None:
    valid = [t for t in tables if all(t.sum(axis=1) > 0) and all(t.sum(axis=0) > 0)]
    if len(valid) < 2:
        return None
    st = StratifiedTable(valid)
    lor_ci = st.logodds_pooled_confint()
    # Recompute p-value from the statistic; statsmodels underflows to 0.0
    cmh_stat = st.test_null_odds().statistic
    return CMHResult(
        odds_ratio=st.oddsratio_pooled,
        ci_low=np.exp(lor_ci[0]),
        ci_high=np.exp(lor_ci[1]),
        p_value=chi2.sf(cmh_stat, 1),
    )


def _format_p(p: float) -> str:
    return f"{p:.2f}" if p >= 0.01 else f"{p:.1e}"


def analyze_category(label: str, result: CategoryResult) -> None:
    tables = result.tables
    print(f"\n{'=' * 60}")
    print(
        f"{label}: {result.n_models} models, "
        f"{len(result.endpoints)} endpoints, {len(tables)} pairs"
    )
    print(f"{'=' * 60}")
    if not tables:
        return

    pooled = sum(tables)
    a, b, c, d = pooled[0, 0], pooled[0, 1], pooled[1, 0], pooled[1, 1]
    print(f"  Pooled counts:  both_BI={a}  only_A={b}  only_B={c}  neither={d}")

    valid = [t for t in tables if all(t.sum(axis=1) > 0) and all(t.sum(axis=0) > 0)]
    print(f"  Valid strata for CMH: {len(valid)}/{len(tables)}")

    cmh = compute_cmh(tables)
    if cmh:
        print(
            f"  CMH Odds Ratio: {cmh.odds_ratio:.2f} "
            f"(95% CI: [{cmh.ci_low:.2f}, {cmh.ci_high:.2f}]), "
            f"p = {_format_p(cmh.p_value)}"
        )


def print_markdown_table(
    all_results: dict[PairCategory, CategoryResult],
) -> None:
    print()
    print("| Pair type | Models | Endpoints | CMH Odds Ratio (95% CI) | *p* |")
    print("|---|---|---|---|---|")
    for cat in PairCategory:
        result = all_results[cat]
        cmh = compute_cmh(result.tables)
        if cmh:
            or_str = f"{cmh.odds_ratio:.2f} ({cmh.ci_low:.2f}, {cmh.ci_high:.2f})"
            p_str = _format_p(cmh.p_value)
        else:
            or_str = "—"
            p_str = "—"
        print(
            f"| {CATEGORY_LABELS[cat]} | {result.n_models} | "
            f"{len(result.endpoints)} | {or_str} | {p_str} |"
        )


def main(
    data_dir: str | None = None,
    show_lcs: bool = False,
    markdown_table: bool = True,
) -> None:
    if data_dir is None:
        data_dir = str(config.bi.data_dir / "bi_prevalence" / "T=0")
    endpoint_bis = load_endpoint_bis(Path(data_dir))
    print(f"Loaded {len(endpoint_bis)} endpoints from {data_dir}")

    results = build_contingency_tables(endpoint_bis, show_lcs=show_lcs)

    for cat in PairCategory:
        analyze_category(CATEGORY_LABELS[cat], results[cat])

    if markdown_table:
        print_markdown_table(results)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
