"""Load phase 2 results and compare the token distributions they hold.

The production core the detectors, the monitor and the site build share; the
analysis and plotting scripts in `bi/analysis/` build on top of it.
"""

import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from trackllm_website.bi.phase_2 import (
    ResponseToken,
    Results,
    load_existing_results,
)


def load_phase2_results(
    endpoint_dir: Path,
    max_prompts: int | None = None,
    max_samples_per_timestamp: int | None = None,
    seed: int | None = None,
) -> Results:
    """Load all phase 2 results for an endpoint across all months."""
    combined: Results = {}
    # sorted: glob order is filesystem order, and it sets the prompt/batch dict
    # order every downstream mean is summed in
    for json_file in sorted(endpoint_dir.glob("*.json")):
        data = load_existing_results(json_file)
        for prompt, batches in data.items():
            if prompt not in combined:
                combined[prompt] = {}
            combined[prompt].update(batches)

    if not combined:
        return combined

    no_filtering = max_prompts is None and max_samples_per_timestamp is None
    if no_filtering:
        return combined

    rng = random.Random(seed)

    if max_prompts is not None and len(combined) > max_prompts:
        selected_prompts = rng.sample(list(combined.keys()), max_prompts)
        combined = {p: combined[p] for p in selected_prompts}

    if max_samples_per_timestamp is not None:
        for batches in combined.values():
            for timestamp, samples in batches.items():
                if len(samples) > max_samples_per_timestamp:
                    batches[timestamp] = rng.sample(samples, max_samples_per_timestamp)

    return combined


def compute_tv_distance(
    dist_p: Counter[ResponseToken], dist_q: Counter[ResponseToken]
) -> float | None:
    """Compute total variation distance between two distributions."""
    all_tokens = set(dist_p.keys()) | set(dist_q.keys())
    total_p = sum(dist_p.values())
    total_q = sum(dist_q.values())
    if total_p == 0 or total_q == 0:
        return None
    tv = 0.0
    # sorted: set order follows PYTHONHASHSEED, and the float summation order
    # would leak into the last digits, making site builds non-reproducible
    for token in sorted(all_tokens):
        p_prob = dist_p[token] / total_p
        q_prob = dist_q[token] / total_q
        tv += abs(p_prob - q_prob)
    return tv / 2


def get_distribution(
    responses: Sequence[Sequence[str]],
) -> Counter[str]:
    """Get token distribution from responses (each item is (timestamp, token))."""
    return Counter(token for _, token in responses)
