"""Hybrid re-initialization: re-probe old BIs, top up via phase 1, rank, keep top-k.

Also the onboarding path for new endpoints (old_bis=[]).
"""

import tempfile
from datetime import datetime
from pathlib import Path

import orjson

from trackllm_website.bi.common import META_KEY, QueryStrategy
from trackllm_website.bi.detection import select_top_bis
from trackllm_website.bi.phase_1 import phase_1a
from trackllm_website.bi.phase_2 import (
    get_output_path,
    load_existing_results,
    save_results,
)
from trackllm_website.bi.sampling import sample_prompts
from trackllm_website.bi.state import Epoch
from trackllm_website.config import Endpoint, config, logger


def parse_phase_1_candidates(results_dir: Path, exclude: list[str]) -> list[str]:
    """Parse phase-1a output files into BI candidates.

    Phase 1a writes one file per endpoint with structure
    {token_count: {prompt: [outputs]}}, plus a top-level META_KEY entry when
    meta tracking is on; a prompt is a border input when it has at least two
    distinct non-empty outputs. The border_inputs.json file (phase 1b) and the
    META_KEY top-level key are skipped.
    """
    excluded = set(exclude)
    candidates: list[str] = []
    for f in results_dir.glob("*.json"):
        if f.name == "border_inputs.json":
            continue
        data = orjson.loads(f.read_bytes())
        for token_count, prompts_dict in data.items():
            if token_count == META_KEY:
                continue
            for prompt, outputs in prompts_dict.items():
                if prompt in excluded:
                    continue
                if len({o for o in outputs if o}) >= 2:
                    candidates.append(prompt)
    return candidates


async def discover_candidates(endpoint: Endpoint, exclude: list[str]) -> list[str]:
    """Run phase 1a discovery for one endpoint, returning new BI candidates."""
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp)
        await phase_1a([endpoint], 0.0, base_dir)
        results_dir = config.bi.get_phase_1_dir(0.0, base_dir)
        return parse_phase_1_candidates(results_dir, exclude)


async def reinit(
    client,
    strategy: QueryStrategy | None,
    endpoint: Endpoint,
    old_bis: list[str],
    now: datetime,
) -> Epoch | None:
    """Returns the new epoch, or None if fewer than min_bis BIs were found."""
    r = config.bi.reinit

    survivors: list[str] = []
    if old_bis:
        reprobe, _ = await sample_prompts(
            client, endpoint, strategy, old_bis, r.reprobe_samples
        )
        survivors = [p for p, s in reprobe.items() if len({tok for _, tok in s}) > 1]
        logger.info(
            f"{endpoint}: {len(survivors)}/{len(old_bis)} BIs survived re-probe"
        )

    candidates = survivors
    if len(candidates) < r.top_k_bis:
        candidates = candidates + await discover_candidates(
            endpoint, exclude=candidates
        )
    # Rank the top-k among at most target_border_inputs candidates; collecting
    # references for more would roughly double onboarding cost for nothing.
    candidates = candidates[: config.bi.phase_1.target_border_inputs]

    if not candidates:
        return None

    reference, n_errors = await sample_prompts(
        client, endpoint, strategy, candidates, r.reference_samples
    )
    if n_errors:
        logger.warning(f"{endpoint}: {n_errors} errors during reference collection")
    reference = {p: s for p, s in reference.items() if s}
    keep = select_top_bis(reference, r.top_k_bis)
    if len(keep) < r.min_bis:
        logger.warning(
            f"{endpoint}: only {len(keep)} BIs after re-init, below min {r.min_bis}"
        )
        return None
    epoch = Epoch(
        start=now,
        border_inputs=keep,
        reference={p: reference[p] for p in keep},
    )
    _persist_reference(endpoint, epoch)
    return epoch


def _persist_reference(endpoint: Endpoint, epoch: Epoch) -> None:
    """Merge the epoch's reference batch into the endpoint's phase-2 monthly file.

    Without this, epoch_tv_series would skip the first real detection day (it
    treats the earliest timestamp in the file as the reference batch).
    """
    batch_key = epoch.start.replace(microsecond=0).isoformat()
    path = get_output_path(endpoint, epoch.start.strftime("%Y-%m"))
    existing = load_existing_results(path)
    for prompt, samples in epoch.reference.items():
        # On a change-firing day this intentionally overwrites the monitor's
        # 10-sample batch at the same key for surviving BIs: the old epoch is
        # already closed, and the new epoch needs the reference here.
        existing.setdefault(prompt, {})[batch_key] = samples
    save_results(path, existing)
