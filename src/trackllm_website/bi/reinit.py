"""Hybrid re-initialization: re-probe old BIs, top up via phase 1, rank, keep top-k.

Also the onboarding path for new endpoints (old_bis=[]).
"""

import tempfile
from datetime import datetime
from pathlib import Path

from typing import Literal

import orjson
from pydantic import BaseModel

from trackllm_website.bi.common import META_KEY, QueryStrategy
from trackllm_website.bi.detection import select_top_bis
from trackllm_website.bi.phase_1 import check_temperature, phase_1a
from trackllm_website.bi.phase_2 import (
    get_output_path,
    load_existing_results,
    save_results,
)
from trackllm_website.bi.sampling import sample_prompts
from trackllm_website.bi.state import Epoch
from trackllm_website.config import Endpoint, config, logger


class ReinitResult(BaseModel):
    """Outcome of a re-init / onboarding attempt.

    reason distinguishes a normal failure to find enough BIs (no_bis) from an
    endpoint that ignores temperature (bad_temperature), so the caller can cache
    the latter instead of retiring it as a normal no_bis.
    """

    epoch: Epoch | None
    reason: Literal["ok", "no_bis", "bad_temperature"]


def parse_phase_1_results(
    results_dir: Path, exclude: list[str]
) -> tuple[list[str], int]:
    """Parse phase-1a output once: (border-input candidates, distinct prompts sampled).

    Files: {token_count: {prompt: [outputs]}} plus a META_KEY entry; a prompt is a
    border input with >=2 distinct non-empty outputs. border_inputs.json and the
    META_KEY top-level key are skipped. `exclude` filters the candidate list (not
    the denominator).
    """
    excluded = set(exclude)
    candidates: list[str] = []
    sampled: set[str] = set()
    for f in sorted(results_dir.glob("*.json")):  # sorted for deterministic ordering
        if f.name == "border_inputs.json":
            continue
        data = orjson.loads(f.read_bytes())
        for token_count, prompts_dict in data.items():
            if token_count == META_KEY:
                continue
            for prompt, outputs in prompts_dict.items():
                sampled.add(prompt)
                if prompt in excluded:
                    continue
                if len({o for o in outputs if o}) >= 2:
                    candidates.append(prompt)
    return candidates, len(sampled)


async def discover_candidates(
    endpoint: Endpoint, exclude: list[str]
) -> tuple[list[str], float]:
    """Run phase 1a discovery for one endpoint.

    Returns (candidates, prevalence) where prevalence = n_border / n_prompts_sampled,
    so the caller can decide whether to run the temperature gate.
    """
    with tempfile.TemporaryDirectory() as tmp:
        base_dir = Path(tmp)
        await phase_1a([endpoint], 0.0, base_dir)
        results_dir = config.bi.get_phase_1_dir(0.0, base_dir)
        candidates, n_sampled = parse_phase_1_results(results_dir, exclude)
        prevalence = len(candidates) / n_sampled if n_sampled else 0.0
        return candidates, prevalence


async def reinit(
    client,
    strategy: QueryStrategy | None,
    endpoint: Endpoint,
    old_bis: list[str],
    now: datetime,
) -> ReinitResult:
    """Re-init / onboard one endpoint.

    On the onboarding path (old_bis empty), a suspiciously high border-input
    prevalence triggers a T=0-vs-T=1 check; an endpoint that ignores temperature
    is reported as bad_temperature so the caller can cache it.
    """
    r = config.bi.reinit

    survivors: list[str] = []
    if old_bis:
        reprobe, _ = await sample_prompts(
            client, endpoint, strategy, old_bis, r.reprobe_samples, temperature=0.0
        )
        survivors = [p for p, s in reprobe.items() if len({tok for _, tok in s}) > 1]
        logger.info(
            f"{endpoint}: {len(survivors)}/{len(old_bis)} BIs survived re-probe"
        )

    candidates = survivors
    if len(candidates) < r.top_k_bis:
        discovered, prevalence = await discover_candidates(endpoint, exclude=candidates)
        # The gate only makes sense on discovery (old_bis empty): if T=0 doesn't
        # pin the output, the discovered "border inputs" are fake.
        if (
            not old_bis
            and prevalence > config.bi.temperature_gate.prevalence_trigger
            and await check_temperature(client, endpoint, strategy, discovered)
        ):
            return ReinitResult(epoch=None, reason="bad_temperature")
        candidates = candidates + discovered
    # Rank the top-k among at most target_border_inputs candidates; collecting
    # references for more would roughly double onboarding cost for nothing.
    candidates = candidates[: config.bi.phase_1.target_border_inputs]

    if not candidates:
        return ReinitResult(epoch=None, reason="no_bis")

    reference, n_errors = await sample_prompts(
        client, endpoint, strategy, candidates, r.reference_samples, temperature=0.0
    )
    if n_errors:
        logger.warning(f"{endpoint}: {n_errors} errors during reference collection")
    reference = {p: s for p, s in reference.items() if s}
    keep = select_top_bis(reference, r.top_k_bis)
    if len(keep) < r.min_bis:
        logger.warning(
            f"{endpoint}: only {len(keep)} BIs after re-init, below min {r.min_bis}"
        )
        return ReinitResult(epoch=None, reason="no_bis")
    epoch = Epoch(
        start=now,
        border_inputs=keep,
        reference={p: reference[p] for p in keep},
    )
    _persist_reference(endpoint, epoch)
    return ReinitResult(epoch=epoch, reason="ok")


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
