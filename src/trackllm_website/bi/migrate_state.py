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
from trackllm_website.config import Endpoint, config, logger
from trackllm_website.util import endpoint_from_slug


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def migrate_endpoint(endpoint: Endpoint, results: dict) -> EndpointBIState:
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
        retired=RetiredInfo(
            reason="stalled", since=_parse(last_success), last_recheck=now
        ),
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
    unmatched: list[str] = []
    for d in sorted(config.bi.phase_2_dir.iterdir()):
        if not d.is_dir():
            continue
        results = load_phase2_results(d)
        if not results:
            logger.warning(f"{d.name}: no results, skipping")
            continue
        try:
            endpoint = endpoint_from_slug(d.name)
        except ValueError:
            logger.warning(f"{d.name}: no matching endpoint, skipping")
            unmatched.append(d.name)
            continue
        state = migrate_endpoint(endpoint, results)
        state.save(config.bi.state_dir)
        n += 1
    logger.info(f"Migrated {n} endpoints to {config.bi.state_dir}")
    if unmatched:
        logger.warning(f"Unmatched slugs ({len(unmatched)}): {unmatched}")


if __name__ == "__main__":
    fire.Fire(migrate)
