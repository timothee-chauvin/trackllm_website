"""Recover changes missed in historical epochs (run manually, results committed).

The live scan only covers open epochs inside its batch window; closed epochs
and epochs past the window are scanned here once, by recursive binary
segmentation at a stricter alpha. Events not near an adaptive onset or the
epoch's closure are written to website/data/b3it/scan_backfill.json, which
the site build merges into the change log.
"""

from datetime import datetime, timedelta

import fire
import orjson

from trackllm_website.bi.results import load_phase2_results
from trackllm_website.bi.detection import (
    adaptive_transitions,
    epoch_tv_series,
    is_unstable,
)
from trackllm_website.bi.scan import ScanEvent, scan_pvalue
from trackllm_website.bi.state import Epoch, load_all_states
from trackllm_website.config import config, logger


def segment_events(results: dict) -> list[ScanEvent]:
    """Recursive binary segmentation: all splits clearing backfill_alpha."""
    event = scan_pvalue(results)
    if event is None or event.p_value > config.bi.scan.backfill_alpha:
        return []
    left = {
        p: {ts: s for ts, s in b.items() if ts < event.split_ts}
        for p, b in results.items()
    }
    right = {
        p: {ts: s for ts, s in b.items() if ts >= event.split_ts}
        for p, b in results.items()
    }
    return segment_events(left) + [event] + segment_events(right)


def spaced_events(events: list[ScanEvent], days: int) -> list[ScanEvent]:
    """Greedily keep events at least `days` apart: a rolling transition can
    produce several nearby splits that are one change operationally."""
    kept: list[ScanEvent] = []
    for e in events:
        if not kept or (
            datetime.fromisoformat(e.split_ts)
            - datetime.fromisoformat(kept[-1].split_ts)
        ) > timedelta(days=days):
            kept.append(e)
    return kept


def already_logged(epoch: Epoch, tv: list) -> bool:
    """Epochs with any logged change are excluded: a backfill split there is
    almost always an earlier phase of the same (ramp-shaped) change."""
    return epoch.change_date is not None or bool(adaptive_transitions(tv))


def eligible(epoch: Epoch, n_batches: int) -> bool:
    """Open epochs inside the live scan window are the live scan's job."""
    return epoch.end is not None or n_batches > config.bi.scan.max_batches


def main() -> None:
    out_path = config.bi.data_dir / "scan_backfill.json"
    states = load_all_states(config.bi.state_dir)
    out: dict[str, list[dict]] = {}
    n_scanned = n_unstable = 0
    for slug, state in sorted(states.items()):
        phase2 = config.bi.phase_2_dir / slug
        if not phase2.exists():
            continue
        results = load_phase2_results(phase2)
        for epoch in state.epochs:
            ref = {p: s for p, s in epoch.reference.items() if s}
            if not ref:
                continue
            er = epoch.filter_results(results)
            if epoch.end is not None:
                er = {
                    p: {
                        ts: s
                        for ts, s in b.items()
                        if datetime.fromisoformat(ts) <= epoch.end
                    }
                    for p, b in er.items()
                }
            # All border inputs, not the ranked top-k: a change confined to a
            # reference-imbalanced prompt (qwen-coder@chutes: 'Sure' 96% ->
            # 'Certainly') ranks last and would be dropped by the ranking.
            n = len({ts for b in er.values() for ts in b})
            if not eligible(epoch, n):
                continue
            if already_logged(epoch, epoch_tv_series(ref, er)):
                continue
            if is_unstable(er):
                n_unstable += 1
                continue
            n_scanned += 1
            events = spaced_events(
                segment_events(er), config.bi.scan.backfill_min_separation_days
            )
            if events:
                out.setdefault(slug, []).extend(
                    {
                        "epoch_start": epoch.start.isoformat(),
                        "date": e.split_ts,
                        "p_value": e.p_value,
                    }
                    for e in events
                )
    out_path.write_bytes(orjson.dumps(out, option=orjson.OPT_INDENT_2))
    total = sum(len(v) for v in out.values())
    logger.info(
        f"scanned {n_scanned} epochs ({n_unstable} skipped as unstable): "
        f"{total} backfill events -> {out_path}"
    )


if __name__ == "__main__":
    fire.Fire(main)
