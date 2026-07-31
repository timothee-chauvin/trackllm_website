"""Build-time derivation of per-endpoint B3IT display data."""

from dataclasses import dataclass
from pathlib import Path

import orjson

from trackllm_website.bi.analyze import load_phase2_results
from trackllm_website.bi.detection import (
    adaptive_transitions,
    epoch_tv_series,
    is_unstable,
    select_top_bis,
)
from trackllm_website.bi.state import Epoch, EndpointBIState, load_all_states
from trackllm_website.config import config
from trackllm_website.generate_site.freshness import last_phase2_query


@dataclass
class B3ITView:
    slug: str
    model: str
    provider: str
    status: str
    retired_reason: str | None
    n_bis: int
    unstable: bool
    epochs: list[dict]
    tv_series: dict
    changes: list[dict]
    last_query: str | None


def _iso(dt) -> str | None:
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def epoch_tv(epoch: Epoch, results: dict) -> list[tuple]:
    """TV series for one epoch, restricted to its top-k ranked border inputs.

    Production monitoring re-initialises every epoch to the top-k BIs
    (``bi.reinit.top_k_bis``); legacy epochs migrated from before the detector
    existed still carry the full unranked candidate set. Applying the same
    ranking here is load-bearing: a change confined to a few BIs is diluted
    below the detection threshold across the full set, so ranking is what
    surfaces those historical changes on the site.
    """
    if not epoch.reference:
        return []
    top = select_top_bis(epoch.reference, config.bi.reinit.top_k_bis)
    reference = {p: epoch.reference[p] for p in top}
    return epoch_tv_series(reference, epoch.filter_results(results))


def derive_b3it(
    state: EndpointBIState, results: dict, backfill: list[dict]
) -> B3ITView:
    """Derive the full B3IT timeline across all epochs, not just the open one.

    Iterating every epoch (each against its own ranked reference) is what makes
    closed and retired endpoints — the entire pre-detector history — visible on
    the site.
    """
    tv: list[tuple] = []
    changes: list = []
    for epoch in state.epochs:
        ep_tv = epoch_tv(epoch, results)
        tv.extend(ep_tv)
        changes.extend(adaptive_transitions(ep_tv))

    display_epoch = state.current_epoch or (state.epochs[-1] if state.epochs else None)
    unstable = False
    if display_epoch is not None and display_epoch.reference:
        top = set(select_top_bis(display_epoch.reference, config.bi.reinit.top_k_bis))
        ep_results = display_epoch.filter_results(results)
        unstable = is_unstable({p: b for p, b in ep_results.items() if p in top})
    return B3ITView(
        slug=state.slug,
        model=state.endpoint.model,
        provider=state.endpoint.provider,
        status=state.status,
        retired_reason=state.retired.reason if state.retired else None,
        n_bis=len(display_epoch.border_inputs) if display_epoch else 0,
        unstable=unstable,
        epochs=[
            {
                "start": _iso(e.start),
                "end": _iso(e.end),
                "end_reason": e.end_reason,
                "change_date": _iso(e.change_date),
            }
            for e in state.epochs
        ],
        tv_series={"dates": [ts for ts, _ in tv], "values": [v for _, v in tv]},
        changes=[{"date": ts, "kind": "onset"} for ts in changes]
        + [{"date": ev["date"], "kind": "scan"} for ev in backfill],
        # From the raw results, not the TV series: the series drops the epoch's
        # reference batch, so a freshly re-initialised endpoint has none.
        last_query=last_phase2_query(results),
    )


def to_json(view: B3ITView) -> dict:
    return {
        "status": view.status,
        "retired_reason": view.retired_reason,
        "n_bis": view.n_bis,
        "unstable": view.unstable,
        "epochs": view.epochs,
        "tv_series": view.tv_series,
        "changes": view.changes,
    }


def discover_b3it_views(
    state_dir: Path, phase_2_dir: Path, backfill_path: Path
) -> dict[str, B3ITView]:
    """Every input is injected, never read from config: a synthetic site (tests,
    fixtures) must not mix in production's state, phase-2 data or scan events."""
    views: dict[str, B3ITView] = {}
    if not state_dir.exists():
        return views
    backfill: dict[str, list[dict]] = (
        orjson.loads(backfill_path.read_bytes()) if backfill_path.exists() else {}
    )
    for state in load_all_states(state_dir).values():
        results = load_phase2_results(phase_2_dir / state.slug)
        views[state.slug] = derive_b3it(state, results, backfill.get(state.slug, []))
    return views
