"""Build-time derivation of per-endpoint B3IT display data."""

from dataclasses import dataclass
from pathlib import Path

from trackllm_website.bi.analyze import load_phase2_results
from trackllm_website.bi.detection import (
    adaptive_transitions,
    epoch_tv_series,
    is_unstable,
    select_top_bis,
)
from trackllm_website.bi.state import Epoch, EndpointBIState, load_all_states
from trackllm_website.config import config


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


def derive_b3it(state: EndpointBIState, results: dict) -> B3ITView:
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
    return B3ITView(
        slug=state.slug,
        model=state.endpoint.model,
        provider=state.endpoint.provider,
        status=state.status,
        retired_reason=state.retired.reason if state.retired else None,
        n_bis=len(display_epoch.border_inputs) if display_epoch else 0,
        unstable=is_unstable(tv),
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
        changes=[{"date": ts, "kind": "onset"} for ts in changes],
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


def discover_b3it_views(state_dir: Path, phase_2_dir: Path) -> dict[str, B3ITView]:
    views: dict[str, B3ITView] = {}
    if not state_dir.exists():
        return views
    for state in load_all_states(state_dir).values():
        results = load_phase2_results(phase_2_dir / state.slug)
        views[state.slug] = derive_b3it(state, results)
    return views
