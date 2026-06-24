"""Build-time derivation of per-endpoint B3IT display data."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from trackllm_website.bi.analyze import load_phase2_results
from trackllm_website.bi.detection import adaptive_transitions, epoch_tv_series, is_unstable
from trackllm_website.bi.state import Epoch, EndpointBIState, load_all_states


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


def current_epoch_results(state: EndpointBIState, results: dict) -> tuple[Epoch | None, dict]:
    """Mirror monitor.decide (lines 54-79): last open epoch + filtered results."""
    epoch = state.current_epoch
    if epoch is None:
        return None, {}
    epoch_results = {
        p: {
            ts: s
            for ts, s in results.get(p, {}).items()
            if datetime.fromisoformat(ts) >= epoch.start
        }
        for p in epoch.border_inputs
    }
    return epoch, epoch_results


def _iso(dt) -> str | None:
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def derive_b3it(state: EndpointBIState, results: dict) -> B3ITView:
    epoch, epoch_results = current_epoch_results(state, results)
    tv = epoch_tv_series(epoch.reference, epoch_results) if epoch else []
    changes = adaptive_transitions(tv) if tv else []
    return B3ITView(
        slug=state.slug,
        model=state.endpoint.model,
        provider=state.endpoint.provider,
        status=state.status,
        retired_reason=state.retired.reason if state.retired else None,
        n_bis=len(epoch.border_inputs) if epoch else 0,
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
