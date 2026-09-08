"""Build-time derivation of per-endpoint B3IT display data."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import orjson

from trackllm_website.bi.results import (
    compute_tv_distance,
    get_distribution,
    load_phase2_results,
)
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
    # The raw votes behind the series, for the chart's hover readout (votes.json,
    # see votes_json): `bis` is the vocabulary of border-input prompts that
    # `references` (one entry per epoch: [[bi index, {token: count, ...}], ...])
    # and `batches` (one per tv_series point: [[bi index, votes, TV vs the
    # reference], ...]) index into, over the epoch's ranked border inputs,
    # tokens by descending count.
    bis: list[str]
    references: list[list]
    batches: list[list]
    changes: list[dict]
    # Change day -> the TV shift its detector gated on (tv_shift on the epoch's
    # full reference series), the headline every surface shows; days the gate
    # rejected are absent.
    change_mags: dict
    # change_date strings (epochs[].change_date format) of scan-detected epoch
    # closures whose TV level shift stayed at or under abs_delta: the epoch was
    # re-initialized, but the site does not publish a change.
    gated_dates: set
    last_query: str | None


def _iso(dt) -> str | None:
    return dt.isoformat().replace("+00:00", "Z") if dt else None


def _gate_shift(gate_tv: list[tuple], ts: str) -> float | None:
    """The number the detectors gate on: bi.detection.tv_shift at `ts` over the
    epoch's full reference series (what monitor.decide and backfill.py score).
    Compared as instants, not strings: change dates are written with a Z suffix
    and batch keys with +00:00. A change in a fresh epoch has no pre-change days;
    TV is already distance from the new reference, so the post level alone is the
    shift there."""
    split = datetime.fromisoformat(ts)
    pre = [v for t, v in gate_tv if datetime.fromisoformat(t) < split]
    post = [v for t, v in gate_tv if datetime.fromisoformat(t) >= split]
    if not post:
        return None
    level = sum(post) / len(post)
    return abs(level - sum(pre) / len(pre)) if pre else level


def ranked_reference(epoch: Epoch) -> dict:
    """The epoch's reference restricted to its top-k ranked border inputs.

    Production monitoring re-initialises every epoch to the top-k BIs
    (``bi.reinit.top_k_bis``); legacy epochs migrated from before the detector
    existed still carry the full unranked candidate set. Applying the same
    ranking here is load-bearing: a change confined to a few BIs is diluted
    below the detection threshold across the full set, so ranking is what
    surfaces those historical changes on the site.
    """
    top = select_top_bis(epoch.reference, config.bi.reinit.top_k_bis)
    return {p: epoch.reference[p] for p in top}


# A vote is shown, not analysed, in votes.json: a token longer than this is cut
# (and merged with the others sharing its head). The per-input TV beside it is
# computed on the full tokens first.
TOKEN_CHARS = 40


def _votes(samples: list, bi_index: dict[str, int], prompt: str) -> list:
    shown: Counter = Counter()
    for t, n in get_distribution(samples).items():
        shown[t if len(t) <= TOKEN_CHARS else f"{t[:TOKEN_CHARS]}…"] += n
    return [
        bi_index.setdefault(prompt, len(bi_index)),
        dict(sorted(shown.items(), key=lambda kv: (-kv[1], kv[0]))),
    ]


def _batch_votes(reference: dict, ep_results: dict, ts: str, bi_index: dict) -> list:
    """[bi index, votes, TV vs the reference] per ranked border input sampled at `ts`."""
    out = []
    for p, ref_samples in reference.items():
        samples = ep_results.get(p, {}).get(ts)
        if not samples:
            continue
        tv = compute_tv_distance(
            get_distribution(ref_samples), get_distribution(samples)
        )
        out.append(
            [*_votes(samples, bi_index, p), round(tv, 3) if tv is not None else None]
        )
    return out


def _detector(epoch: Epoch) -> str | None:
    return (epoch.params or {}).get("detector")


def derive_b3it(
    state: EndpointBIState, results: dict, backfill: list[dict]
) -> B3ITView:
    """Derive the full B3IT timeline across all epochs, not just the open one.

    Iterating every epoch (each against its own ranked reference) is what makes
    closed and retired endpoints — the entire pre-detector history — visible on
    the site.
    """
    tv: list[tuple] = []
    bi_index: dict[str, int] = {}
    batches: list[list] = []
    references: list[list] = []
    changes: list = []
    change_mags: dict[str, float | None] = {}
    gated_dates: set[str] = set()
    abs_delta = config.bi.detection.abs_delta
    for epoch in state.epochs:
        ep_results = epoch.filter_results(results)
        reference = ranked_reference(epoch) if epoch.reference else {}
        ep_tv = epoch_tv_series(reference, ep_results) if reference else []
        tv.extend(ep_tv)
        references.append([_votes(s, bi_index, p) for p, s in reference.items()])
        batches.extend(
            _batch_votes(reference, ep_results, ts, bi_index) for ts, _ in ep_tv
        )
        # Magnitudes come from the series the detectors scored, the epoch's full
        # reference (monitor.decide, backfill.py), not the ranked one the site
        # plots: what is published is the number the gate used.
        gate_tv = epoch_tv_series(epoch.reference, ep_results)
        for ts in adaptive_transitions(ep_tv):
            changes.append({"date": ts, "kind": "onset", "detector": "adaptive"})
            change_mags[ts[:10]] = _gate_shift(gate_tv, ts)
        if epoch.end_reason == "change_detected" and epoch.change_date:
            cd = _iso(epoch.change_date)
            # Scan-detected closures are held to the adaptive rule's abs_delta: a
            # permutation-significant split that moved the output distribution by
            # less than a visible change re-initialized the epoch but is not
            # published as a change.
            gate = _gate_shift(gate_tv, cd)
            if _detector(epoch) == "scan" and (gate is None or gate <= abs_delta):
                gated_dates.add(cd)
            else:
                change_mags[cd[:10]] = gate
        # Backfill events were already gated when backfill.py wrote them; here
        # they only get a magnitude (None when the epoch's data is gone).
        for ev in backfill:
            if ev["date"][:10] < epoch.start.isoformat()[:10] or (
                epoch.end and ev["date"][:10] > epoch.end.isoformat()[:10]
            ):
                continue
            change_mags[ev["date"][:10]] = _gate_shift(gate_tv, ev["date"])

    display_epoch = state.current_epoch or (state.epochs[-1] if state.epochs else None)
    unstable = False
    if display_epoch is not None and display_epoch.reference:
        top = set(ranked_reference(display_epoch))
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
                "detector": _detector(e),
                "n_ref": len(ref),
            }
            for e, ref in zip(state.epochs, references)
        ],
        tv_series={"dates": [ts for ts, _ in tv], "values": [v for _, v in tv]},
        bis=list(bi_index),
        references=references,
        batches=batches,
        changes=changes
        + [{"date": ev["date"], "kind": "scan", "detector": "scan"} for ev in backfill],
        change_mags=change_mags,
        gated_dates=gated_dates,
        # From the raw results, not the TV series: the series drops the epoch's
        # reference batch, so a freshly re-initialized endpoint has none.
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


def votes_json(view: B3ITView) -> dict:
    """The raw votes (B3ITView.bis), a file of their own: fetched on the readout's
    first request, since they outweigh the series many times over."""
    return {"bis": view.bis, "reference": view.references, "batches": view.batches}


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
