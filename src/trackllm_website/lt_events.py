"""LT change-event log: current detection, with stable first-detected dates and
the publication gate (lt_drift.LT_MIN_SHIFT on each change's level shift)."""

from datetime import datetime
from pathlib import Path

import orjson
from pydantic import BaseModel

from trackllm_website.lt_drift import LT_MIN_SHIFT
from trackllm_website.lt_scores import PEAK_DISTANCE, ChangePoint, LTScores
from trackllm_website.util import atomic_write_bytes

EVENTS_FILENAME = "lt_changes.json"


class LTChangeEvent(BaseModel):
    endpoint: str
    index: int
    date: datetime
    sigma: float | None  # mirrors ChangePoint.sigma; None when deviation undefined
    first_detected: datetime
    # Defaults only so an lt_changes.json written before the gate existed still
    # loads: the first recompute after that fills both fields for every event.
    level_shift: float | None = None  # mirrors ChangePoint.level_shift
    # Sticky: set the first time the level shift clears LT_MIN_SHIFT, never unset.
    # The shift is a moving mean over the days after the change, so it would
    # otherwise flip a published change on and off as days arrive.
    published: bool = False


def merge_events(
    slug: str,
    existing: list[LTChangeEvent],
    changes: list[ChangePoint],
    dates: list[datetime],
    now: datetime,
) -> list[LTChangeEvent]:
    """A recomputed change within PEAK_DISTANCE indices of an existing event is
    the same event: keep first_detected and published, refresh the rest.

    Each event is claimed by at most one change, so two nearby changes cannot
    collapse onto one event. An event no current change claims is dropped:
    detection is authoritative, and an event it no longer produces would
    otherwise outlive the bug that created it.
    """
    unclaimed = list(existing)
    merged: list[LTChangeEvent] = []
    for cp in changes:
        passes = cp.level_shift is not None and cp.level_shift >= LT_MIN_SHIFT
        match = next(
            (e for e in unclaimed if abs(e.index - cp.index) <= PEAK_DISTANCE), None
        )
        if match is not None:
            unclaimed.remove(match)
            match.index = cp.index
            match.date = dates[cp.index]
            match.sigma = cp.sigma
            match.level_shift = cp.level_shift
            match.published = match.published or passes
            merged.append(match)
        else:
            merged.append(
                LTChangeEvent(
                    endpoint=slug,
                    index=cp.index,
                    date=dates[cp.index],
                    sigma=cp.sigma,
                    first_detected=now,
                    level_shift=cp.level_shift,
                    published=passes,
                )
            )
    merged.sort(key=lambda e: e.index)
    return merged


def update_endpoint_events(
    all_events: dict[str, list[LTChangeEvent]],
    slug: str,
    scores: LTScores,
    now: datetime,
) -> None:
    """Merge an endpoint's recomputed changes into the in-memory event log."""
    all_events[slug] = merge_events(
        slug, all_events.get(slug, []), scores.changes, scores.dates, now
    )


def load_events(path: Path) -> dict[str, list[LTChangeEvent]]:
    if not path.exists():
        return {}
    raw = orjson.loads(path.read_bytes())
    return {
        slug: [LTChangeEvent.model_validate(e) for e in events]
        for slug, events in raw.items()
    }


def save_events(path: Path, events: dict[str, list[LTChangeEvent]]) -> None:
    atomic_write_bytes(
        path,
        orjson.dumps(
            {
                s: [e.model_dump(mode="json") for e in evts]
                for s, evts in events.items()
            },
            option=orjson.OPT_INDENT_2,
        ),
    )
