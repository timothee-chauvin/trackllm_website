"""LT change-event log: current detection, with stable first-detected dates."""

from datetime import datetime
from pathlib import Path

import orjson
from pydantic import BaseModel

from trackllm_website.lt_scores import PEAK_DISTANCE, ChangePoint, LTScores
from trackllm_website.util import atomic_write_bytes

EVENTS_FILENAME = "lt_changes.json"


class LTChangeEvent(BaseModel):
    endpoint: str
    index: int
    date: datetime
    sigma: float | None  # mirrors ChangePoint.sigma; None when deviation undefined
    first_detected: datetime


def merge_events(
    slug: str,
    existing: list[LTChangeEvent],
    changes: list[ChangePoint],
    dates: list[datetime],
    now: datetime,
) -> list[LTChangeEvent]:
    """A recomputed change within PEAK_DISTANCE indices of an existing event is
    the same event: keep first_detected, refresh index/date/sigma.

    Each event is claimed by at most one change, so two nearby changes cannot
    collapse onto one event. An event no current change claims is dropped:
    detection is authoritative, and an event it no longer produces would
    otherwise outlive the bug that created it.
    """
    unclaimed = list(existing)
    merged: list[LTChangeEvent] = []
    for cp in changes:
        match = next(
            (e for e in unclaimed if abs(e.index - cp.index) <= PEAK_DISTANCE), None
        )
        if match is not None:
            unclaimed.remove(match)
            match.index = cp.index
            match.date = dates[cp.index]
            match.sigma = cp.sigma
            merged.append(match)
        else:
            merged.append(
                LTChangeEvent(
                    endpoint=slug,
                    index=cp.index,
                    date=dates[cp.index],
                    sigma=cp.sigma,
                    first_detected=now,
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
