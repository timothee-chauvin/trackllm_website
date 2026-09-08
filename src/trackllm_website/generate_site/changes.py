from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass
class ChangeEvent:
    date: str
    slug: str
    model: str
    provider: str
    method: str
    # LT: the level shift across the change (lt_drift.level_shift), the number the
    # publication gate passed on. B3IT: None here; feed.py takes the TV shift from
    # the view, the number its detector gated on.
    magnitude: float | None


def merge_changes(lt_changes, lt_by_slug, b3it_views) -> list[ChangeEvent]:
    events: list[ChangeEvent] = []
    for slug, evs in lt_changes.items():
        ep = lt_by_slug.get(slug)
        model = ep.model if ep else slug
        provider = ep.provider if ep else ""
        # lt_events logs every detected changepoint; only those whose level shift
        # cleared the gate are published.
        for ev in evs:
            if not ev["published"]:
                continue
            events.append(
                ChangeEvent(ev["date"], slug, model, provider, "LT", ev["level_shift"])
            )
    for slug, view in b3it_views.items():
        seen: set[datetime] = set()

        def _emit(date: str) -> None:
            key = datetime.fromisoformat(date)
            if key in seen:
                return
            seen.add(key)
            events.append(
                ChangeEvent(date, slug, view.model, view.provider, "B3IT", None)
            )

        # Authoritative epoch closures (live detector).
        for epoch in view.epochs:
            if (
                epoch.get("end_reason") == "change_detected"
                and epoch.get("change_date")
                and epoch["change_date"] not in view.gated_dates
            ):
                _emit(epoch["change_date"])
        # Onsets derived from the TV series of every epoch, including closed and
        # migrated legacy epochs whose changes never triggered a closure. This is
        # the entire pre-detector history; dedup guards against a live-detected
        # change being counted twice.
        for change in getattr(view, "changes", None) or []:
            _emit(change["date"])
    events.sort(key=lambda e: datetime.fromisoformat(e.date), reverse=True)
    return events


def to_json(events) -> list[dict]:
    return [asdict(e) for e in events]
