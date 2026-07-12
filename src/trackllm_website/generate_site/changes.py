from dataclasses import asdict, dataclass
from datetime import datetime

from trackllm_website.lt_scores import normalize_sigma


def _lt_magnitude_display(sigma: float | None) -> str:
    # The detector already stores effectively-infinite sigmas as null; the
    # re-normalization here only shields against pre-migration legacy entries.
    if sigma is None or normalize_sigma(sigma) is None:
        return "∞σ"
    return f"{sigma:.0f}σ"


@dataclass
class ChangeEvent:
    date: str
    slug: str
    model: str
    provider: str
    method: str
    magnitude: float | None
    magnitude_display: str


def merge_changes(lt_changes, lt_by_slug, b3it_views) -> list[ChangeEvent]:
    events: list[ChangeEvent] = []
    for slug, evs in lt_changes.items():
        ep = lt_by_slug.get(slug)
        model = ep.model if ep else slug
        provider = ep.provider if ep else ""
        for ev in evs:
            sigma = ev["sigma"]
            events.append(
                ChangeEvent(
                    ev["date"],
                    slug,
                    model,
                    provider,
                    "LT",
                    sigma,
                    _lt_magnitude_display(sigma),
                )
            )
    for slug, view in b3it_views.items():
        seen: set[datetime] = set()

        def _emit(date: str) -> None:
            key = datetime.fromisoformat(date)
            if key in seen:
                return
            seen.add(key)
            events.append(
                ChangeEvent(date, slug, view.model, view.provider, "B3IT", None, "")
            )

        # Authoritative epoch closures (live detector).
        for epoch in view.epochs:
            if epoch.get("end_reason") == "change_detected" and epoch.get(
                "change_date"
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
