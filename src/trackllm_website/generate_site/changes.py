from dataclasses import asdict, dataclass


@dataclass
class ChangeEvent:
    date: str
    slug: str
    model: str
    provider: str
    method: str
    magnitude: float | None


def merge_changes(lt_changes, lt_by_slug, b3it_views) -> list[ChangeEvent]:
    events: list[ChangeEvent] = []
    for slug, evs in lt_changes.items():
        ep = lt_by_slug.get(slug)
        model = ep.model if ep else slug
        provider = ep.provider if ep else ""
        for ev in evs:
            events.append(ChangeEvent(ev["date"], slug, model, provider, "LT", ev.get("sigma")))
    for slug, view in b3it_views.items():
        for epoch in view.epochs:
            if epoch.get("end_reason") == "change_detected" and epoch.get("change_date"):
                events.append(ChangeEvent(epoch["change_date"], slug, view.model,
                                          view.provider, "B3IT", None))
    events.sort(key=lambda e: e.date, reverse=True)
    return events


def to_json(events) -> list[dict]:
    return [asdict(e) for e in events]
