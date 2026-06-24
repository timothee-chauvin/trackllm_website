import math
from dataclasses import asdict, dataclass
from datetime import datetime

# LT sigmas at/above this threshold (or null/non-finite) are effectively infinite
# and shown as ∞ — they arise from near-zero baseline variance in the detector.
SIGMA_INF_THRESHOLD = 1e4


def _lt_magnitude_display(sigma: float | None) -> str:
    if sigma is None or not math.isfinite(sigma) or abs(sigma) >= SIGMA_INF_THRESHOLD:
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
        for epoch in view.epochs:
            if epoch.get("end_reason") == "change_detected" and epoch.get(
                "change_date"
            ):
                events.append(
                    ChangeEvent(
                        epoch["change_date"],
                        slug,
                        view.model,
                        view.provider,
                        "B3IT",
                        None,
                        "",
                    )
                )
    events.sort(key=lambda e: datetime.fromisoformat(e.date), reverse=True)
    return events


def to_json(events) -> list[dict]:
    return [asdict(e) for e in events]
