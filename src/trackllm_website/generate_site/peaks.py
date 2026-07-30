"""How far a change moved: the peak level its series reached after the change.

One rule and one set of windows for every surface (feed.py, model.py, mirrored
in endpoint.ts), so "drift reached" can never disagree between pages.
"""

LT_PEAK_WINDOW = 20
B3IT_PEAK_WINDOW = 8


def peak_from(day: str, pairs: list[tuple[str, float]], window: int) -> float | None:
    """Peak value from the first point on/after `day`, over the next `window` points.

    None when the series has no point on/after `day` -- the level the change
    reached is then unknown, never 0.0.
    """
    on_or_after = [v for d, v in pairs if d >= day][:window]
    return max(on_or_after) if on_or_after else None
