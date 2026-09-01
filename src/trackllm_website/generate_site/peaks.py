"""How far a change moved.

B3IT: the peak TV its series reached after the change (TV is already distance
from the epoch's own reference). LT: the shift in drift level across the change,
|mean after - mean before| -- drift is distance from one fixed reference, so for
any change but the first, "level after" would include earlier changes.

One rule and one set of windows for every surface (feed.py, timeline.py), so a
magnitude can never disagree between pages.
"""

LT_PEAK_WINDOW = 20
B3IT_PEAK_WINDOW = 8


def shift_from(day: str, pairs: list[tuple[str, float]], window: int) -> float | None:
    """|mean of the `window` points on/after `day` - mean of the `window` before|.

    None when either side is empty: the level reached is then unknown.
    """
    before = [v for d, v in pairs if d < day][-window:]
    after = [v for d, v in pairs if d >= day][:window]
    if not before or not after:
        return None
    return abs(sum(after) / len(after) - sum(before) / len(before))
