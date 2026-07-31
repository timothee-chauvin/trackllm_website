"""The build's clock: the newest observation the site has, across both methods.

Every surface ages a change against this instant -- "14d ago" on the feed, the
30-day windows, the retired/stable cutoffs -- so all of them must read the same
one, and it must span the methods. Taking the newest LT observation alone dated
a B3IT change made after it in the future: a negative age published to the feed,
and a change silently dropped from the 30-day counts.
"""

from datetime import datetime

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.lt import LTData, latest_date


def site_now(
    lt_data: dict[str, LTData], b3it_views: dict[str, B3ITView]
) -> datetime | None:
    """The most recent observation of either method, or None if there is none."""
    lt_last = latest_date(lt_data)
    observations = [lt_last] if lt_last is not None else []
    observations += [
        datetime.fromisoformat(view.tv_series["dates"][-1])
        for view in b3it_views.values()
        if view.tv_series["dates"]
    ]
    return max(observations, default=None)
