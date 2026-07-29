"""Which endpoints the site shows at all.

An endpoint with no plottable observation has nothing to tell a reader: no drift
line, no changes, no status worth trusting -- so it is not an endpoint as far as
the site is concerned. Dropping those once, here, is what keeps the fleet
consistent: the directory, the model, provider and org pages and the endpoint
pages themselves all derive from the two dicts this returns, and none of them
re-decides what counts as tracked.

Two ways to have nothing. An LT endpoint whose queries all errored keeps its
directory layout (info.json, month dirs) but never gets an lt_scores.json. A
B3IT endpoint retired before its first post-reference batch keeps its state file
but has no phase-2 results, and so an empty tv_series -- the series drops the
epoch's reference batch.

That last one cuts both ways: a freshly onboarded endpoint is seriesless too,
and is held back until its second batch lands. That is the same reading, not an
exception -- until then there is nothing to draw.
"""

from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.lt import EndpointInfo, LTData


def with_observations(
    lt_by_slug: dict[str, EndpointInfo],
    lt_data: dict[str, LTData],
    b3it_views: dict[str, B3ITView],
) -> tuple[dict[str, EndpointInfo], dict[str, B3ITView]]:
    """Both fleets, each restricted to the endpoints carrying a series.

    Per method, not per endpoint: one whose LT queries all errored but whose
    B3IT series is fine stays on the site as B3IT-only, rather than keeping an
    lt badge with nothing behind it.
    """
    return (
        {slug: ep for slug, ep in lt_by_slug.items() if slug in lt_data},
        {slug: view for slug, view in b3it_views.items() if view.tv_series["dates"]},
    )
