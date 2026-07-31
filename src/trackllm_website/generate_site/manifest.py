"""The endpoint page's manifest: what endpoint.ts must not derive for itself.

Both fields are canonical. `state` is the directory row's own verdict (overview.py
`_row_state`), aged against the build's clock rather than the reader's browser
clock; `changes` is this endpoint's slice of the merged change list, already
levelled by model.py -- not the per-method series' own changepoints, which
double-detect some changes on adjacent days.
"""


def build_manifests(
    overview_rows: list[dict], model_views: dict[str, dict]
) -> dict[str, dict]:
    changes_by_slug = {
        e["slug"]: {
            "lt": e["lt"]["changes"] if e["lt"] else [],
            "b3it": e["b3it"]["changes"] if e["b3it"] else [],
        }
        for view in model_views.values()
        for e in view["endpoints"]
    }
    empty: dict[str, list] = {"lt": [], "b3it": []}
    return {
        row["slug"]: {
            "slug": row["slug"],
            "state": row["status"],
            "changes": changes_by_slug.get(row["slug"], empty),
        }
        for row in overview_rows
    }
