from trackllm_website.generate_site.org import build_org_views
from trackllm_website.util import slugify


def _model_view(model: str, endpoints: list[tuple[str, str]], changes: list[str]):
    """A build_model_views-shaped view: (provider, base) pairs plus change dates."""
    return {
        "model": model,
        "org": model.split("/")[0],
        "date_min": "2026-06-01",
        "date_max": "2026-06-30",
        "n_endpoints": len(endpoints),
        "n_providers": len({base for _, base in endpoints}),
        "n_changed": 1 if changes else 0,
        "max_drift": 0.0,
        "changes": [{"date": d, "method": "lt", "provider": "p"} for d in changes],
        "endpoints": [
            {"provider": provider, "base": base} for provider, base in endpoints
        ],
    }


def test_build_org_views_groups_models_by_org():
    views = build_org_views(
        {
            slugify("m/a"): _model_view("m/a", [("p1", "p1")], ["2026-06-10"]),
            slugify("m/b"): _model_view("m/b", [("p2", "p2")], []),
            slugify("n/c"): _model_view("n/c", [("p1", "p1")], []),
        }
    )
    assert set(views) == {slugify("m"), slugify("n")}
    m = views[slugify("m")]
    assert m["name"] == "m"
    assert m["n_models"] == 2
    assert m["n_endpoints"] == 2
    assert m["n_changes"] == 1
    assert m["n_changed"] == 1
    assert {row["name"] for row in m["models"]} == {"a", "b"}


def test_org_model_rows_link_to_the_model_page_slug():
    model_slug = slugify("m/a")
    views = build_org_views({model_slug: _model_view("m/a", [("p1", "p1")], [])})
    (row,) = views[slugify("m")]["models"]
    assert row["slug"] == model_slug
    assert row["name"] == "a"


def test_org_providers_are_unioned_not_summed():
    """One provider serving three of an org's models is one provider."""
    views = build_org_views(
        {
            slugify("m/a"): _model_view("m/a", [("p1", "p1"), ("p1/fp8", "p1")], []),
            slugify("m/b"): _model_view("m/b", [("p1", "p1")], []),
        }
    )
    org = views[slugify("m")]
    assert org["n_providers"] == 1
    assert org["n_endpoints"] == 3


def test_org_models_sort_most_changed_first():
    views = build_org_views(
        {
            slugify("m/quiet"): _model_view("m/quiet", [("p1", "p1")], []),
            slugify("m/loud"): _model_view(
                "m/loud", [("p1", "p1")], ["2026-06-10", "2026-06-20"]
            ),
            slugify("m/mid"): _model_view("m/mid", [("p1", "p1")], ["2026-06-05"]),
        }
    )
    assert [m["name"] for m in views[slugify("m")]["models"]] == [
        "loud",
        "mid",
        "quiet",
    ]


def test_org_row_carries_last_change_date():
    views = build_org_views(
        {
            slugify("m/a"): _model_view(
                "m/a", [("p1", "p1")], ["2026-06-05", "2026-06-20"]
            ),
            slugify("m/b"): _model_view("m/b", [("p1", "p1")], []),
        }
    )
    rows = {m["name"]: m for m in views[slugify("m")]["models"]}
    assert rows["a"]["last_change"] == "2026-06-20"
    assert rows["b"]["last_change"] is None
