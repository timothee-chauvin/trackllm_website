from datetime import date, datetime, timedelta, timezone

import pytest

from trackllm_website.config import HeroConfig
from trackllm_website.generate_site.b3it import B3ITView
from trackllm_website.generate_site.hero import HERO_MIN_POINTS, build_hero

NOW = datetime(2026, 3, 1, tzinfo=timezone.utc)
DAY0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _series(n: int, jump_at: int, low: float, high: float):
    return [(DAY0 + timedelta(days=i), low if i < jump_at else high) for i in range(n)]


def _pin(start: str, end: str, method: str = "lt", slug: str = "ep") -> HeroConfig:
    return HeroConfig(
        slug=slug,
        method=method,
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
    )


def _change(slug: str = "ep", day: int = 20) -> dict:
    return {
        "date": (DAY0 + timedelta(days=day)).isoformat(),
        "slug": slug,
        "model": "org/model-x",
        "provider": "p/fp8",
        "method": "LT",
        "magnitude": 99.0,
        "magnitude_display": "99σ",
    }


def _view(slug: str, series) -> B3ITView:
    return B3ITView(
        slug=slug,
        model="org/model-x",
        provider="p/fp8",
        status="monitoring",
        retired_reason=None,
        n_bis=3,
        unstable=False,
        epochs=[],
        tv_series={
            "dates": [d.isoformat() for d, _ in series],
            "values": [v for _, v in series],
        },
        changes=[],
        last_query=series[-1][0].isoformat(),
    )


def test_draws_the_pinned_endpoint_over_the_pinned_window():
    drift = {"ep": _series(41, 20, 0.05, 1.2)}
    hero = build_hero([_change()], drift, {}, NOW, _pin("2026-01-01", "2026-02-10"))
    assert hero["slug"] == "ep"
    assert hero["start"] == "2026-01-01"
    assert hero["end"] == "2026-02-10"
    assert len(hero["values"]) == 41


def test_window_clips_to_the_configured_dates():
    """Points outside the vetted range must not reach the page -- the whole point of
    pinning is that the picture cannot drift as new data lands."""
    drift = {"ep": _series(60, 20, 0.05, 1.2)}
    hero = build_hero([_change()], drift, {}, NOW, _pin("2026-01-06", "2026-01-25"))
    assert hero["start"] == "2026-01-06"
    assert hero["end"] == "2026-01-25"
    assert len(hero["values"]) == 20


def test_changefrac_lands_on_the_step():
    drift = {"ep": _series(41, 20, 0.05, 1.2)}
    hero = build_hero([_change()], drift, {}, NOW, _pin("2026-01-01", "2026-02-10"))
    k = round(hero["changeFrac"] * (len(hero["values"]) - 1))
    assert hero["values"][k - 1] == 0.05
    assert hero["values"][k] == 1.2


def test_unknown_slug_raises_rather_than_dropping_the_hero():
    """A stale pin must break the build, not quietly blank the site's first claim."""
    with pytest.raises(ValueError, match="no lt series"):
        build_hero([_change()], {}, {}, NOW, _pin("2026-01-01", "2026-02-10"))


def test_window_with_too_few_points_raises():
    drift = {"ep": _series(41, 20, 0.05, 1.2)}
    with pytest.raises(ValueError, match="only .* points"):
        build_hero(
            [_change()],
            drift,
            {},
            NOW,
            _pin(
                "2026-01-01",
                (DAY0 + timedelta(days=HERO_MIN_POINTS - 2)).date().isoformat(),
            ),
        )


def test_pin_without_a_matching_change_raises():
    """The card names a detection date, so a window with no detected change in it
    would have to invent one."""
    drift = {"ep": _series(41, 20, 0.05, 1.2)}
    with pytest.raises(ValueError, match="no LT change"):
        build_hero([], drift, {}, NOW, _pin("2026-01-01", "2026-02-10"))


def test_b3it_pin_reads_the_view_series():
    series = _series(41, 20, 0.05, 0.9)
    change = _change(day=20) | {"method": "B3IT"}
    hero = build_hero(
        [change],
        {},
        {"ep": _view("ep", series)},
        NOW,
        _pin("2026-01-01", "2026-02-10", "b3it"),
    )
    assert hero["method"] == "b3it"
    assert hero["magnitude"] == 0.9


def test_payload_carries_links_baseline_and_peak():
    drift = {"ep": _series(41, 20, 0.05, 1.2)}
    hero = build_hero([_change()], drift, {}, NOW, _pin("2026-01-01", "2026-02-10"))
    assert hero["model"] == "model-x"
    assert hero["org"] == "org"
    assert hero["modelSlug"] == "org2fmodel-x"
    assert hero["providerSlug"] == "p"
    assert hero["method"] == "lt"
    assert hero["baseline"] == 0.05
    assert hero["magnitude"] == 1.2
    assert hero["date"] == "2026-01-21"
    assert hero["daysAgo"] == (NOW - (DAY0 + timedelta(days=20))).days
    assert hero["yMax"] > hero["magnitude"]


def test_values_are_not_downsampled():
    """Downsampling averages neighbours, which is the smoothing we just removed."""
    drift = {"ep": [(DAY0 + timedelta(days=i), 0.1 + i * 0.01) for i in range(120)]}
    hero = build_hero(
        [_change(day=60)], drift, {}, NOW, _pin("2026-01-01", "2026-04-30")
    )
    assert len(hero["values"]) == 120
    assert hero["values"] == [round(0.1 + i * 0.01, 4) for i in range(120)]
