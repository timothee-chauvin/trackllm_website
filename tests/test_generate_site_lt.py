import json
from datetime import datetime, timezone
from pathlib import Path

from conftest import write_month_dir
from trackllm_website.generate_site.lt import discover_lt_endpoints, get_last_query_date


def _make_lt_endpoint(root: Path, slug: str, model: str, provider: str):
    d = root / slug / "default"
    d.mkdir(parents=True)
    (d / "info.json").write_text(
        json.dumps({"prompt": "hi", "endpoint": {"model": model, "provider": provider}})
    )
    write_month_dir(d, "2026-06", [["24 10:00:00", 0]])


def _make_prompt(endpoint_dir: Path, slug: str, months: dict[str, list]) -> Path:
    d = endpoint_dir / slug
    d.mkdir(parents=True)
    (d / "info.json").write_text(
        json.dumps({"prompt": slug, "endpoint": {"model": "m/a", "provider": "p"}})
    )
    for month, queries in months.items():
        write_month_dir(d, month, queries)
    return d


def test_discover_lt_endpoints(tmp_path):
    _make_lt_endpoint(tmp_path, "m2fa23p", "m/a", "p")
    eps = discover_lt_endpoints(tmp_path)
    assert len(eps) == 1
    assert eps[0].model == "m/a"
    assert eps[0].provider == "p"
    assert eps[0].prompts[0].months == ["2026-06"]


def test_last_query_date_falls_back_to_older_month(tmp_path):
    """A newest month holding only errors must not hide the successes before it."""
    _make_prompt(
        tmp_path,
        "a_1",
        {"2026-06": [["30 09:00:00", 0]], "2026-07": [["02 08:00:00", "e0"]]},
    )
    assert get_last_query_date(tmp_path) == datetime(
        2026, 6, 30, 9, 0, 0, tzinfo=timezone.utc
    )


def test_last_query_date_scans_every_prompt(tmp_path):
    """Each prompt is scanned on its own: a date found for one prompt must not
    cut short the month scan of the next one."""
    _make_prompt(tmp_path, "a_1", {"2026-03": [["15 12:00:00", 0]]})
    _make_prompt(
        tmp_path,
        "b_2",
        {"2026-06": [["30 09:00:00", 0]], "2026-07": [["02 08:00:00", "e0"]]},
    )
    assert get_last_query_date(tmp_path) == datetime(
        2026, 6, 30, 9, 0, 0, tzinfo=timezone.utc
    )


def test_endpoint_with_all_errors_in_newest_month_is_not_stale(tmp_path):
    endpoint_dir = tmp_path / "m2fa23p"
    _make_prompt(endpoint_dir, "a_1", {"2026-03": [["15 12:00:00", 0]]})
    _make_prompt(
        endpoint_dir,
        "b_2",
        {"2026-06": [["30 09:00:00", 0]], "2026-07": [["02 08:00:00", "e0"]]},
    )
    (info,) = discover_lt_endpoints(tmp_path)
    assert info.last_query_str == "2026-06-30"


def test_last_query_date_none_when_only_errors(tmp_path):
    _make_prompt(tmp_path, "a_1", {"2026-07": [["02 08:00:00", "e0"]]})
    assert get_last_query_date(tmp_path) is None


def test_last_query_date_takes_year_and_month_from_the_directory(tmp_path):
    """The stored timestamp is a bare day of month; parsing it without the
    directory's year is deprecated on 3.13 and raises from 3.15."""
    _make_prompt(tmp_path, "a_1", {"2028-02": [["29 10:00:00", 0]]})
    assert get_last_query_date(tmp_path) == datetime(
        2028, 2, 29, 10, 0, 0, tzinfo=timezone.utc
    )
