from datetime import datetime, timezone

import numpy as np
import pytest

from trackllm_website.config import Endpoint, config
from trackllm_website.storage import (
    MonthlyData,
    PartialMonthlyDataError,
    Response,
    ResponseError,
    ResponseLogprobs,
    ResultsStorage,
    format_query_date,
    parse_query_date,
)
from trackllm_website.util import slugify

ENDPOINT = Endpoint(api="openrouter", model="org/model", provider="prov", cost=(1, 1))


def _lp(*pairs: tuple[str, float]) -> ResponseLogprobs:
    return ResponseLogprobs(
        tokens=[t for t, _ in pairs], logprobs=[np.float32(lp) for _, lp in pairs]
    )


def _date(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 7, day, hour, 30, 0, tzinfo=timezone.utc)


def _sample_monthly() -> MonthlyData:
    return MonthlyData(
        year=2026,
        month=7,
        logprob_responses=[
            (_date(1), _lp(("a", -0.1), ("b", -2.5))),
            (_date(2), _lp(("a", -0.1), ("b", -2.5))),  # duplicate vector, new date
            (_date(3), _lp(("c", -0.7))),
        ],
        error_responses=[
            (_date(2, hour=12), ResponseError(http_code=429, message="rate limited")),
            (_date(4), ResponseError(http_code=429, message="rate limited")),
            (_date(5), ResponseError(http_code=500, message="boom")),
        ],
    )


def test_serialize_load_round_trip(tmp_path):
    data = _sample_monthly()
    data.serialize(tmp_path)

    loaded = MonthlyData.load_existing(tmp_path, year=2026, month=7)

    assert loaded.logprob_responses == data.logprob_responses
    assert loaded.error_responses == data.error_responses


def test_load_missing_month_dir_returns_empty(tmp_path):
    loaded = MonthlyData.load_existing(tmp_path / "2026-07", year=2026, month=7)
    assert loaded.logprob_responses == []
    assert loaded.error_responses == []


def test_load_partial_file_set_raises(tmp_path):
    """A month dir with some but not all files means a crashed write: loading must
    fail loudly instead of returning empty (which would wipe the month on rewrite)."""
    for missing in (MonthlyData.logprob_filename, MonthlyData.queries_filename):
        month_dir = tmp_path / missing
        _sample_monthly().serialize(month_dir)
        (month_dir / missing).unlink()
        with pytest.raises(PartialMonthlyDataError, match=missing):
            MonthlyData.load_existing(month_dir, year=2026, month=7)


def test_load_error_index_out_of_range_raises_clearly(tmp_path):
    """queries.json referencing "eN" beyond errors.json must not be a bare IndexError."""
    _sample_monthly().serialize(tmp_path)
    (tmp_path / MonthlyData.errors_filename).write_bytes(b'{"seen_errors": []}')
    with pytest.raises(PartialMonthlyDataError, match=MonthlyData.errors_filename):
        MonthlyData.load_existing(tmp_path, year=2026, month=7)


def _response(date: datetime, **kwargs) -> Response:
    return Response(date=date, endpoint=ENDPOINT, prompt="hi", cost=0, **kwargs)


def test_merge_response_dedup():
    data = MonthlyData(year=2026, month=7, logprob_responses=[], error_responses=[])
    lp_response = _response(_date(1), logprobs=_lp(("a", -0.1)))
    err_response = _response(
        _date(2), error=ResponseError(http_code=500, message="boom")
    )

    for response in (lp_response, lp_response, err_response, err_response):
        data.merge_response(response)
    assert len(data.logprob_responses) == 1
    assert len(data.error_responses) == 1

    # Same date but different content is not a duplicate
    data.merge_response(_response(_date(1), logprobs=_lp(("b", -0.2))))
    assert len(data.logprob_responses) == 2


def test_store_response_round_trip(tmp_path):
    storage = ResultsStorage(tmp_path)
    storage.store_response(_response(_date(1), logprobs=_lp(("a", -0.1))))
    storage.store_response(_response(_date(2), logprobs=_lp(("a", -0.2))))

    prompt_dir = storage._get_prompt_dir(ENDPOINT, "hi")
    loaded = MonthlyData.load_existing(prompt_dir / "2026-07", year=2026, month=7)
    assert [d for d, _ in loaded.logprob_responses] == [_date(1), _date(2)]


def test_is_stalled_ignores_non_directory_artifacts(tmp_path):
    """Regression: an `lt_scores.json` file lives in the endpoint dir alongside the
    per-prompt directories. The stall check globs `<slug>/*`, so it must skip
    non-directories instead of iterdir()-ing a file (NotADirectoryError), which
    crashed the daily update-endpoints job for days."""
    storage = ResultsStorage(tmp_path)
    model_dir = tmp_path / slugify(f"{ENDPOINT.model}#{ENDPOINT.provider}")
    model_dir.mkdir(parents=True)
    (model_dir / "lt_scores.json").write_text("{}")  # the artifact that broke it
    (model_dir / "someprompt").mkdir()  # a real prompt directory

    assert storage.is_stalled(ENDPOINT) is False


def test_leap_day_round_trips(tmp_path):
    """queries.json stores the day of month alone; the year and month come from the
    directory, so a Feb 29 must survive the round trip."""
    leap = datetime(2028, 2, 29, 10, 0, 0, tzinfo=timezone.utc)
    data = MonthlyData(
        year=2028,
        month=2,
        logprob_responses=[(leap, _lp(("a", -0.1)))],
        error_responses=[],
    )
    data.serialize(tmp_path)

    loaded = MonthlyData.load_existing(tmp_path, year=2028, month=2)
    assert [d for d, _ in loaded.logprob_responses] == [leap]


def test_query_date_round_trips_through_its_month_dir():
    """The stored string carries no year: the parse must take it from the caller,
    never from strptime's default (deprecated on 3.13, raises from 3.15)."""
    for date in (
        _date(1),
        _date(28, hour=23),
        datetime(2028, 2, 29, tzinfo=timezone.utc),
    ):
        assert parse_query_date(date.year, date.month, format_query_date(date)) == date


def _err_response(day: int, hour: int = 0, month: int = 7) -> Response:
    return _response(
        datetime(2026, month, day, hour, 30, 0, tzinfo=timezone.utc),
        error=ResponseError(http_code=404, message="gone"),
    )


def _store_all(storage: ResultsStorage, responses: list[Response]) -> None:
    for r in responses:
        storage.store_response(r)


def test_is_stalled_after_stall_days_of_all_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(config.api, "lt_stall_days", 7)
    storage = ResultsStorage(tmp_path)
    _store_all(storage, [_err_response(d, h) for d in range(1, 8) for h in (3, 15)])

    assert storage.is_stalled(ENDPOINT) is True


def test_is_stalled_counts_queried_days_not_queries(tmp_path, monkeypatch):
    """Many errors packed into a few days are not a stall: the rule is day-based
    so its verdict does not stretch when the hourly cron fires less often."""
    monkeypatch.setattr(config.api, "lt_stall_days", 7)
    storage = ResultsStorage(tmp_path)
    _store_all(storage, [_err_response(d, h) for d in range(1, 4) for h in range(24)])

    assert storage.is_stalled(ENDPOINT) is False


def test_is_stalled_only_looks_at_most_recent_queried_days(tmp_path, monkeypatch):
    monkeypatch.setattr(config.api, "lt_stall_days", 3)
    storage = ResultsStorage(tmp_path)
    errors = [_err_response(d) for d in (2, 5, 9)]
    _store_all(storage, errors + [_response(_date(1), logprobs=_lp(("a", -0.1)))])
    assert storage.is_stalled(ENDPOINT) is True

    _store_all(storage, [_response(_date(5, hour=20), logprobs=_lp(("a", -0.1)))])
    assert storage.is_stalled(ENDPOINT) is False


def test_is_stalled_spans_month_boundary_and_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(config.api, "lt_stall_days", 4)
    storage = ResultsStorage(tmp_path)
    june = [_err_response(d, month=6) for d in (29, 30)]
    july = [_err_response(d, month=7) for d in (1, 2)]
    _store_all(storage, june + july)
    for r in july:
        storage.store_response(r.model_copy(update={"prompt": "other"}))

    assert storage.is_stalled(ENDPOINT) is True
