import pytest

from trackllm_website.generate_site.rates import (
    MIN_ENDPOINT_YEARS,
    drift_rate,
    poisson_interval,
)


def test_zero_changes_uses_rule_of_three():
    # no events over T endpoint-years bounds the rate at 3/T, it does not prove 0
    assert poisson_interval(0, 3.0) == (0.0, 1.0)


def test_interval_is_symmetric_around_the_point_estimate():
    lo, hi = poisson_interval(4, 2.0)
    assert lo == pytest.approx((4 - 1.96 * 2) / 2.0)
    assert hi == pytest.approx((4 + 1.96 * 2) / 2.0)


def test_lower_bound_clamped_at_zero():
    lo, hi = poisson_interval(1, 1.0)
    assert lo == 0.0
    assert hi == pytest.approx(1 + 1.96)


def test_no_interval_without_exposure():
    assert poisson_interval(0, 0.0) is None
    assert poisson_interval(3, 0.0) is None


def test_rate_withheld_below_threshold():
    assert drift_rate(1, MIN_ENDPOINT_YEARS - 0.01) is None
    assert drift_rate(0, 0.0) is None


def test_rate_published_at_exactly_the_threshold():
    assert drift_rate(1, MIN_ENDPOINT_YEARS) == pytest.approx(2.0)


def test_zero_changes_over_enough_exposure_is_a_rate_of_zero():
    """The Overview's "Nothing detected yet" board is exactly this case: past the
    gate, no events is a measured 0.0, not a withheld rate."""
    assert drift_rate(0, MIN_ENDPOINT_YEARS) == 0.0
    assert drift_rate(0, 12.0) == 0.0


def test_rate_is_changes_per_endpoint_year():
    assert drift_rate(49, 15.04) == pytest.approx(49 / 15.04)
