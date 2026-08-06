"""Tests for util.format_cost, the single formatter behind every price on the site.

Its TypeScript twin (website/src/components.ts:fmtCost) is tested by
website/test/format_cost.test.ts against the same cases.
"""

import pytest

from trackllm_website.util import format_cost


@pytest.mark.parametrize(
    "value,expected",
    [
        (12.3456, "12.35"),
        (0.0, "0.00"),
        (1.0, "1.00"),
        (1234.5, "1234.50"),
        (0.1, "0.10"),
        (0.567, "0.57"),
        # under $0.10, two decimals would leave fewer than two significant
        # digits, so the precision grows instead
        (0.0567, "0.057"),
        (0.012, "0.012"),
        (0.01, "0.010"),
        (0.0049, "0.0049"),
        (0.001, "0.0010"),
        (0.0000123, "0.000012"),
        (1e-9, "0.0000000010"),
        (-0.0000123, "-0.000012"),
        (-12.3456, "-12.35"),
    ],
)
def test_format_cost(value, expected):
    assert format_cost(value) == expected


def _sig_digits(s: str) -> int:
    return len(s.lstrip("-").replace(".", "").lstrip("0"))


@pytest.mark.parametrize(
    "value", [1e-12, 3e-7, 0.00449, 0.0051, 0.0567, 0.099, 0.1, 7.0, 1e6]
)
def test_nonzero_costs_keep_at_least_two_significant_digits(value):
    formatted = format_cost(value)
    assert float(formatted) != 0
    assert _sig_digits(formatted) >= 2
