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
        (0.005, "0.01"),
        (0.01, "0.01"),
        (0.019, "0.02"),
        # below the 2-decimal floor: 2 significant digits instead of "0.00"
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


@pytest.mark.parametrize("value", [1e-12, 3e-7, 0.00449, 0.004999, 0.0051])
def test_nonzero_costs_never_render_as_zero(value):
    assert float(format_cost(value)) != 0
