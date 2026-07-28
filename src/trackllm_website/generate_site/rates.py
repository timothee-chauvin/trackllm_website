"""Normalised drift rates and their uncertainty. Pure: no I/O, no build state."""

import math

# Below this much monitoring a rate is not a measurement: one change in three
# weeks computes to ~17/year. Every surface withholds the rate instead.
MIN_ENDPOINT_YEARS = 0.5

_Z = 1.96


def poisson_interval(k: int, exposure: float) -> tuple[float, float] | None:
    """95% interval for k changes observed over `exposure` endpoint-years.

    k == 0 uses the rule of three: zero events is evidence of a rate below
    3/exposure, not evidence of a rate of zero.
    """
    if exposure <= 0:
        return None
    if k == 0:
        return (0.0, 3.0 / exposure)
    half = _Z * math.sqrt(k)
    return (max(0.0, (k - half) / exposure), (k + half) / exposure)


def drift_rate(k: int, exposure: float) -> float | None:
    """Detected changes per endpoint-year, or None when exposure is too thin."""
    if exposure < MIN_ENDPOINT_YEARS:
        return None
    return k / exposure
