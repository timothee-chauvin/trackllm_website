import math
from datetime import datetime, timedelta, timezone

import numpy as np
import orjson

from trackllm_website.lt_scores import (
    N_PER_TEST,
    ChangePoint,
    LTScores,
    detect_changes,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _zero_variance_jump(length: int = 200, spike_index: int = 175) -> np.ndarray:
    """A flat (zero-variance) baseline with a single jump.

    The running-std window leading up to `spike_index` is perfectly constant, so
    the normalized deviation there is mathematically infinite (divide-by-zero).
    """
    scores = np.zeros(length)
    scores[spike_index] = 5.0
    return scores


def test_zero_variance_jump_produces_no_infinite_sigma():
    """A divide-by-zero deviation must never leak into a ChangePoint as inf/NaN.

    orjson serializes inf/NaN as JSON `null`, which previously broke reload.
    """
    changes, sigmas = detect_changes(_zero_variance_jump())
    assert changes, "expected the jump to be detected as a change"
    for cp in changes:
        assert cp.sigma is None or math.isfinite(cp.sigma)
    detected = next(cp for cp in changes if cp.index == 175)
    assert detected.sigma is None  # undefined significance, represented explicitly


def test_lt_scores_survive_json_roundtrip_with_nonfinite_sigma():
    """The exact failure that stopped hourly commits: a value that serializes to
    JSON `null` must still validate back into the model."""
    scores = _zero_variance_jump()
    changes, sigmas = detect_changes(scores)
    dates = [START + timedelta(hours=i) for i in range(len(scores))]
    result = LTScores(
        n_per_test=N_PER_TEST,
        dates=dates,
        scores=scores.tolist(),
        sigmas=[None if not np.isfinite(v) else v for v in sigmas.tolist()],
        changes=changes,
    )
    blob = orjson.dumps(result.model_dump(mode="json"))
    # Round-trips through `null` without raising a ValidationError.
    reloaded = LTScores.model_validate(orjson.loads(blob))
    assert [cp.sigma for cp in reloaded.changes] == [cp.sigma for cp in changes]


def test_change_point_accepts_none_sigma():
    assert ChangePoint(index=10, sigma=None).sigma is None
