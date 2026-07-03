import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import orjson

from trackllm_website.config import Endpoint
from trackllm_website.lt_scores import (
    N_PER_TEST,
    SIGMA_INF_THRESHOLD,
    ChangePoint,
    LTScores,
    compute_endpoint_scores,
    detect_changes,
    normalize_sigma,
)
from trackllm_website.storage import Response, ResponseLogprobs, ResultsStorage

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


def test_normalize_sigma():
    assert normalize_sigma(12.3) == 12.3
    assert normalize_sigma(0.0) == 0.0
    assert normalize_sigma(2.0e38) is None
    assert normalize_sigma(-2.0e38) is None
    assert normalize_sigma(SIGMA_INF_THRESHOLD) is None
    assert normalize_sigma(float("inf")) is None
    assert normalize_sigma(float("nan")) is None


def test_near_zero_variance_jump_normalizes_huge_finite_sigma_to_none():
    """A tiny-but-nonzero baseline std yields a finite astronomically-large
    deviation (~5e38 here, matching real data); it must be represented as None
    so every consumer displays ∞ instead of a 39-digit number."""
    scores = _zero_variance_jump()
    scores += np.random.default_rng(0).normal(0, 1e-38, len(scores))
    changes, _ = detect_changes(scores)
    detected = next(cp for cp in changes if abs(cp.index - 175) <= 1)
    assert detected.sigma is None


def test_empty_logprob_response_is_skipped(tmp_path):
    """A provider returning an empty completion stores a logprob response with no
    tokens. Such an observation carries no signal and previously crashed scoring
    with `ValueError: min() iterable argument is empty` in build_tensor, which
    took down every hourly run-main. It must be dropped, not crash."""
    ep = Endpoint(api="openrouter", model="org/model", provider="prov", cost=(1, 1))
    storage = ResultsStorage(Path(tmp_path) / "lt")
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    n = 2 * N_PER_TEST + 1
    for i in range(n):
        # One real provider returned tokens=[]/logprobs=[] (empty completion).
        lp = (
            ResponseLogprobs(tokens=[], logprobs=[])
            if i == n // 2
            else ResponseLogprobs(
                tokens=["a", "b"],
                logprobs=[np.float32(-0.1 - 0.01 * i), np.float32(-1.0)],
            )
        )
        storage.store_response(
            Response(
                date=base + timedelta(hours=i),
                endpoint=ep,
                prompt="Hi",
                logprobs=lp,
                cost=0,
            )
        )

    endpoint_dir = Path(tmp_path) / "lt" / "org2fmodel23prov"
    result = compute_endpoint_scores(endpoint_dir)
    assert result is not None
    assert len(result.dates) == len(result.scores)
