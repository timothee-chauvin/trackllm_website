import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import orjson

import trackllm_website.lt_scores as lt_scores
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


_REPRODUCIBILITY_PROBE = """
import hashlib
from datetime import datetime, timedelta, timezone
from trackllm_website.lt_drift import compute_drift_series
from trackllm_website.lt_scores import build_tensor

toks = [f"tok{i}" for i in range(40)]
dicts = [{t: -0.5 - (i * 0.017 + j * 0.003) for j, t in enumerate(toks)} for i in range(60)]
start = datetime(2026, 1, 1, tzinfo=timezone.utc)
obs = [(start + timedelta(hours=i), d) for i, d in enumerate(dicts)]
payload = repr(build_tensor(dicts).tolist()) + repr(compute_drift_series(obs, None))
print(hashlib.sha256(payload.encode()).hexdigest())
"""


def test_scoring_is_reproducible_across_hash_seeds():
    """Token sets were iterated in hash order, so summation order -- and the last
    ULP of every score and drift value -- changed between processes. That made a
    recompute rewrite ~300 lt_scores.json files with pure noise, burying real
    changes in the diff.
    """
    digests = set()
    for seed in ("0", "1", "2"):
        proc = subprocess.run(
            [sys.executable, "-c", _REPRODUCIBILITY_PROBE],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert proc.returncode == 0, proc.stderr
        digests.add(proc.stdout.strip())
    assert len(digests) == 1, f"output varies with hash seed: {digests}"


def _bimodal_change(length: int = 400, first: int = 250) -> np.ndarray:
    """One change whose exceedance hump has two crests, N_PER_TEST indices apart.

    The two-sample statistic at index i compares [i-N, i) against [i, i+N), so a
    single step change keeps it elevated across ~2*N_PER_TEST indices. Noise
    inside that span can leave two local maxima -- real data does this on 14
    endpoint-pairs, always exactly one day apart.
    """
    scores = 0.01 * (np.arange(length) % 3)  # tiny baseline variation so std > 0
    second = first + N_PER_TEST
    for k in range(13):
        scores[first + k] = 5.0 - 0.15 * k
        scores[second - k] = 5.0 - 0.15 * k
    for k in range(1, 13):
        scores[second + k] = max(5.0 - 0.4 * k, 0.03)
    return scores


def test_one_change_is_not_reported_twice_within_its_influence_window():
    """Regression: PEAK_DISTANCE was N_PER_TEST, half a change's influence width,
    so both crests of a single change's hump survived as separate changes."""
    changes, _ = detect_changes(_bimodal_change())
    assert len(changes) == 1, f"one change reported at {[c.index for c in changes]}"


def test_peak_distance_matches_the_statistic_influence_width():
    """Peak separation and the running-baseline exclusion zone describe the same
    quantity -- how far a single change reaches into the statistic. They must not
    drift apart: a smaller PEAK_DISTANCE double-counts changes."""
    assert lt_scores.PEAK_DISTANCE == lt_scores.STAT_EXCLUSION_ZONE == 2 * N_PER_TEST


def test_genuinely_separate_changes_are_still_reported_separately():
    """The wider peak distance must not merge changes that are truly distinct."""
    scores = _bimodal_change(length=700)
    scores[500:513] = [5.0 - 0.15 * k for k in range(13)]
    changes, _ = detect_changes(scores)
    assert len(changes) == 2
    assert changes[1].index - changes[0].index > 2 * N_PER_TEST


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


def test_compute_endpoint_scores_populates_drift(tmp_path, monkeypatch):
    ep = tmp_path / "endpoint"
    prompt = ep / "prompt1"
    prompt.mkdir(parents=True)
    (prompt / "info.json").write_text("{}")
    base = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    data = [
        (
            base + timedelta(days=day, hours=k),
            {"A": -0.02, "B": -4.0} if day < 15 else {"A": -4.0, "B": -0.02},
        )
        for day in range(30)
        for k in range(4)
    ]
    monkeypatch.setattr(lt_scores, "load_prompt_logprobs", lambda _dir: data)
    s = compute_endpoint_scores(ep)
    assert s is not None and len(s.drift) == len(s.drift_dates) > 0
    assert s.drift[0] < 0.3 and max(s.drift) > 1.0

