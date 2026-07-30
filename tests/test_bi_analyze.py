"""Determinism of the pure analysis functions feeding the site build."""

import os
import subprocess
import sys

_SCRIPT = """
from collections import Counter
from trackllm_website.bi.analyze import compute_tv_distance

p = Counter({f"tok{i}": 1 for i in range(60)})
q = Counter({f"tok{i}": i + 1 for i in range(60)})
print(repr(compute_tv_distance(p, q)))
"""


def test_tv_distance_independent_of_hash_seed():
    """Summing over set-ordered tokens leaked PYTHONHASHSEED into the last
    digits of every TV value, so no two site builds were byte-identical."""
    outputs = {
        subprocess.run(
            [sys.executable, "-c", _SCRIPT],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("0", "1", "2", "3")
    }
    assert len(outputs) == 1
