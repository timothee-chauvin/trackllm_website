import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from trackllm_website.config import Endpoint
from trackllm_website.storage import Response, ResponseLogprobs, ResultsStorage

EP = Endpoint(api="openrouter", model="org/model", provider="prov", cost=(1, 1))


def _seed_endpoint(data_dir, n=50):
    """One endpoint/prompt with enough logprob responses to produce LT scores."""
    storage = ResultsStorage(data_dir)
    base = datetime(2026, 6, 1, tzinfo=timezone.utc)
    for i in range(n):
        lp = ResponseLogprobs(
            tokens=["a", "b", "c"],
            logprobs=[np.float32(-0.1 - 0.01 * i), np.float32(-1.0), np.float32(-2.0)],
        )
        storage.store_response(
            Response(
                date=base + timedelta(hours=i),
                endpoint=EP,
                prompt="Hi",
                logprobs=lp,
                cost=0,
            )
        )


def test_lt_cli_computes_without_main_module_class_mismatch(tmp_path):
    """Run the LT scoring entry point exactly as the workflow does.

    Regression for the `python -m` bug: when lt_scores was executed as __main__,
    its LTScores became __main__.LTScores, which beartype rejected in
    lt_events.update_endpoint_events (`not instance of trackllm_website.lt_scores
    .LTScores`), crashing run-main every hour. The CLI now lives in its own module
    so lt_scores is only ever imported under its canonical name.
    """
    _seed_endpoint(tmp_path)
    env = {**os.environ, "DATA_DIR": str(tmp_path), "OPENROUTER_API_KEY": "dummy"}
    result = subprocess.run(
        [sys.executable, "-m", "trackllm_website.lt_scores_cli", "all"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    # Reaching save_events (writes lt_changes.json) means the update_endpoint_events
    # call that beartype used to reject went through.
    assert (tmp_path / "lt_changes.json").exists()
