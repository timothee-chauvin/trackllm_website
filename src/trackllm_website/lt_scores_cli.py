"""CLI entry point for LT score computation.

Kept separate from lt_scores so that module is never executed as __main__: running
`python -m trackllm_website.lt_scores` would load it under the name __main__, making
__main__.LTScores a different class from trackllm_website.lt_scores.LTScores and
tripping beartype's isinstance check in lt_events.update_endpoint_events.
"""

import fire

from trackllm_website.lt_scores import compute_all, compute_latest

if __name__ == "__main__":
    fire.Fire({"all": compute_all, "latest": compute_latest})
