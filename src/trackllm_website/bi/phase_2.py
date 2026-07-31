"""Phase 2 storage: the types and the on-disk layout of border-input samples.

Phase 2 is the repeated sampling of an endpoint's border inputs to detect model
changes. The sampling itself lives in `bi/sampling.py` and is driven by
`bi/monitor.py`; this module owns only what the samples are and where they go.
"""

from pathlib import Path
from typing import NewType

import orjson

from trackllm_website.config import Endpoint, config
from trackllm_website.util import atomic_write_bytes, slugify

Prompt = NewType("Prompt", str)
Timestamp = NewType("Timestamp", str)
ResponseToken = NewType("ResponseToken", str)

Results = dict[Prompt, dict[Timestamp, list[tuple[Timestamp, ResponseToken]]]]


def get_output_path(endpoint: Endpoint, year_month: str) -> Path:
    """Get the output JSON path for an endpoint."""
    endpoint_dir = config.bi.phase_2_dir / slugify(
        f"{endpoint.model}#{endpoint.provider}"
    )
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    return endpoint_dir / f"{year_month}.json"


def load_existing_results(path: Path) -> Results:
    """Load existing results from JSON file, restoring sample tuples.

    JSON round-trips tuples into lists; without the conversion, re-saving
    loaded results violates the `Results` type hint.
    """
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        raw = orjson.loads(f.read())
    return {
        prompt: {ts: [tuple(s) for s in samples] for ts, samples in batches.items()}
        for prompt, batches in raw.items()
    }


def save_results(
    path: Path,
    results: Results,
) -> None:
    """Save results to JSON file."""
    atomic_write_bytes(path, orjson.dumps(results))
