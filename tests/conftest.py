import json
from datetime import datetime, timezone
from pathlib import Path

from trackllm_website.bi.phase_2 import save_results
from trackllm_website.bi.state import EndpointBIState, Epoch
from trackllm_website.config import Endpoint
from trackllm_website.util import slugify


def b3it_slug(model: str, provider: str) -> str:
    """The slug both b3it state files and phase-2 dirs are keyed by.

    EndpointBIState derives its own slug from the endpoint, so a caller-supplied
    one that disagreed would silently split the state file from its phase-2 data
    and yield a view with an empty tv_series.
    """
    return slugify(f"{model}#{provider}")


def write_lt_endpoint(
    root: Path, slug: str, model: str, provider: str, *, dates, changes, drift
):
    d = root / "data" / "lt" / slug
    prompt_dir = d / "default"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "info.json").write_text(
        json.dumps({"prompt": "hi", "endpoint": {"model": model, "provider": provider}})
    )
    month = dates[-1][:7]
    day = dates[-1][8:10]
    month_dir = prompt_dir / month
    month_dir.mkdir()
    (month_dir / "queries.json").write_text(json.dumps([[f"{day} 00:00:00", 0]]))
    (d / "lt_scores.json").write_text(
        json.dumps(
            {
                "n_per_test": 24,
                "dates": dates,
                "scores": [0.5] * len(dates),
                "sigmas": [None] * len(dates),
                "changes": changes,
                "drift_dates": dates,
                "drift": drift,
            }
        )
    )


def write_b3it_state(root: Path, model: str, provider: str, *, status):
    """A b3it endpoint with no phase-2 data, so its view has an empty tv_series."""
    state = {
        "endpoint": {
            "api": "openrouter",
            "model": model,
            "provider": provider,
            "cost": [0.1, 0.2],
            "max_logprobs": None,
        },
        "status": status,
        "retired": None,
        "epochs": [
            {
                "start": "2026-01-01T00:00:00Z",
                "border_inputs": [],
                "reference": {},
                "end": None,
            }
        ],
    }
    sd = root / "data" / "b3it" / "state"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / f"{b3it_slug(model, provider)}.json").write_text(json.dumps(state))


def write_b3it_series(
    root: Path, model: str, provider: str, *, status, retired, month, tokens
):
    """A b3it endpoint with one epoch and a daily phase-2 series of `tokens`.

    Unlike write_b3it_state this produces a real TV series: one sampled token per
    day of `month`, compared against an all-"A" reference. A constant "A" series
    is a stable endpoint; switching token part-way yields a detected transition.
    """
    ep = Endpoint(api="openrouter", model=model, provider=provider, cost=(0.1, 0.2))
    reference = {"p1": [(f"{month}-01T00:00:00Z", "A")] * 10}
    results = {"p1": {}}
    for day, token in enumerate(tokens, start=1):
        ts = f"{month}-{day:02d}T00:00:00+00:00"
        results["p1"][ts] = [(ts, token)] * 10
    state = EndpointBIState(
        endpoint=ep,
        status=status,
        retired=retired,
        epochs=[
            Epoch(
                start=datetime.fromisoformat(f"{month}-01").replace(
                    tzinfo=timezone.utc
                ),
                border_inputs=["p1"],
                reference=reference,
            )
        ],
    )
    state.save(root / "data" / "b3it" / "state")
    p2_dir = root / "data" / "b3it" / "phase_2" / b3it_slug(model, provider)
    p2_dir.mkdir(parents=True)
    save_results(p2_dir / "p1.json", results)
