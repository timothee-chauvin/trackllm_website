import json
from pathlib import Path


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


def write_b3it_state(root: Path, slug: str, model: str, provider: str, *, status):
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
    (sd / f"{slug}.json").write_text(json.dumps(state))
