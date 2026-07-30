from datetime import datetime, timezone

import yaml

from trackllm_website.config import Endpoint
from trackllm_website.update_endpoints import save_endpoints_catalog


def ep(model, provider, cost, **kwargs):
    return Endpoint(api="openrouter", model=model, provider=provider, cost=cost, **kwargs)


def test_catalog_round_trip(tmp_path):
    path = tmp_path / "endpoints_catalog.yaml"
    created = datetime(2026, 1, 2, tzinfo=timezone.utc)
    endpoints = [
        ep(
            "org/b",
            "paid",
            (1.0, 2.0),
            created=created,
            supports_temperature=True,
            supports_logprobs=True,
        ),
        ep("org/a", "free", (0, 0), supports_temperature=False, supports_logprobs=False),
        ep("org/a", "bare", (0.5, 1.5)),  # upstream omitted supported_parameters
    ]
    save_endpoints_catalog(endpoints, path)
    entries = yaml.safe_load(path.read_text())["endpoints_catalog"]

    assert [(e["model"], e["provider"]) for e in entries] == [
        ("org/a", "bare"),
        ("org/a", "free"),
        ("org/b", "paid"),
    ]
    bare, free, paid = entries
    assert bare == {
        "model": "org/a",
        "provider": "bare",
        "cost": [0.5, 1.5],
        "created": None,
        "supports_temperature": None,
        "supports_logprobs": None,
        "free": False,
    }
    assert free["free"] is True
    assert free["cost"] == [0, 0]
    assert free["supports_logprobs"] is False
    assert paid["free"] is False
    assert paid["created"] == "2026-01-02T00:00:00+00:00"
    assert paid["supports_temperature"] is True
    assert paid["supports_logprobs"] is True
