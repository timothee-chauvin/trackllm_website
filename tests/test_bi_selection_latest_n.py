from datetime import datetime, timezone

from trackllm_website.bi.selection import (
    Rule,
    SelectionPolicy,
    select_monitoring_targets,
)
from trackllm_website.config import Endpoint


def ep(model, provider, cpr, created):
    return Endpoint(
        api="openrouter",
        model=model,
        provider=provider,
        cost=(1, 1),
        cost_per_request=cpr,
        created=datetime(2026, created, 1, tzinfo=timezone.utc),
    )


def test_latest_n_keeps_newest_per_pattern():
    pol = SelectionPolicy(
        budget_per_month=100,
        max_endpoint_cost=100,
        exclude=[],
        rules=[
            Rule(
                name="flagships",
                kind="models",
                patterns=["z-ai/glm-*"],
                providers_per_model=1,
                latest_n=2,
                flagship=True,
            )
        ],
    )
    cands = [
        ep("z-ai/glm-5.2", "p", 0.00001, 6),
        ep("z-ai/glm-5.1", "p", 0.00001, 4),
        ep("z-ai/glm-5", "p", 0.00001, 2),
        ep("z-ai/glm-4.7", "p", 0.00001, 1),
    ]
    sel, _ = select_monitoring_targets(cands, pol, [])
    assert sorted(e.model for e in sel) == ["z-ai/glm-5.1", "z-ai/glm-5.2"]


def test_latest_n_cheapest_provider_per_kept_model():
    pol = SelectionPolicy(
        budget_per_month=100,
        max_endpoint_cost=100,
        exclude=[],
        rules=[
            Rule(
                name="f",
                kind="models",
                patterns=["m/a"],
                providers_per_model=1,
                latest_n=1,
                flagship=True,
            )
        ],
    )
    cands = [ep("m/a", "cheap", 0.00001, 6), ep("m/a", "pricey", 0.0001, 6)]
    sel, _ = select_monitoring_targets(cands, pol, [])
    assert [e.provider for e in sel] == ["cheap"]
