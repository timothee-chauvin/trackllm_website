"""Integration test: latest_n + popular + exclude together, mirroring bi_selection.toml."""

from datetime import datetime, timezone

from trackllm_website.bi.selection import (
    Rule,
    SelectionPolicy,
    select_monitoring_targets,
)
from trackllm_website.config import Endpoint


def ep(model, provider, cpr, month):
    return Endpoint(
        api="openrouter",
        model=model,
        provider=provider,
        cost=(1, 1),
        cost_per_request=cpr,
        created=datetime(2026, month, 1, tzinfo=timezone.utc),
    )


def _policy():
    """A real SelectionPolicy mirroring bi_selection.toml's rule shapes."""
    return SelectionPolicy(
        budget_per_month=100.0,
        max_endpoint_cost=0.50,
        exclude=["*-fast", "*search*", "openrouter/*"],
        rules=[
            Rule(
                name="flagships",
                kind="models",
                patterns=["anthropic/claude-opus-*", "z-ai/glm-*"],
                providers_per_model=1,
                latest_n=2,
                flagship=True,
            ),
            Rule(
                name="popular",
                kind="popular",
                patterns=[],
                providers_per_model=1,
                max_monthly_cost=0.25,
            ),
            Rule(
                name="long-tail",
                kind="models",
                patterns=["*"],
                providers_per_model=1,
                max_monthly_cost=0.10,
            ),
        ],
    )


def test_latest_n_popular_exclude_integration():
    policy = _policy()
    candidates = [
        # glm family: 3 versions; only newest 2 (5.2, 5.1) should be kept by flagships.
        ep("z-ai/glm-5.2", "novita", 0.000001, 6),
        ep("z-ai/glm-5.1", "novita", 0.000001, 4),
        ep("z-ai/glm-5", "novita", 0.000001, 2),
        # expensive newest flagship: over the $0.50 ceiling but flagship => still selected.
        ep("anthropic/claude-opus-5", "anthropic", 0.001, 6),  # $6/mo
        # a popular non-flagship model, under the popular rule's $0.25/mo ceiling.
        ep("mistralai/mistral-small", "mistral", 0.00002, 3),  # $0.12/mo
        # excluded by globs (must not be selected by any rule):
        ep("openai/gpt-4o-fast", "openai", 0.000001, 5),
        ep("openai/gpt-4o-search-preview", "openai", 0.000001, 5),
        ep("openrouter/owl-alpha", "stealth", 0.000001, 5),
        # -preview is NOT excluded: a cheap one can be picked by long-tail.
        ep("google/gemini-3-preview", "google", 0.000001, 5),
    ]
    popular_models = ["mistralai/mistral-small"]

    selected, labels = select_monitoring_targets(candidates, policy, popular_models)
    sel_models = {e.model for e in selected}

    # newest-2 per family: older 3rd glm version is NOT selected by flagships.
    assert "z-ai/glm-5.2" in sel_models
    assert "z-ai/glm-5.1" in sel_models
    glm5 = next(e for e in candidates if e.model == "z-ai/glm-5")
    assert labels.get(glm5) != "flagships"

    # popular non-flagship model present in candidates is selected and labelled "popular".
    mistral = next(e for e in candidates if e.model == "mistralai/mistral-small")
    assert mistral in selected
    assert labels[mistral] == "popular"

    # *-fast, *search*, openrouter/* are excluded everywhere.
    assert "openai/gpt-4o-fast" not in sel_models
    assert "openai/gpt-4o-search-preview" not in sel_models
    assert "openrouter/owl-alpha" not in sel_models

    # -preview is not excluded (long-tail can select it).
    assert "google/gemini-3-preview" in sel_models

    # flagship over the $0.50 ceiling is still selected (engine doesn't enforce ceiling).
    opus = next(e for e in candidates if e.model == "anthropic/claude-opus-5")
    assert opus in selected
    assert labels[opus] == "flagships"
