import pytest
from pydantic import ValidationError

from trackllm_website.bi.selection import Rule, load_policy


def test_loads_policy(tmp_path):
    p = tmp_path / "sel.toml"
    p.write_text("""
budget_per_month = 10.0
max_endpoint_cost = 0.5
exclude = ["*image*"]
[[rule]]
name = "flagships"
kind = "models"
patterns = ["anthropic/claude-fable-5"]
providers_per_model = 1
flagship = true
[[rule]]
name = "long-tail"
kind = "models"
patterns = ["*"]
providers_per_model = 1
max_monthly_cost = 0.1
""")
    policy = load_policy(p)
    assert policy.budget_per_month == 10.0
    assert policy.max_endpoint_cost == 0.5
    assert policy.rules[0].name == "flagships"
    assert policy.rules[0].flagship is True
    assert policy.rules[0].providers_per_model == 1
    assert policy.rules[1].max_monthly_cost == 0.1
    # flagship model patterns surface for the vetting exemption
    assert "anthropic/claude-fable-5" in policy.flagship_patterns()


@pytest.mark.parametrize("kind", ["models", "popular"])
def test_providers_per_model_required(kind):
    # omitting it used to silently mean "all providers" (eps[:None])
    with pytest.raises(ValidationError, match="providers_per_model"):
        Rule(name="r", kind=kind, patterns=["*"])


def test_endpoints_per_provider_required():
    with pytest.raises(ValidationError, match="endpoints_per_provider"):
        Rule(name="r", kind="providers", patterns=["*"])


def test_providers_rule_does_not_need_providers_per_model():
    Rule(name="r", kind="providers", patterns=["*"], endpoints_per_provider=1)


def test_repo_policy_loads():
    # the nightly lifecycle workflow loads this file; a policy that fails
    # validation must never reach main
    from trackllm_website.config import config, root

    load_policy(root / config.bi.selection_path)
