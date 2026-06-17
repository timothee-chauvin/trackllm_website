from trackllm_website.bi.selection import load_policy


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
