from trackllm_website.bi.costs import build_cost_summary
from trackllm_website.bi.selection import Rule, SelectionPolicy
from trackllm_website.config import Endpoint


def ep(m, p, cpr):
    return Endpoint(api="openrouter", model=m, provider=p, cost=(1, 1), cost_per_request=cpr)


def test_summary_run_rate_and_top():
    policy = SelectionPolicy(
        budget_per_month=10,
        max_endpoint_cost=10,
        exclude=[],
        rules=[
            Rule(
                name="long-tail",
                kind="models",
                patterns=["*"],
                providers_per_model=1,
                max_monthly_cost=10,
            )
        ],
    )
    cands = [ep("m/a", "p", 0.00001), ep("m/b", "p", 0.00005)]
    summary = build_cost_summary(cands, policy)
    assert summary["budget_per_month"] == 10
    assert abs(summary["run_rate_per_month"] - (0.00001 + 0.00005) * 6000) < 1e-6
    # top endpoints sorted by monthly cost descending
    assert summary["endpoints"][0]["model"] == "m/b"
    assert summary["by_rule"]["long-tail"]["count"] == 2
