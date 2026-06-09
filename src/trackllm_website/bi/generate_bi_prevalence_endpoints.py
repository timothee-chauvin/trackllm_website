"""
Sample the n cheapest instances per model and per provider (union), and filter by max_avg_cost,
and store them to endpoints_bi_prevalence.yaml.
"""

import yaml

from trackllm_website.config import Endpoint, root

# Keep the N cheapest endpoints per model and per provider (union of both sets).
PREVALENCE_N = 2
# Drop any endpoint whose average cost exceeds this ($/million tokens).
PREVALENCE_MAX_AVG_COST = 0.5


def avg_cost(e: Endpoint) -> float:
    """Average cost per million tokens (input + output) / 2"""
    return (e.cost[0] + e.cost[1]) / 2


def get_cheap_endpoints(
    endpoints: list[Endpoint], n: int = 3, max_avg_cost: float = 1
) -> list[Endpoint]:
    """Get the n cheapest endpoints per model and per provider (union).

    Args:
        endpoints: List of all endpoints.
        n: Number of cheapest endpoints to keep per model/provider.
        max_avg_cost: Maximum average cost per million tokens to include.
    """
    # Group by model
    model_endpoints: dict[str, list[Endpoint]] = {}
    for e in endpoints:
        model_endpoints.setdefault(e.model, []).append(e)

    # Group by provider (without suffix)
    provider_endpoints: dict[str, list[Endpoint]] = {}
    for e in endpoints:
        provider = e.provider.split("/")[0]  # provider_without_suffix
        provider_endpoints.setdefault(provider, []).append(e)

    # Get n cheapest per model
    cheap_per_model: set[Endpoint] = set()
    for model, eps in model_endpoints.items():
        sorted_eps = sorted(eps, key=avg_cost)
        cheap_per_model.update(sorted_eps[:n])

    # Get n cheapest per provider
    cheap_per_provider: set[Endpoint] = set()
    for provider, eps in provider_endpoints.items():
        sorted_eps = sorted(eps, key=avg_cost)
        cheap_per_provider.update(sorted_eps[:n])

    # Union and filter by max cost
    cheap_endpoints = {
        e for e in (cheap_per_model | cheap_per_provider) if avg_cost(e) <= max_avg_cost
    }

    # Sort by model name, then provider, then cost for consistent output
    return sorted(cheap_endpoints, key=avg_cost)


def main():
    endpoints_bi_path = root / "endpoints_bi.yaml"
    with open(endpoints_bi_path) as f:
        data = yaml.safe_load(f)

    endpoints = [Endpoint(**e) for e in data["endpoints_bi"]]
    print(f"Total endpoints: {len(endpoints)}")

    cheap_endpoints = get_cheap_endpoints(
        endpoints, n=PREVALENCE_N, max_avg_cost=PREVALENCE_MAX_AVG_COST
    )
    print(
        f"Cheap endpoints (union of {PREVALENCE_N} per model + {PREVALENCE_N} per provider): {len(cheap_endpoints)}"
    )

    output_data = {
        "endpoints_bi_prevalence": [
            {
                "api": e.api,
                "model": e.model,
                "provider": e.provider,
                "cost": list(e.cost),
            }
            for e in cheap_endpoints
        ]
    }

    output_path = root / "endpoints_bi_prevalence.yaml"
    with open(output_path, "w") as f:
        yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

    print(f"Saved to {output_path}")

    models = set(e.model for e in cheap_endpoints)
    providers = set(e.provider.split("/")[0] for e in cheap_endpoints)
    print(f"Unique models: {len(models)}")
    print(f"Unique providers: {len(providers)}")

    total_avg_cost = sum(avg_cost(e) for e in cheap_endpoints)
    print(f"Sum of avg costs: {total_avg_cost:.2f}")


if __name__ == "__main__":
    main()
