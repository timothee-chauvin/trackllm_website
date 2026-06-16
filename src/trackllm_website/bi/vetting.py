"""Vet BI candidate endpoints by measured cost-per-request, and cache the rejects.

Buckets: candidate (usable, carries measured cost), liar (billed != advertised),
too_expensive (set by the catalog refresh against the selection ceiling),
bad_temperature (set by phase 1 when T=0 is ignored). Only liars are permanent;
the others are rechecked periodically since prices / providers change.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from trackllm_website.bi.common import QueryStrategy, strategy_to_query_args
from trackllm_website.config import Endpoint, logger

PRICE_TOLERANCE = 0.01
Bucket = Literal["candidate", "liar", "too_expensive", "bad_temperature", "transient"]


class VetResult(BaseModel):
    bucket: Bucket
    cost_per_request: float | None = None


def _advertised_cost(
    input_tokens: int, output_tokens: int, endpoint: Endpoint
) -> float:
    return (
        input_tokens * endpoint.cost[0] / 1e6 + output_tokens * endpoint.cost[1] / 1e6
    )


async def vet_endpoint(client, endpoint: Endpoint, strategy: QueryStrategy) -> VetResult:
    """Probe one endpoint with its resolved strategy; classify it.

    A transient error (network / 5xx) returns bucket="transient" so the caller
    does NOT cache it — it will be retried next run.
    """
    response = await client.query(
        endpoint,
        "a",
        temperature=0.0,
        logprobs=False,
        **strategy_to_query_args(strategy),
    )
    if response.error:
        logger.info(f"{endpoint} vet: transient error {response.error.message[:80]}")
        return VetResult(bucket="transient")

    advertised = _advertised_cost(
        response.input_tokens, response.output_tokens, endpoint
    )
    measured = response.cost
    if measured is None and response.generation_id:
        measured = await client.get_generation_cost(response.generation_id)
    if measured is None:
        return VetResult(bucket="transient")  # couldn't price it; retry later

    if advertised > 0 and measured > advertised * (1 + PRICE_TOLERANCE):
        logger.info(
            f"{endpoint} vet: liar (billed {measured:.8f} vs {advertised:.8f})"
        )
        return VetResult(bucket="liar")

    return VetResult(bucket="candidate", cost_per_request=measured)


class EndpointCache(BaseModel):
    liars: list[Endpoint]
    too_expensive: list[Endpoint]
    bad_temperature: list[Endpoint]

    def is_cached(self, endpoint: Endpoint) -> bool:
        return self.bucket_of(endpoint) is not None

    def bucket_of(self, endpoint: Endpoint) -> Bucket | None:
        if endpoint in self.liars:
            return "liar"
        if endpoint in self.too_expensive:
            return "too_expensive"
        if endpoint in self.bad_temperature:
            return "bad_temperature"
        return None

    def add_liar(self, endpoint: Endpoint) -> None:
        if endpoint not in self.liars:
            self.liars.append(endpoint)

    def add_too_expensive(self, endpoint: Endpoint) -> None:
        if endpoint not in self.too_expensive:
            self.too_expensive.append(endpoint)

    def add_bad_temperature(self, endpoint: Endpoint) -> None:
        if endpoint not in self.bad_temperature:
            self.bad_temperature.append(endpoint)

    def save(self, path: Path) -> None:
        def dump(es: list[Endpoint]) -> list[dict]:
            return [
                {
                    "api": e.api,
                    "model": e.model,
                    "provider": e.provider,
                    "cost": list(e.cost),
                }
                for e in sorted(es, key=lambda e: (e.model, e.provider or ""))
            ]

        data = {
            "liars": dump(self.liars),
            "too_expensive": dump(self.too_expensive),
            "bad_temperature": dump(self.bad_temperature),
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "EndpointCache":
        if not path.exists():
            return cls(liars=[], too_expensive=[], bad_temperature=[])
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        def parse(key: str) -> list[Endpoint]:
            return [
                Endpoint(
                    api=e["api"],
                    model=e["model"],
                    provider=e.get("provider"),
                    cost=tuple(e["cost"]),
                )
                for e in data.get(key, [])
            ]

        return cls(
            liars=parse("liars"),
            too_expensive=parse("too_expensive"),
            bad_temperature=parse("bad_temperature"),
        )
