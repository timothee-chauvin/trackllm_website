"""Vet BI candidate endpoints by measured cost-per-request, and cache the rejects.

Buckets: candidate (usable, carries measured cost), liar (billed != advertised),
too_expensive (set by the catalog refresh against the selection ceiling),
bad_temperature (set by phase 1 when T=0 is ignored). Only liars are permanent;
the others are rechecked periodically since prices / providers change.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from trackllm_website.bi.common import QueryStrategy, strategy_to_query_args
from trackllm_website.config import Endpoint, logger
from trackllm_website.util import atomic_write_bytes

PRICE_TOLERANCE = 0.01
Bucket = Literal[
    "candidate", "liar", "too_expensive", "bad_temperature", "transient", "unprobeable"
]

# batch / no_text are structural (derived from the catalog without probing);
# flaky means the vetting probe itself failed `threshold` runs in a row.
UnprobeableReason = Literal["batch", "no_text", "flaky"]


class UnprobeableEntry(BaseModel):
    endpoint: Endpoint
    reason: UnprobeableReason
    detail: str | None = None


class FailureStreak(BaseModel):
    count: int
    last_error: str


class VetResult(BaseModel):
    bucket: Bucket
    cost_per_request: float | None = None
    detail: str | None = None  # transient only: the error, for the failure streak


async def vet_endpoint(
    client, endpoint: Endpoint, strategy: QueryStrategy
) -> VetResult:
    """Probe one endpoint with its resolved strategy; classify it.

    expected = response.cost (token math at advertised price, incl. reasoning).
    actual = OpenRouter's real charge for the generation. A liar bills more than
    the token math implies. A transient error (network / 5xx) or an un-priceable
    response returns bucket="transient" so the caller does NOT cache it.
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
        return VetResult(bucket="transient", detail=response.error.message)
    expected = response.cost  # compute_cost(usage): token math at advertised price
    if not response.generation_id:
        return VetResult(bucket="transient", detail="response has no generation id")
    # Blocks 5-75s by design (get_generation_cost backoff); callers fan out concurrently.
    actual = await client.get_generation_cost(
        response.generation_id, session=client.session
    )
    if actual is None:
        # couldn't price it; retry later
        return VetResult(bucket="transient", detail="could not fetch generation cost")
    if expected > 0 and actual > expected * (1 + PRICE_TOLERANCE):
        logger.info(
            f"{endpoint} vet: liar (billed {actual:.8f} vs expected {expected:.8f})"
        )
        return VetResult(bucket="liar")
    return VetResult(bucket="candidate", cost_per_request=actual)


class EndpointCache(BaseModel):
    liars: list[Endpoint]
    too_expensive: list[Endpoint]
    bad_temperature: list[Endpoint]
    # defaults keep pre-unprobeable cache files loadable
    unprobeable: list[UnprobeableEntry] = []
    failure_streaks: dict[str, FailureStreak] = {}  # str(endpoint) -> streak
    last_recheck: datetime | None = None

    def is_cached(self, endpoint: Endpoint) -> bool:
        return self.bucket_of(endpoint) is not None

    def bucket_of(self, endpoint: Endpoint) -> Bucket | None:
        if endpoint in self.liars:
            return "liar"
        if endpoint in self.too_expensive:
            return "too_expensive"
        if endpoint in self.bad_temperature:
            return "bad_temperature"
        if any(entry.endpoint == endpoint for entry in self.unprobeable):
            return "unprobeable"
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

    def add_unprobeable(
        self, endpoint: Endpoint, reason: UnprobeableReason, detail: str | None
    ) -> None:
        if not any(entry.endpoint == endpoint for entry in self.unprobeable):
            self.unprobeable.append(
                UnprobeableEntry(endpoint=endpoint, reason=reason, detail=detail)
            )

    def record_failure(self, endpoint: Endpoint, error: str, threshold: int) -> None:
        """One more failed vetting run; at `threshold` consecutive failures the
        endpoint moves to the unprobeable bucket and stops being probed."""
        key = str(endpoint)
        prior = self.failure_streaks.get(key)
        streak = FailureStreak(count=(prior.count if prior else 0) + 1, last_error=error)
        if streak.count >= threshold:
            self.failure_streaks.pop(key, None)
            self.add_unprobeable(endpoint, reason="flaky", detail=error)
            logger.info(f"{endpoint}: unprobeable after {streak.count} failures ({error[:80]})")
        else:
            self.failure_streaks[key] = streak

    def record_success(self, endpoint: Endpoint) -> None:
        self.failure_streaks.pop(str(endpoint), None)

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
            "last_recheck": self.last_recheck.isoformat()
            if self.last_recheck is not None
            else None,
            "liars": dump(self.liars),
            "too_expensive": dump(self.too_expensive),
            "bad_temperature": dump(self.bad_temperature),
            "unprobeable": [
                dump([entry.endpoint])[0]
                | {"reason": entry.reason, "detail": entry.detail}
                for entry in sorted(
                    self.unprobeable,
                    key=lambda x: (x.endpoint.model, x.endpoint.provider or ""),
                )
            ],
            "failure_streaks": {
                key: {"count": s.count, "last_error": s.last_error}
                for key, s in sorted(self.failure_streaks.items())
            },
        }
        atomic_write_bytes(
            path, yaml.dump(data, default_flow_style=False, sort_keys=False).encode()
        )

    @classmethod
    def load(cls, path: Path) -> "EndpointCache":
        if not path.exists():
            return cls(liars=[], too_expensive=[], bad_temperature=[])
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        def parse_one(e: dict) -> Endpoint:
            return Endpoint(
                api=e["api"],
                model=e["model"],
                provider=e.get("provider"),
                cost=tuple(e["cost"]),
            )

        def parse(key: str) -> list[Endpoint]:
            return [parse_one(e) for e in data.get(key, [])]

        raw_recheck = data.get("last_recheck")
        return cls(
            liars=parse("liars"),
            too_expensive=parse("too_expensive"),
            bad_temperature=parse("bad_temperature"),
            unprobeable=[
                UnprobeableEntry(
                    endpoint=parse_one(e), reason=e["reason"], detail=e["detail"]
                )
                for e in data.get("unprobeable", [])
            ],
            failure_streaks={
                key: FailureStreak(**s)
                for key, s in (data.get("failure_streaks") or {}).items()
            },
            last_recheck=datetime.fromisoformat(raw_recheck) if raw_recheck else None,
        )


def clear_recheckable(cache: EndpointCache) -> int:
    """Empty the buckets due for periodic re-vetting (everything but liars),
    plus the in-progress failure streaks; returns how many entries were cleared."""
    n = len(cache.too_expensive) + len(cache.bad_temperature) + len(cache.unprobeable)
    cache.too_expensive = []
    cache.bad_temperature = []
    cache.unprobeable = []
    cache.failure_streaks = {}
    return n


def should_recheck(cache: EndpointCache, now: datetime, recheck_days: int) -> bool:
    """Whether the too_expensive / bad_temperature buckets are due for re-vetting.

    Prices drop and providers fix temperature, so these rejects are cleared and
    re-probed periodically (liars stay permanent).
    """
    if cache.last_recheck is None:
        return True
    return now - cache.last_recheck >= timedelta(days=recheck_days)
