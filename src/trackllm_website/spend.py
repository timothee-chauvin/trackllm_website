"""Actual-spend ledger: a context-scoped accumulator and per-endpoint JSONL store.

The accumulator lives in a ContextVar so OpenRouterClient.query can add each
response's cost without threading a parameter through every call path, and so a
cancelled (timed-out) coroutine still leaves its partial spend in the caller-owned
bucket. The ledger records only facts (money spent), one line per logical activity.
"""

import contextvars
import logging
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import orjson
from pydantic import BaseModel


@dataclass
class Spend:
    cost: float = 0.0
    n_queries: int = 0
    n_errors: int = 0

    def merge(self, other: "Spend") -> None:
        self.cost += other.cost
        self.n_queries += other.n_queries
        self.n_errors += other.n_errors


logger = logging.getLogger("trackllm-website")

_active: contextvars.ContextVar[Spend | None] = contextvars.ContextVar(
    "active_spend", default=None
)


def record_query(cost: float, is_error: bool) -> None:
    """Add one query's outcome to the active bucket, if any (else no-op).

    `cost` is always added: it is the actually-billed amount, which is 0.0 for
    true (network/HTTP) errors but non-zero for billed-but-errored responses
    such as "No logprobs returned" (tokens generated and charged). `n_errors`
    counts error responses separately.
    """
    bucket = _active.get()
    if bucket is None:
        return
    bucket.n_queries += 1
    bucket.cost += cost
    if is_error:
        bucket.n_errors += 1


@contextmanager
def track() -> Iterator[Spend]:
    """Open a fresh Spend bucket as the active accumulator for the with-body.

    Child asyncio tasks created within inherit it (ContextVar copy-on-task).
    The yielded Spend stays readable after the block, including after a caught
    cancellation/timeout inside it.
    """
    bucket = Spend()
    token = _active.set(bucket)
    try:
        yield bucket
    finally:
        _active.reset(token)


class SpendEntry(BaseModel):
    timestamp: datetime
    kind: str
    cost: float
    n_queries: int
    n_errors: int


def append_entry(
    spend_dir: Path, slug: str, kind: str, spend: Spend, now: datetime
) -> None:
    entry = SpendEntry(
        timestamp=now,
        kind=kind,
        cost=spend.cost,
        n_queries=spend.n_queries,
        n_errors=spend.n_errors,
    )
    path = spend_dir / slug / f"{now:%Y-%m}.jsonl"
    # The ledger is secondary bookkeeping; a write failure must never abort the
    # primary monitoring/onboarding run (which has already persisted its real
    # work). Log loudly and continue rather than propagating.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as f:
            f.write(orjson.dumps(entry.model_dump(mode="json")) + b"\n")
    except Exception:
        logger.exception(
            f"spend ledger write failed for {slug} {now:%Y-%m} (kind={kind}); continuing"
        )


def iter_ledger(spend_dir: Path) -> Iterator[tuple[str, dict]]:
    """Yield (slug, record) for every line of every per-endpoint ledger file."""
    if not spend_dir.exists():
        return
    for f in sorted(spend_dir.glob("*/*.jsonl")):
        slug = f.parent.name
        for line in f.read_bytes().splitlines():
            if not line.strip():
                continue
            yield slug, orjson.loads(line)


def today_by_kind(spend_dir: Path, day: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for _, rec in iter_ledger(spend_dir):
        if str(rec["timestamp"]).startswith(day):
            totals[rec["kind"]] += rec["cost"]
    return dict(totals)


def cumulative_by_kind(spend_dir: Path) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for _, rec in iter_ledger(spend_dir):
        totals[rec["kind"]] += rec["cost"]
    return dict(totals)
