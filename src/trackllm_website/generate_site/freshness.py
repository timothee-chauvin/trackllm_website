"""When each monitoring method last collected data.

The front page shows this as a relative age ("14m ago"), which the browser
computes at load time from the absolute instant emitted here -- a page built six
hours ago must not claim to be fresh. Both methods store UTC, but not in the same
spelling (LT writes "...Z", B3IT "...+00:00"), so everything goes through as_utc
and comes back out as "...Z".
"""

from collections.abc import Iterable
from datetime import datetime, timezone

from trackllm_website.bi.phase_2 import Results


def as_utc(instant: str | datetime) -> datetime:
    """A stored timestamp as a UTC-aware datetime. Raises on anything unparseable."""
    dt = datetime.fromisoformat(instant) if isinstance(instant, str) else instant
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def latest(instants: Iterable[str | datetime | None]) -> str | None:
    """The most recent of a set of stored timestamps, as an ISO "...Z" string."""
    parsed = [as_utc(i) for i in instants if i is not None]
    if not parsed:
        return None
    return max(parsed).isoformat().replace("+00:00", "Z")


def last_phase2_query(results: Results) -> str | None:
    """When an endpoint was last queried by B3IT, from the phase-2 samples.

    A batch is keyed by the time its run started; the samples inside carry the
    query times, and a long batch trails its own key by up to an hour. Only the
    newest batch is scanned: batches run to completion in order, so no earlier
    one can hold a later sample.
    """
    batch_keys = [ts for batches in results.values() for ts in batches]
    if not batch_keys:
        return None
    last_batch = max(batch_keys, key=as_utc)
    samples = [
        ts for batches in results.values() for ts, _ in batches.get(last_batch, [])
    ]
    return latest([last_batch, *samples])
