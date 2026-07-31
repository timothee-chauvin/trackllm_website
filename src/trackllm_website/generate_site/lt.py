import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from trackllm_website.lt_scores import SCORES_FILENAME
from trackllm_website.storage import parse_query_date

# Endpoints with last query older than this are considered inactive
INACTIVE_THRESHOLD_DAYS = 3


@dataclass
class PromptInfo:
    """Info about a prompt directory."""

    slug: str
    prompt: str
    months: list[str]


@dataclass
class EndpointInfo:
    """Info about an endpoint."""

    model: str
    provider: str
    slug: str
    prompts: list[PromptInfo]
    last_query_date: datetime | None = None

    @property
    def is_active(self) -> bool:
        if self.last_query_date is None:
            return False
        threshold = datetime.now(timezone.utc) - timedelta(days=INACTIVE_THRESHOLD_DAYS)
        return self.last_query_date > threshold

    @property
    def last_query_str(self) -> str:
        if self.last_query_date is None:
            return "Never"
        return self.last_query_date.strftime("%Y-%m-%d")


def get_prompt_last_query_date(prompt_dir: Path) -> datetime | None:
    """Get the date of the last successful query for a single prompt."""
    # Get month directories sorted in reverse (newest first)
    month_dirs = sorted(
        [d for d in prompt_dir.iterdir() if d.is_dir() and "-" in d.name],
        reverse=True,
    )

    for month_dir in month_dirs:
        queries_file = month_dir / "queries.json"
        if not queries_file.exists():
            continue

        try:
            with open(queries_file, "r") as f:
                queries = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        if not queries:
            continue

        # Parse year-month from directory name
        try:
            year, month = map(int, month_dir.name.split("-"))
        except ValueError:
            continue

        # Find the last successful query (not an error)
        for date_str, idx in reversed(queries):
            # Skip errors (e.g., "e0", "e1")
            if isinstance(idx, str) and idx.startswith("e"):
                continue

            # Parse date "DD HH:MM:SS" against the month dir it was found in
            try:
                return parse_query_date(year, month, date_str)
            except ValueError:
                continue

        # This month held only errors (or unparsable dates): try the older ones

    return None


def get_last_query_date(endpoint_dir: Path) -> datetime | None:
    """Get the date of the last successful query for an endpoint.

    Each prompt is scanned on its own: a date found for one prompt must not cut
    short the month scan of another, whose newest month may hold only errors.
    """
    dates = (
        get_prompt_last_query_date(prompt_dir)
        for prompt_dir in sorted(endpoint_dir.iterdir())
        if prompt_dir.is_dir()
    )
    return max((d for d in dates if d is not None), default=None)


def get_endpoint_info(endpoint_dir: Path) -> EndpointInfo | None:
    """Get info about an endpoint directory."""
    prompts: list[PromptInfo] = []
    model = None
    provider = None

    for prompt_dir in sorted(endpoint_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue

        info_file = prompt_dir / "info.json"
        if not info_file.exists():
            continue

        with open(info_file, "r") as f:
            info = json.load(f)

        if model is None:
            endpoint = info.get("endpoint", {})
            model = endpoint.get("model", "Unknown")
            provider = endpoint.get("provider", "Unknown")

        months = sorted(
            [d.name for d in prompt_dir.iterdir() if d.is_dir() and "-" in d.name]
        )

        if months:
            prompts.append(
                PromptInfo(
                    slug=prompt_dir.name,
                    prompt=info.get("prompt", "Unknown"),
                    months=months,
                )
            )

    if not prompts or model is None:
        return None

    last_query_date = get_last_query_date(endpoint_dir)

    return EndpointInfo(
        model=model,
        provider=provider,
        slug=endpoint_dir.name,
        prompts=prompts,
        last_query_date=last_query_date,
    )


def discover_lt_endpoints(lt_dir: Path) -> list[EndpointInfo]:
    endpoints: list[EndpointInfo] = []
    for endpoint_dir in sorted(lt_dir.iterdir()):
        if not endpoint_dir.is_dir():
            continue
        info = get_endpoint_info(endpoint_dir)
        if info:
            endpoints.append(info)
    return endpoints


def load_lt_scores(lt_dir: Path, slug: str) -> dict | None:
    """Load an endpoint's already-generated lt_scores.json, if present and non-empty.

    The raw reader behind load_lt_data, so every surface goes through the one
    on-disk contract (n_per_test/dates/scores/changes/drift/drift_dates).
    """
    path = lt_dir / slug / SCORES_FILENAME
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d if d.get("dates") else None


@dataclass
class LTData:
    """lt_scores.json with its dates parsed and drift zipped into a series."""

    dates: list[datetime]
    scores: list[float]
    n_per_test: int
    changes: list[dict]
    drift: list[tuple[datetime, float]]


def load_lt_data(lt_dir: Path, slug: str) -> LTData | None:
    d = load_lt_scores(lt_dir, slug)
    if d is None:
        return None
    drift_dates = [datetime.fromisoformat(s) for s in d.get("drift_dates", [])]
    return LTData(
        dates=[datetime.fromisoformat(s) for s in d["dates"]],
        scores=d["scores"],
        n_per_test=d["n_per_test"],
        changes=d["changes"],
        drift=list(zip(drift_dates, d.get("drift", []))),
    )


def load_all_lt_data(lt_dir: Path, slugs: Iterable[str]) -> dict[str, LTData]:
    """Parse every endpoint's scores once, so no two site surfaces parse them differently."""
    out: dict[str, LTData] = {}
    for slug in slugs:
        d = load_lt_data(lt_dir, slug)
        if d is not None:
            out[slug] = d
    return out


def latest_date(lt_data: dict[str, LTData]) -> datetime | None:
    """The build's clock: the most recent observation across all LT endpoints."""
    return max((d.dates[-1] for d in lt_data.values()), default=None)
