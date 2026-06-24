import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

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


def get_last_query_date(endpoint_dir: Path) -> datetime | None:
    """Get the date of the last successful query for an endpoint."""
    latest_date: datetime | None = None

    for prompt_dir in endpoint_dir.iterdir():
        if not prompt_dir.is_dir():
            continue

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

                # Parse date "DD HH:MM:SS"
                try:
                    day_time = datetime.strptime(date_str, "%d %H:%M:%S")
                    query_date = datetime(
                        year,
                        month,
                        day_time.day,
                        day_time.hour,
                        day_time.minute,
                        day_time.second,
                        tzinfo=timezone.utc,
                    )

                    if latest_date is None or query_date > latest_date:
                        latest_date = query_date

                    # Found latest in this month, move to next prompt
                    break
                except ValueError:
                    continue

            # If we found a date in the latest month, no need to check older months
            if latest_date is not None:
                break

    return latest_date


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
