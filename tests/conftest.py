import json
from datetime import datetime, timezone
from pathlib import Path

from trackllm_website.bi.phase_2 import save_results
from trackllm_website.bi.selection import SelectionPolicy
from trackllm_website.bi.state import EndpointBIState, Epoch, load_all_states
from trackllm_website.bi.vetting import EndpointCache
from trackllm_website.config import Endpoint
from trackllm_website.generate_site.lt import discover_lt_endpoints, load_all_lt_data
from trackllm_website.generate_site.status import CatalogEntry
from trackllm_website.generate_site.status_io import (
    SiteStatuses,
    StatusInputs,
    resolve_site_statuses,
)
from trackllm_website.update_endpoints import LTFailureCache
from trackllm_website.util import slugify


def empty_status_inputs() -> StatusInputs:
    """StatusInputs for a synthetic site with no committed catalog/cache files."""
    return StatusInputs(
        catalog=[],
        endpoints_lt=[],
        endpoints_bi=[],
        bi_cache=EndpointCache(liars=[], too_expensive=[], bad_temperature=[]),
        lt_failures=LTFailureCache(failures=[]),
        policy=SelectionPolicy(
            budget_per_month=0.0, max_endpoint_cost=0.0, exclude=[], rules=[]
        ),
        max_cost_mtok=30.0,
    )


def catalog_entry(model: str, provider: str, **overrides) -> CatalogEntry:
    base = dict(
        model=model,
        provider=provider,
        cost=(1.0, 2.0),
        created=datetime(2026, 7, 1, tzinfo=timezone.utc),
        supports_temperature=True,
        supports_logprobs=True,
        free=False,
    )
    return CatalogEntry(**{**base, **overrides})


def site_statuses_for(root: Path, inputs: StatusInputs) -> SiteStatuses:
    """The SiteStatuses render.py would derive for this synthetic site."""
    lt_dir = root / "data" / "lt"
    lt_endpoints = discover_lt_endpoints(lt_dir) if lt_dir.exists() else []
    lt_data = load_all_lt_data(lt_dir, [e.slug for e in lt_endpoints])
    lt_by_slug = {e.slug: e for e in lt_endpoints if e.slug in lt_data}
    bi_states = load_all_states(root / "data" / "b3it" / "state")
    return resolve_site_statuses(inputs, lt_by_slug, set(), bi_states)


def b3it_slug(model: str, provider: str) -> str:
    """The slug both b3it state files and phase-2 dirs are keyed by.

    EndpointBIState derives its own slug from the endpoint, so a caller-supplied
    one that disagreed would silently split the state file from its phase-2 data
    and yield a view with an empty tv_series.
    """
    return slugify(f"{model}#{provider}")


def write_lt_endpoint(
    root: Path, slug: str, model: str, provider: str, *, dates, changes, drift
):
    d = root / "data" / "lt" / slug
    prompt_dir = d / "default"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "info.json").write_text(
        json.dumps({"prompt": "hi", "endpoint": {"model": model, "provider": provider}})
    )
    month = dates[-1][:7]
    day = dates[-1][8:10]
    month_dir = prompt_dir / month
    month_dir.mkdir()
    (month_dir / "queries.json").write_text(json.dumps([[f"{day} 00:00:00", 0]]))
    (d / "lt_scores.json").write_text(
        json.dumps(
            {
                "n_per_test": 24,
                "dates": dates,
                "scores": [0.5] * len(dates),
                "sigmas": [None] * len(dates),
                "changes": changes,
                "drift_dates": dates,
                "drift": drift,
            }
        )
    )


def write_b3it_state(root: Path, model: str, provider: str, *, status):
    """A b3it endpoint with no phase-2 data, so its view has an empty tv_series."""
    retired = (
        None
        if status == "monitoring"
        else {
            "reason": "delisted",
            "since": "2026-01-01T00:00:00Z",
            "last_recheck": "2026-01-01T00:00:00Z",
        }
    )
    state = {
        "endpoint": {
            "api": "openrouter",
            "model": model,
            "provider": provider,
            "cost": [0.1, 0.2],
            "max_logprobs": None,
        },
        "status": status,
        "retired": retired,
        "epochs": [
            {
                "start": "2026-01-01T00:00:00Z",
                "border_inputs": [],
                "reference": {},
                "end": None,
            }
        ],
    }
    sd = root / "data" / "b3it" / "state"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / f"{b3it_slug(model, provider)}.json").write_text(json.dumps(state))


def write_b3it_series(
    root: Path, model: str, provider: str, *, status, retired, month, tokens
):
    """A b3it endpoint with one epoch and a daily phase-2 series of `tokens`.

    Unlike write_b3it_state this produces a real TV series: one sampled token per
    day of `month`, compared against an all-"A" reference. A constant "A" series
    is a stable endpoint; switching token part-way yields a detected transition.
    """
    ep = Endpoint(api="openrouter", model=model, provider=provider, cost=(0.1, 0.2))
    reference = {"p1": [(f"{month}-01T00:00:00Z", "A")] * 10}
    results = {"p1": {}}
    for day, token in enumerate(tokens, start=1):
        ts = f"{month}-{day:02d}T00:00:00+00:00"
        results["p1"][ts] = [(ts, token)] * 10
    state = EndpointBIState(
        endpoint=ep,
        status=status,
        retired=retired,
        epochs=[
            Epoch(
                start=datetime.fromisoformat(f"{month}-01").replace(
                    tzinfo=timezone.utc
                ),
                border_inputs=["p1"],
                reference=reference,
            )
        ],
    )
    state.save(root / "data" / "b3it" / "state")
    p2_dir = root / "data" / "b3it" / "phase_2" / b3it_slug(model, provider)
    p2_dir.mkdir(parents=True)
    save_results(p2_dir / "p1.json", results)
