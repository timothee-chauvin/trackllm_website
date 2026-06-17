"""OpenRouter token-usage rankings: which models are popular.

Source: GET /api/v1/datasets/rankings-daily (top-50 models/day by tokens + an
'other' row, ~30-day window). model_permaslug carries a -YYYYMMDD suffix, an
optional :variant suffix (e.g. :free) and a provider-specific word order, so we
map to /models ids via canonical_slug (falling back to id).
"""

import re
from collections import Counter

import requests

from trackllm_website.config import config

RANKINGS_URL = "https://openrouter.ai/api/v1/datasets/rankings-daily"
MODELS_URL = "https://openrouter.ai/api/v1/models"
_DATE_SUFFIX = re.compile(r"-\d{8}$")


def aggregate_rankings(rows: list[dict]) -> list[tuple[str, int]]:
    """Sum total_tokens per permaslug over the window, drop 'other', sort desc."""
    agg: Counter[str] = Counter()
    for r in rows:
        slug = r["model_permaslug"]
        if slug == "other":
            continue
        agg[slug] += int(r["total_tokens"])
    return agg.most_common()


def map_to_model_ids(permaslugs: list[str], canonical_to_id: dict[str, str]) -> list[str]:
    """Map ranking permaslugs to /models ids, preserving order, skipping unmatched.

    A permaslug looks like ``provider/name-YYYYMMDD[:variant]``. We strip the
    optional ``:variant`` and the date suffix, then look the base up in
    ``canonical_to_id`` (which should also carry id->id entries as a fallback).
    De-dupes: a model can appear under several versioned/variant slugs.
    """
    out: list[str] = []
    seen: set[str] = set()
    for ps in permaslugs:
        core = ps.partition(":")[0]
        base = _DATE_SUFFIX.sub("", core)
        mid = canonical_to_id.get(base)
        if mid and mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


def fetch_popular_models(top_n: int) -> list[str]:
    """Return the top_n most-used model ids (popularity order). Network call."""
    headers = {"Authorization": f"Bearer {config.openrouter_api_key}"}
    rows = requests.get(RANKINGS_URL, headers=headers, timeout=30).json()["data"]
    models = requests.get(MODELS_URL, headers=headers, timeout=30).json()["data"]
    # canonical_slug first, then id->id as fallback for slugs that match an id directly.
    canonical_to_id = {m["id"]: m["id"] for m in models}
    canonical_to_id.update({m["canonical_slug"]: m["id"] for m in models})
    ranked = aggregate_rankings(rows)
    ids = map_to_model_ids([p for p, _ in ranked], canonical_to_id)
    return ids[:top_n]
