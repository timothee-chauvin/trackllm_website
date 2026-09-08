"""The raw observations behind the LT drift lane, one file per endpoint, for the
chart's hover readout: ``website/data/lt/<slug>/daily.json``.

Schema::

    {
      "drift_prompt": 0,      # index into `prompts` of the one the drift series
                              # is computed from (lt_scores.longest_prompt)
      "prompts": [
        {
          "text": "Hi",       # the prompt (info.json)
          "tokens": [...],    # vocabulary: every token in some day's top list
          "ref": [...],       # per vocabulary token, its mean logprob over the
                              # reference period; null for a token the reference
                              # never returned, which compute_drift_series scores
                              # at the series floor:
          "floor": -17.14,    # the lowest logprob the series ever returned
          "days": [           # one per observed UTC day -- the drift series' days
            ["2025-12-03", 24, [[0, -0.11], [3, -2.9]]]
            # day, number of queries, the TOP_TOKENS tokens with the highest mean
            # logprob that day as [vocabulary index, mean logprob]
          ]
        }
      ]
    }

Logprobs are in nats, LOGPROB_FLOOR-clipped and rounded to 3 dp. The aggregation
is lt_drift.daily_means -- the same reading, flooring and day grouping the drift
series itself goes through -- so a day here is a day on the lane.
"""

from datetime import datetime
from pathlib import Path

import orjson

from trackllm_website.generate_site.lt import EndpointInfo, LTData
from trackllm_website.lt_drift import DailyMeans, daily_means
from trackllm_website.lt_scores import load_endpoint_logprobs, longest_prompt

TOP_TOKENS = 8
DAILY_FILENAME = "daily.json"


def prompt_daily(text: str, dm: DailyMeans) -> dict:
    vocab: dict[str, int] = {}
    days = []
    for day, n, mean in dm.days:
        # mean has sorted keys, so a tie between tokens resolves the same way on
        # every build
        top = sorted(mean, key=lambda t: mean[t], reverse=True)[:TOP_TOKENS]
        days.append(
            [
                day.isoformat(),
                n,
                [[vocab.setdefault(t, len(vocab)), round(mean[t], 3)] for t in top],
            ]
        )
    tokens = list(vocab)
    return {
        "text": text,
        "tokens": tokens,
        "ref": [round(dm.ref_mean[t], 3) if t in dm.ref_mean else None for t in tokens],
        "floor": round(dm.floor, 3),
        "days": days,
    }


def build_daily(
    endpoint_dir: Path, prompt_texts: dict[str, str], first_change: datetime | None
) -> dict | None:
    """`prompt_texts` maps a prompt directory name to its text. None when the
    endpoint has no drift series to explain."""
    prompts = load_endpoint_logprobs(endpoint_dir)
    if not prompts:
        return None
    means = [daily_means(data, first_change) for _, data in prompts]
    longest = longest_prompt([data for _, data in prompts])
    if means[longest] is None:
        return None
    kept = [(d, dm) for (d, _), dm in zip(prompts, means) if dm is not None]
    return {
        "drift_prompt": [d for d, _ in kept].index(prompts[longest][0]),
        "prompts": [prompt_daily(prompt_texts[d.name], dm) for d, dm in kept],
    }


def write_lt_daily(
    lt_dir: Path, endpoints: list[EndpointInfo], lt_data: dict[str, LTData]
) -> None:
    """Rewrite every endpoint's daily.json from scratch; an endpoint that lost its
    series loses its file too."""
    for f in lt_dir.glob(f"*/{DAILY_FILENAME}"):
        f.unlink()
    for ep in endpoints:
        if ep.slug not in lt_data:
            continue
        daily = build_daily(
            lt_dir / ep.slug,
            {p.slug: p.prompt for p in ep.prompts},
            lt_data[ep.slug].first_change,
        )
        if daily is not None:
            (lt_dir / ep.slug / DAILY_FILENAME).write_bytes(orjson.dumps(daily))
