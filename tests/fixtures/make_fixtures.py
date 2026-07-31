"""One-off: extract real-data fixtures for tests. Run from repo root."""

import random
from pathlib import Path

import orjson

from trackllm_website.bi.results import load_phase2_results
from trackllm_website.config import config

FIXTURE_DIR = Path("tests/fixtures/phase_2")
# slug -> last day to keep (None = everything)
SLUGS = {
    "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8": "2026-03-12",  # clean change 2026-01-24
    "qwen2fqwen3-235b-a22b-250723wandb2fbf16": "2026-03-12",  # unstable (TV~0.47 from day 2)
    "openai2fgpt-4o-mini23azure": "2026-03-12",  # stable throughout
    "mistralai2fmistral-7b-instruct-v0.323together": "2026-03-12",  # change 2026-01-30 then death
    "tencent2fhy323atlas-cloud2ffp8": None,  # missed early change 2026-07-20 (TV -> 1.0 at batch 3)
    "z-ai2fglm-5.223deepinfra2ffp4": None,  # changed between reference burst and first batch
    "z-ai2fglm-5.223siliconflow2ffp8": None,  # genuinely unstable (day-to-day dispersion 0.54)
}
MAX_PROMPTS = 20
MAX_SAMPLES = 10
# 2026-03-12: ends before a subsampling-induced dip in the qwen3-235b trailing
# window that would otherwise mask its (genuine) instability.


def main() -> None:
    rng = random.Random(0)
    for slug, last_day in SLUGS.items():
        results = load_phase2_results(config.bi.phase_2_dir / slug)
        prompts = sorted(results)[:MAX_PROMPTS]
        ref_ts = min(ts for p in prompts for ts in results[p])
        out = {}
        for p in prompts:
            out[p] = {}
            # sorted: monthly-file key order is not stable across rewrites, and the
            # rng stream must not depend on it
            for ts, samples in sorted(results[p].items()):
                if last_day is not None and ts[:10] > last_day:
                    continue
                if ts != ref_ts and len(samples) > MAX_SAMPLES:
                    samples = rng.sample(samples, MAX_SAMPLES)
                out[p][ts] = samples
        dest = FIXTURE_DIR / slug
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "data.json").write_bytes(orjson.dumps(out))
        print(slug, sum(len(b) for b in out.values()), "batches")


if __name__ == "__main__":
    main()
