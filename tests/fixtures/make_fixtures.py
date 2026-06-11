"""One-off: extract real-data fixtures for tests. Run from repo root."""

import random
from pathlib import Path

import orjson

from trackllm_website.bi.analyze import load_phase2_results
from trackllm_website.config import config

FIXTURE_DIR = Path("tests/fixtures/phase_2")
SLUGS = [
    "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8",  # clean change 2026-01-24
    "qwen2fqwen3-235b-a22b-250723wandb2fbf16",  # unstable (TV~0.47 from day 2)
    "openai2fgpt-4o-mini23azure",  # stable throughout
    "mistralai2fmistral-7b-instruct-v0.323together",  # change 2026-01-30 then death
]
MAX_PROMPTS = 20
MAX_SAMPLES = 10
# 2026-03-12: ends before a subsampling-induced dip in the qwen3-235b trailing
# window that would otherwise mask its (genuine) instability.
LAST_DAY = "2026-03-12"


def main() -> None:
    rng = random.Random(0)
    for slug in SLUGS:
        results = load_phase2_results(config.bi.phase_2_dir / slug)
        prompts = sorted(results)[:MAX_PROMPTS]
        ref_ts = min(ts for p in prompts for ts in results[p])
        out = {}
        for p in prompts:
            out[p] = {}
            for ts, samples in results[p].items():
                if ts[:10] > LAST_DAY:
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
