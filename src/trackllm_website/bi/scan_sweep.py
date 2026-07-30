"""Calibration sweep for the early-epoch changepoint scan (research script).

Replays the sequential procedure — scan at every batch count from min_batches
up to a window W — over all historical epochs, and reports:
- false-alarm rate on no-event epochs per (alpha, W), stable/unstable split
- detection rate and latency on the known missed early changes
- head-to-head vs the adaptive rule on the epochs where it fired

Run from the repo root: `uv run python -m trackllm_website.bi.scan_sweep`
"""

import statistics
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import fire
import orjson

from trackllm_website.bi.analyze import load_phase2_results
from trackllm_website.bi.detection import (
    adaptive_transitions,
    epoch_tv_series,
    is_unstable,
)
from trackllm_website.bi.scan import scan_pvalue
from trackllm_website.bi.state import load_all_states
from trackllm_website.config import config, logger

ALPHAS = (0.001, 0.002, 0.005, 0.01, 0.02)
WINDOWS = (20, 30, 45)
W_MAX = max(WINDOWS)

# (slug, epoch start date) -> first post-change batch index, from the July 2026
# investigation of changes absorbed at epoch start (token-verified for the
# largest ones). Index 1 = changed between the reference burst and the first
# daily batch (undetectable by the scan with a single-burst reference).
KNOWN_EARLY_CHANGES = {
    ("tencent2fhy323atlas-cloud2ffp8", "2026-07-17"): 3,
    ("tencent2fhy323gmicloud2fbf16", "2026-07-17"): 3,
    ("tencent2fhy3-preview23gmicloud2fbf16", "2026-07-10"): 4,
    ("z-ai2fglm-5.223wafer2ffp4", "2026-07-17"): 5,
    ("z-ai2fglm-5.223cloudflare", "2026-07-17"): 8,
    ("z-ai2fglm-5.223parasail2ffp4", "2026-07-17"): 3,
    ("z-ai2fglm-5.223phala", "2026-07-17"): 6,
    ("z-ai2fglm-5.223deepinfra2ffp4", "2026-07-17"): 1,
    ("deepseek2fdeepseek-v4-flash23fireworks", "2026-07-04"): 10,
    ("deepseek2fdeepseek-v4-flash23deepinfra2ffp4", "2026-07-04"): 5,
    ("deepseek2fdeepseek-v4-flash23deepseek", "2026-07-04"): 8,
    ("deepseek2fdeepseek-v4-pro23fireworks", "2026-07-10"): 1,
    ("deepseek2fdeepseek-v4-pro23deepinfra2ffp4", "2026-07-10"): 1,
    ("qwen2fqwen-2.5-coder-32b-instruct23chutes2ffp8", "2026-01-14"): 3,
    ("google2fgemma-3-27b-it23chutes2fbf16", "2026-01-14"): 10,
}


def replay_slug(slug: str) -> list[dict]:
    """Sequential scan replay for every epoch of one endpoint."""
    states = load_all_states(config.bi.state_dir)
    state = states[slug]
    results = load_phase2_results(config.bi.phase_2_dir / slug)
    out = []
    for epoch in state.epochs:
        er = epoch.filter_results(results)
        if epoch.end is not None:
            er = {
                p: {
                    ts: s
                    for ts, s in b.items()
                    if datetime.fromisoformat(ts) <= epoch.end
                }
                for p, b in er.items()
            }
        ref = {p: s for p, s in epoch.reference.items() if s}
        all_ts = sorted({ts for b in er.values() for ts in b})
        n = len(all_ts)
        if not ref or n < config.bi.scan.min_batches:
            continue
        tv = epoch_tv_series(ref, er)
        tv_ts = [t for t, _ in tv]
        onsets = [tv_ts.index(e) for e in adaptive_transitions(tv)]

        p_series, split_series = [], []
        for t in range(config.bi.scan.min_batches, min(n, W_MAX) + 1):
            keep = set(all_ts[:t])
            truncated = {
                p: {ts: s for ts, s in b.items() if ts in keep} for p, b in er.items()
            }
            event = scan_pvalue(truncated)
            p_series.append(event.p_value if event else None)
            split_series.append(all_ts.index(event.split_ts) if event else None)

        out.append(
            {
                "slug": slug,
                "start": epoch.start.date().isoformat(),
                "n": n,
                "end_reason": epoch.end_reason,
                "unstable": is_unstable(tv),
                "adaptive_onsets": onsets,
                "p_series": p_series,
                "split_series": split_series,
            }
        )
    return out


def first_fire(rec: dict, alpha: float, window: int) -> int | None:
    """First batch count t at which the sequential scan fires, if any."""
    for i, p in enumerate(rec["p_series"]):
        t = config.bi.scan.min_batches + i
        if t > window:
            return None
        if p is not None and p <= alpha:
            return t
    return None


def main(out: str = "scan_sweep_results.json") -> None:
    states = load_all_states(config.bi.state_dir)
    slugs = [
        s for s, st in sorted(states.items()) if (config.bi.phase_2_dir / s).exists()
    ]
    records = []
    with ProcessPoolExecutor() as pool:
        for recs in pool.map(replay_slug, slugs):
            records.extend(recs)
    logger.info(f"replayed {len(records)} epochs from {len(slugs)} endpoints")

    known = [r for r in records if (r["slug"], r["start"]) in KNOWN_EARLY_CHANGES]
    event = [
        r
        for r in records
        if r not in known and (r["adaptive_onsets"] or r["end_reason"] == "change_detected")
    ]
    null_pool = [r for r in records if r not in known and r not in event]

    print(f"\nepochs: {len(records)} = {len(null_pool)} null pool "
          f"+ {len(event)} adaptive-event + {len(known)} known early changes")

    print("\n== false-alarm rate on the null pool (any fire within window W) ==")
    print(f"{'alpha':>7} " + " ".join(f"{'W=' + str(w):>12}" for w in WINDOWS))
    for group, name in ((null_pool, "all"),):
        stable = [r for r in group if not r["unstable"]]
        unstab = [r for r in group if r["unstable"]]
        for alpha in ALPHAS:
            cells = []
            for w in WINDOWS:
                fs = sum(first_fire(r, alpha, w) is not None for r in stable)
                fu = sum(first_fire(r, alpha, w) is not None for r in unstab)
                cells.append(f"{fs}/{len(stable)}|{fu}/{len(unstab)}")
            print(f"{alpha:>7} " + " ".join(f"{c:>12}" for c in cells))
    print("(cells: stable-fires/stable-total | unstable-fires/unstable-total)")

    print("\n== known early changes (change batch -> fire batch, W=45) ==")
    for alpha in ALPHAS:
        fired, lat = 0, []
        for r in known:
            f = first_fire(r, alpha, W_MAX)
            if f is not None:
                fired += 1
                lat.append(f - KNOWN_EARLY_CHANGES[(r["slug"], r["start"])])
        med = statistics.median(lat) if lat else None
        print(f"alpha={alpha}: detected {fired}/{len(known)}, "
              f"median latency {med} batches")
    print("\nper-epoch detail (alpha=0.005, W=45):")
    for r in sorted(known, key=lambda r: r["slug"]):
        change = KNOWN_EARLY_CHANGES[(r["slug"], r["start"])]
        f = first_fire(r, 0.005, W_MAX)
        print(f"  {r['slug'][:52]:52s} change@{change:>2} fire@{f} "
              f"n={r['n']}")

    print("\n== adaptive-event epochs: scan fire vs rule detection (alpha=0.005) ==")
    for r in event:
        if not r["adaptive_onsets"]:
            continue
        onset = r["adaptive_onsets"][0]
        f = first_fire(r, 0.005, W_MAX)
        print(f"  {r['slug'][:52]:52s} rule detects@{onset + 2:>3} scan@{f}")

    with open(out, "wb") as f:
        f.write(orjson.dumps(records))
    logger.info(f"detailed records written to {out}")


if __name__ == "__main__":
    fire.Fire(main)
