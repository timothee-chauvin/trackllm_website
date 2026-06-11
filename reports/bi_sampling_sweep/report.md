# BI sampling sweep: number of BIs × detection samples/BI

Detection rule held fixed (current): TV ≥ 0.5 for 4 consecutive days after 4 consecutive days below, plus a ≥ 0.5 single-day jump.
Reference = full first batch (~100 samples). Subsampling is nested: smaller configs use subsets of larger ones. Costs assume daily monitoring, 1 output token/call, averaged over endpoints with cost data.

## Summary

| Config | Avg cost/endpoint/yr | Endpoints detected | Noise floor (median TV) | Noise (median std) |
|---|---|---|---|---|
| 5 BIs × 3 samples | $0.03 | 6 | 0.249 | 0.084 |
| 5 BIs × 10 samples | $0.09 | 8 | 0.198 | 0.081 |
| 5 BIs × 50 samples | $0.46 | 7 | 0.191 | 0.083 |
| 10 BIs × 3 samples | $0.05 | 6 | 0.265 | 0.069 |
| 10 BIs × 10 samples | $0.18 | 9 | 0.196 | 0.074 |
| 10 BIs × 50 samples | $0.91 | 8 | 0.184 | 0.083 |
| 20 BIs × 3 samples | $0.11 | 7 | 0.274 | 0.068 |
| 20 BIs × 10 samples | $0.36 | 9 | 0.224 | 0.075 |
| 20 BIs × 50 samples | $1.80 | 8 | 0.201 | 0.079 |
| all BIs × 3 samples | $0.30 | 8 | 0.277 | 0.057 |
| all BIs × 10 samples | $1.01 | 9 | 0.230 | 0.061 |
| all BIs × 50 samples | $5.04 | 8 | 0.211 | 0.070 |

## Detections per config

| Endpoint | 5 BIs × 3 samples | 5 BIs × 10 samples | 5 BIs × 50 samples | 10 BIs × 3 samples | 10 BIs × 10 samples | 10 BIs × 50 samples | 20 BIs × 3 samples | 20 BIs × 10 samples | 20 BIs × 50 samples | all BIs × 3 samples | all BIs × 10 samples | all BIs × 50 samples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| deepseek-chat-v3-0324 (atlas-cloud/fp8) | 2026-02-03 | 2026-02-03 | 2026-02-03 | 2026-02-03 | 2026-02-03 | 2026-02-03 | 2026-02-03 | 2026-02-03 | 2026-02-03 | 2026-02-03 | 2026-02-03 | 2026-02-03, 2026-03-19 |
| deepseek-chat-v3-0324 (hyperbolic/fp8) | — | 2026-01-24 | 2026-01-24 | — | 2026-01-24 | 2026-01-24 | — | 2026-01-24 | 2026-01-24 | 2026-01-24 | 2026-01-24 | 2026-01-24 |
| deepseek-v3.2 (phala) | 2026-01-28 | 2026-01-28 | 2026-01-28 | 2026-01-28 | 2026-01-28 | 2026-01-28 | 2026-01-28 | 2026-01-28 | 2026-01-28 | 2026-01-28 | 2026-01-28 | 2026-01-28 |
| gemma-3-4b-it (chutes) | 2026-02-05 | — | — | — | — | — | — | — | — | — | — | — |
| phi-4 (nextbit/int4) | — | 2026-02-28, 2026-03-20 | — | — | 2026-02-28, 2026-03-20 | — | — | 2026-02-28, 2026-03-20 | — | — | 2026-02-28, 2026-03-20 | — |
| ministral-3b (mistral) | — | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 |
| ministral-8b (mistral) | 2026-01-31 | 2026-01-31 | 2026-01-31 | — | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 |
| mistral-7b-instruct-v0.3 (together) | 2026-01-30 | 2026-01-30 | 2026-01-30 | 2026-01-30 | 2026-01-30 | 2026-01-30 | 2026-01-30 | 2026-01-30 | 2026-01-30 | 2026-01-30 | 2026-01-30 | 2026-01-30 |
| pixtral-12b (mistral) | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 | 2026-01-31 |
| qwen-2.5-72b-instruct (hyperbolic/bf16) | — | — | — | 2026-01-24 | 2026-01-24 | 2026-01-24 | 2026-01-24 | 2026-01-24 | 2026-01-24 | 2026-01-24 | 2026-01-24 | 2026-01-24 |

## Plots

Overview grid: `sweep_grid.png`

- 5 BIs × 3 samples: `5-BIs-d7-3-samples.png`
- 5 BIs × 10 samples: `5-BIs-d7-10-samples.png`
- 5 BIs × 50 samples: `5-BIs-d7-50-samples.png`
- 10 BIs × 3 samples: `10-BIs-d7-3-samples.png`
- 10 BIs × 10 samples: `10-BIs-d7-10-samples.png`
- 10 BIs × 50 samples: `10-BIs-d7-50-samples.png`
- 20 BIs × 3 samples: `20-BIs-d7-3-samples.png`
- 20 BIs × 10 samples: `20-BIs-d7-10-samples.png`
- 20 BIs × 50 samples: `20-BIs-d7-50-samples.png`
- all BIs × 3 samples: `all-BIs-d7-3-samples.png`
- all BIs × 10 samples: `all-BIs-d7-10-samples.png`
- all BIs × 50 samples: `all-BIs-d7-50-samples.png`
