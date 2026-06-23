# RLVE difficulty-strategy results (ProRL-1.5B-v2, 4 envs)

| arm | steps | mean success | mean eff. ratio | last-5 success | held-out (final) |
|---|---|---|---|---|---|
| RLVE-90 (acc≥0.9) | 30 | 0.580 | 0.962 | 0.500 | — |
| Signal-RLVE (λ_info=0) | 30 | 0.461 | 0.974 | 0.472 | — |
| FEP-RLVE (ours) | 30 | 0.444 | 0.970 | 0.486 | — |

_eff. ratio = 1 − p^K − (1−p)^K (K=8): expected fraction of informative GRPO groups._
