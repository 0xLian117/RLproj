# RLVE difficulty-strategy results (ProRL-1.5B-v2, 4 envs)

| arm | steps | mean success | mean eff. ratio | last-5 success | held-out (final) |
|---|---|---|---|---|---|
| RLVE-90 (acc≥0.9) | 30 | 0.552 | 0.853 | 0.519 | -0.890625 |
| Signal-RLVE (λ_info=0) | 30 | 0.491 | 0.852 | 0.444 | -0.8125 |
| FEP-RLVE (ours) | 30 | 0.477 | 0.850 | 0.531 | -0.875 |

_eff. ratio = 1 − p^K − (1−p)^K (K=4): expected fraction of informative GRPO groups._
