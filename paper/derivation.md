# Appendix A — why the learning signal is maximized at p = 0.5

Setup. A verifier returns a binary reward `r ∈ {0,1}`. For a fixed prompt at a
given difficulty, let `p = Pr[r = 1]` be the policy's success probability. We
draw a group of `G` i.i.d. rollouts; let `k = Σ_j r_j ∼ Binomial(G, p)` be the
number correct.

### A.1 GRPO advantages vanish on zero-variance groups
GRPO/DAPO sets the advantage of rollout `j` to
`A_j = (r_j − r̄) / (σ_r + ε)`, with `r̄ = k/G` and
`σ_r² = (1/G) Σ_j (r_j − r̄)² = (k/G)(1 − k/G) = p̂(1−p̂)`.
If `k = 0` or `k = G` then `σ_r = 0` and every `A_j = 0`: the group produces **no
policy gradient**. DAPO encodes this by discarding such groups (dynamic
sampling). So the trainable rollouts are exactly those in *informative* groups
with `0 < k < G`.

### A.2 Probability a group is informative
```
P_inf(p) = Pr[0 < k < G] = 1 − Pr[k=0] − Pr[k=G] = 1 − (1−p)^G − p^G .
```
Differentiate: `P_inf'(p) = G(1−p)^{G−1} − G p^{G−1} = 0`
⇒ `(1−p)^{G−1} = p^{G−1}` ⇒ `p = 1−p` ⇒ **`p* = 1/2`**.
Second derivative is negative there, so `p = 0.5` is the unique maximizer for all
`G ≥ 2`. (E.g. `G=8`: `P_inf(0.5) ≈ 0.992` vs `P_inf(0.9) ≈ 0.57`.)

### A.3 Expected within-group reward variance
The population reward variance is `Var[r] = p(1−p)`, maximized at `p = 0.5`
(`d/dp [p − p²] = 1 − 2p = 0`). The expected empirical variance
`E[p̂(1−p̂)] = p(1−p)(1 − 1/G)` is a positive multiple of it, so it shares the
maximizer. Thus both the *frequency* of informative groups and the *magnitude*
of the advantages they carry peak at `p = 0.5`.

### A.4 Consequence for difficulty control
Because per-environment success `p` is a monotone-decreasing function of the
difficulty level `d`, there exists a difficulty `d*` with `p(d*) = 0.5` that
maximizes the expected learning signal. RLVE's rule raises the difficulty ceiling
only once `p ≥ τ_acc = 0.9 > p*`, i.e. it keeps the model in a regime of low
signal (`P_inf(0.9) ≪ P_inf(0.5)`) and never lowers difficulty if the model
regresses. **STAD** instead servo-controls `d` so that the measured success rate
tracks `p* = 0.5`, which is exactly the band that maximizes both quantities
above. For non-binary/shaped rewards the optimum shifts away from 0.5 (it
maximizes `Var[r]` under the reward distribution), and `p*` becomes a tunable
target; we use `p* = 0.5` for the binary-correctness reward.
