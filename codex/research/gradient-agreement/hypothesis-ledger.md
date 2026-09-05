# Gradient-agreement hypothesis ledger

**Status: Experimental; parked before evidence.** The development matrix ran and
failed its learning-validity checks. This record separates
what was measured in July, September's engineering repairs, and the proposed
mechanism. [Experiment specification](experiment-spec.md).

## Evidence map

| Item | Evidence level | What it supports | What remains unresolved |
|---|---|---|---|
| [Learned V1](../../evidence/capacity_migration_learned_dense_seed1_10.md) | Observed in retained report | Guided recycle beat static; fixed-split separation failed | Incomplete random matrix, sufficient-capacity fit, state audit and recall |
| [First-transfer calibration](../../evidence/loss_lookahead_calibration_seed11_20.md) | Observed in retained report | Task-conditioned selection beat random at one event | Full ordering and best-versus-worst gates failed |
| [Horizon-3 admission](../../evidence/multibatch_controller_v2_seed21_30.md) | Observed in retained report | Two-transfer task direction beat static/random/wrong-task | No exact-one-step separation; one non-finite comparator; no fixed split |
| [September integrity repairs](../../dsn/dsn-20260905-correctness-and-experiment-integrity.md) | Observed implementation verification | Better correctness, matched initialization option and artifact validation | No new scientific replication |
| Cross-batch dot-product identity | Derived algebra | Agreement score removes the within-batch squared terms from squared summed gradients | No proof that it predicts useful recipients over 24 training updates |
| Agreement reduces noisy allocation decisions | Speculative | Motivation for H7 | Requires the energy ablation, wrong-task control and untouched-seed test |
| [Gradient-agreement development](development-results.md) | Observed local development | 45 finite, paired, mechanically valid runs; exact seed-0 repeats; full factor audit | A readiness and sufficient-capacity learning fail; no confirmatory inference |

## Competing hypotheses

| ID | Status | Prediction | Decisive control | Consequence |
|---|---|---|---|---|
| H7: consistent recipient demand | Proposed | Agreement improves B acquisition over energy, exact one-step and all required controls while localizing B rank | Same donor rule and batches, energy instead of cross-batch dots | Advance only if the complete declared gate passes |
| H8: instantaneous gradient size suffices | Proposed alternative | Gradient energy matches or beats agreement | `gradient_energy` | Demote the agreement mechanism; do not relabel an ablation win as a passed treatment |
| H9: reserving capacity explains the gain | Live alternative from V1 | Future-aware fixed split matches or beats the adaptive strategy | Separate full A/B fixed-split trajectory from matched initial factors | Stop adaptive advancement in this fixture |
| H10: generic removal/reset or extra supervision | Partly weakened by older evidence, unresolved for this controller | Fixed random or wrong-task agreement matches treatment | Matched timing/reset; A-data agreement | Do not claim a task-direction mechanism |
| H11: optimization or measurement failure dominates | Live alternative from previous non-finite runs and incomplete audits | Ranking changes or disappears with complete finite controls; joint capacity cannot fit; restoration or state checks fail | Uniform clipping, sufficient-capacity diagnostic, full-factor audit and complete matrix | Park inference and repair the benchmark before another version |

## Research choices

| Priority | Choice | Expected information | Cost and limit | Decision |
|---|---|---|---|---|
| 1 | Conservative refinement: make numerical policy, factor identity and fixed-split comparison explicit | High: prevents another uninterpretable matrix | Small fixture; validity checks only | Required before evidence, not a new efficacy hypothesis |
| 2 | Mechanism change: cross-batch prospective recipient gradients with removal-cost donors | High: tests a signal without candidate training trajectories | Three batch gradients per event plus donor-loss calculations; runtime unmeasured | Specify now; implement and smoke before freezing |
| 3 | Falsification: energy, wrong-task, random and future-aware fixed split | High: distinguishes agreement, direction and adaptive allocation | Included in the same bounded matrix | Required, never optional post-result additions |
| Parked | Longer exhaustive rollouts, fitted utility predictor, full return-A, London/Gemma | Low immediate value or high cost before admission | Changes hypothesis or scope | No automatic continuation into these tracks |

## Relation to prior work

Gradient-informed growth and budgeted adaptation already have precedents.
[RigL, Evci et al. (ICML 2020)](https://proceedings.mlr.press/v119/evci20a.html)
updates sparse topology using parameter magnitudes and occasional gradient
calculations. [AdaLoRA, Zhang et al. (ICLR 2023)](https://arxiv.org/abs/2303.10512v2)
allocates low-rank adaptation budgets by importance and prunes singular-value
components. These primary sources motivate simple gradient and allocation
baselines; they do not establish this proposal's performance. This controller
uses a fixture-specific exact removal cost and an explicit cross-batch recipient
score. It is neither a faithful implementation of those methods nor a claim of
novelty or superiority to them. Wider claims would require actual comparative
implementations and a separate literature review.

The candidate is implemented, but the development matrix supports H11's validity
concern: readiness and sufficient-capacity learning failed. H7–H10 remain
unresolved; do not use development rankings to promote or demote efficacy. The
current decision is **park before evidence**. The alternative that fixed
reservation is enough remains live.

## Next diagnostic after the failed gate

Before changing the controller, separately specify a dense linear reference fit
on the stored routed training features and test against the same held-out arrays.
This is a high-information, low-cost check of whether the data support a good
solution independently of factorized SGD. If that reference fails, investigate
conditioning/data coverage. If it succeeds, investigate factorized optimization
and matched-checkpoint training/held-out gaps. Any changed training data or update
budget requires a new protocol with fixed evaluation arrays. This diagnostic is
proposed only; no extra fit or evidence run was performed.
