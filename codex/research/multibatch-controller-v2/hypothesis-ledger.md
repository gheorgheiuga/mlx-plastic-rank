# Multi-Batch Controller V2 Hypothesis Ledger

## Research decision

- **Decision to enable:** Decide whether a strict-recycle, multi-batch rollout
  signal is informative enough to enter the full `A -> B -> A` V2 protocol.
- **Current best evidence:** One-step B-loss selection beat static, random, and
  an A-loss control at the first B opportunity, but its full candidate ordering
  and predicted-best-versus-worst gates failed. It is demoted as a directional
  controller.
- **Scope and constraints:** Tiny MLX dense-router fixture; two fixed-cadence B
  transfers at steps 0 and 12; 24 B updates; active-rank budget 6; strict
  recycle; identical A checkpoint and actual B training data across conditions.
  Seed 0 is development-only. Evidence seeds 21-30 are frozen before results are
  observed. V1 and the seed 11-20 evidence package remain unchanged.

## Hypotheses

| ID | Status | Mechanism | Falsifiable prediction | Alternatives | Decisive test | Evidence | Decision |
|---|---|---|---|---|---|---|---|
| H4 | weakened | A three-update rollout over distinct training microbatches, scored on held-out probe data after exact strict recycle, estimates persistent transfer utility better than one-step lookahead. | Across untouched seeds, the multi-batch controller beats exact one-step, fixed-random, static, and wrong-task controls on B-score AUC and localizes more B rank than random; the separate site-oracle structural control reaches full B-site rank. | H5, H6 | Two-transfer fixed-cadence controller matrix from the same A checkpoint. | Beat static, random, wrong-task, and random alignment controls; failed to separate from exact one-step over nine finite pairs, and the one-step matrix was incomplete. | Demote horizon-3 as the V2 controller. |
| H5 | weakened | Direction contributes little after transfer timing and strict reset are fixed; rank movement mainly regularizes optimization. | Multi-batch, one-step, and fixed-random policies have overlapping paired B-AUC intervals and similar final B-site rank. | H4, H6 | Same event count, actual update schedule, data, and recycle semantics across directional policies. | Horizon-3 beat fixed random on B AUC and B-site coverage, so pure timing/reset is insufficient. | Retain only as a partial explanation. |
| H6 | weakened | Extra counterfactual compute or supervised task exposure, not longer-horizon B utility, explains any multi-batch gain. | The compute-matched A-task rollout performs like the B-task rollout, or the B-task treatment fails to beat exact one-step once actual training is matched. | H4, H5 | Compare B-task horizon-3 with A-task horizon-3 and B-task exact one-step controls. | B-task horizon-3 beat wrong-task horizon-3, weakening generic compute exposure; it did not beat exact one-step, so longer-horizon value remains unidentified. | Park exhaustive shadow-rollout tuning. |

## Research queue

| Priority | Test or method | Expected information gain | Cost | Dependency | Stop condition |
|---|---|---:|---:|---|---|
| 1 | Future-aware fixed split versus any new adaptive controller | High | Low | New untouched seeds and a qualitatively new controller | Stop adaptive work if fixed split remains equivalent or better |
| 2 | Offline learned 12-update utility predictor | Medium | High | Explicit decision to resume controller discovery | Stop if held-out seeds do not beat exact one-step, fixed random, and fixed split |
| 3 | Full `A -> B -> A` V2 with unlabeled cue and complete A/B state audit | High | Medium | A new controller passes a complete directional gate | Stop if any controller or accounting gate fails |

## Decision history

| Date | Decision | Evidence | Caveat | Re-entry condition |
|---|---|---|---|---|
| 2026-07-14 | Test a new strict-recycle horizon-3 mechanism before building the rest of V2. | One-step direction ordering failed, while a narrower task-conditioned selection signal survived. | Synthetic fixture, two transfers, unequal controller-selection compute. | Full V2 is allowed only if horizon-3 passes its frozen directional and control gates. |
| 2026-07-14 | Demote horizon-3 and park further exhaustive shadow-rollout tuning. | Valid partial wins versus static/random/wrong-task and B localization, but no separation from exact one-step and one deterministic non-finite comparator seed. | Tiny supervised fixture; selection compute unequal; no return-A phase. | Re-enter only with a qualitatively new controller that beats exact one-step, fixed random, and fixed split in a complete frozen matrix. |
