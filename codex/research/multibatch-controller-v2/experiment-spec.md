# Multi-Batch Controller V2 Experiment Specification

## Objective

- **Decision enabled:** Admit or reject a horizon-3 multi-batch rank-transfer
  controller as the actuator for a later full `A -> B -> A` V2 experiment.
- **Claim under test:** Exact strict-recycle candidate rollouts over three
  distinct B-train microbatches, scored on B-probe loss, choose two fixed-cadence
  transfers that improve 24-step B acquisition and B-site localization beyond
  simpler directional controls.
- **Out of scope:** Return-A behavior, unlabeled cue-triggered wake, dormant-state
  completeness, physical-memory conservation, large models, compute efficiency,
  and Pop-theorem validation.

## Protocol

- **Hypothesis:** H4 in `hypothesis-ledger.md`.
- **Treatment:** `b_horizon3`: at B steps 0 and 12, enumerate every legal
  one-in/one-out transfer, apply exact strict recycle in a restored shadow branch,
  run three SGD updates on fixed disjoint 8-example B-train microbatches, score
  terminal B-probe loss, restore the actual state exactly, and commit the best
  candidate.
- **Primary baseline:** `fixed_random`: commit a seeded prediction-independent
  legal transfer at the same two steps.
- **Additional baselines:** `static` with no transfers; `site_oracle`, which is a
  structural localization control that knows the hidden B site but not
  component utility; and `b_exact_one_step`, which uses the same exact
  strict-recycle candidate protocol but only the first microbatch and one
  virtual update.
- **Ablation:** `b_exact_one_step` isolates rollout horizon from exact commit
  semantics.
- **Negative/broken control:** `a_horizon3_wrong_task` uses the same candidate
  count, horizon, microbatch structure, and probe procedure on task A while the
  actual phase trains B. The B-task treatment must beat it.
- **Dataset and split:** Existing deterministic `tiny_mlx_dense_v1` tasks. The
  first 24 train examples form three fixed 8-example selection microbatches; the
  16-example probe batch scores candidates; the 32-example eval batch measures
  B score. All policies receive the same 24 full-batch actual B updates.
- **Parameter/data budget:** Active rank 6 and physical rank 16 for every
  condition. Every directional condition commits exactly two strict-recycle
  transfers. Actual B data and updates are matched. Controller-selection compute
  is intentionally unequal and reported as virtual gradient evaluations; no
  compute-efficiency conclusion is permitted.
- **Seeds:** smoke/development `0`; evidence `21-30` in canonical order.
- **Primary metric:** Mean B eval score across the 24 post-update checkpoints
  (`b_score_auc`).
- **Secondary metrics:** B final score, final hidden-B-site rank coverage,
  A score at B end, transfer count, selection candidate count, virtual gradient
  evaluations, checkpoint identity, budget conservation, and strict-recycle
  integrity.
- **Uncertainty and aggregation:** Fixture seed is the experimental unit.
  Deterministic paired percentile bootstrap intervals over 10 evidence seeds,
  2,000 resamples, 95% confidence. Steps and candidates are not replicates.

## Frozen gate

- **Pass criteria:** All 10 evidence seeds and all conditions are finite; all
  conditions begin at the same A checkpoint; every declared two-transfer policy
  commits exactly two transfers; active rank remains 6 and strict recycle is
  verified. `b_horizon3` must beat `static`, `fixed_random`,
  `b_exact_one_step`, and `a_horizon3_wrong_task` on paired B-score AUC with
  every 95% lower bound above zero. Its final B-site rank coverage must beat
  `fixed_random` with a lower bound above zero. `site_oracle` must reach full
  hidden-B-site rank after its available transfers as a structural sanity check;
  it is not a performance upper bound because it does not know component utility.
- **Kill criteria:** Horizon-3 matches fixed random or exact one-step; fails to
  reject the wrong-task control; improves score without B-site localization;
  exceeds the budget; hides an incomplete/non-finite condition; or only appears
  better after treating steps/candidates as independent samples.
- **Invalid-run criteria:** Candidate branches do not restore the exact current
  condition checkpoint; selection microbatches differ across candidates within
  a condition; actual B updates/data differ across conditions; any declared
  two-transfer policy misses an opportunity; or any reported value is non-finite.
- **Exclusions fixed in advance:** None. Any failed condition or seed fails the
  complete matrix and remains in diagnostics.
- **Stopping rule:** Run seed 0 smoke once, freeze implementation, run evidence
  seeds 21-30 once with `--require-pass`, then repeat seed 0. Do not retune
  horizon, microbatches, event steps, learning rate, controls, metrics, or gates
  after evidence is visible.

## Predictions

| Outcome | Interpretation | Decision |
|---|---|---|
| Full gate passes | Longer-horizon task-conditioned direction survives two transfers under valid controls. | **Iterate** to a separate full V2; do not promote capacity migration. |
| Beats random/static but not exact one-step | Extra horizon adds no identified value. | **Demote** horizon-3; retain only the prior narrow move signal. |
| Matches wrong-task horizon-3 | Extra compute or generic reset explains the result. | **Demote** task-conditioned rollout. |
| Improves B score without localization | Dense-route optimization benefit is not identified as capacity migration toward B. | **Demote** the directional migration claim. |
| Matrix or controls invalid | Measurement is non-discriminative. | **Park** conclusions and repair only the benchmark. |

## Reproducibility

- **Entry command:** `.venv/bin/python scripts/multibatch_controller_benchmark.py --mode evidence --seeds 21-30 --output-dir out/capacity_migration/multibatch_controller_v2/evidence --require-pass`
- **Expected outputs:** `protocol.json`, `provenance.json`, `trajectory.jsonl`,
  `summary.json`, `summary.csv`, `diagnostics.csv`, and `interpretation.md`.
- **Environment:** Python 3.13, project `.venv`, MLX/Metal on Apple Silicon.
- **Estimated runtime:** At most 15 minutes for smoke, evidence, and seed-0
  repeat on the local Metal backend.
- **Resume/checkpoint strategy:** Seed-level independent units; preserve partial
  rows and failure metadata. The first implementation remains non-resumable
  because the total frozen budget is small.

## Post-run record

- **Protocol deviations:** During development-only smoke, the inherited hidden-B
  site oracle reached full B-site rank but was not a performance upper bound
  because it does not know component utility. Before evidence, it was relabeled
  `site_oracle` and its invalid performance gate was replaced with a structural
  full-B-rank sanity gate. No treatment, horizon, transfer step, evidence seed,
  metric, comparative threshold, or learning rate changed.
- **Control behavior:** Horizon-3, static, random, wrong-task, and site-oracle
  completed all seeds with checkpoint, budget, strict-recycle, and shadow-restore
  invariants intact. Exact one-step seed 26 produced non-finite actual-training
  gradients; a one-condition rerun reproduced the failure. Site oracle reached
  full B-site rank for every seed. The wrong-task control used identical horizon
  and per-candidate work, but realized candidate counts diverged slightly after
  different first transfers (`186` versus `184.5` mean virtual gradients).
- **Result:** Horizon-3 minus static B AUC was `+0.1815`, 95% CI
  `[+0.0771, +0.3130]`; minus fixed random `+0.1814`
  `[+0.0360, +0.3232]`; minus wrong-task `+0.1872`
  `[+0.0645, +0.3287]`; and minus exact one-step over nine finite pairs
  `+0.0809` `[-0.0555, +0.2866]`. Final B-site coverage versus random was
  `+0.1667`, `[+0.0333, +0.3000]`.
- **Decision:** **Demote** horizon-3 as the V2 controller and park further
  exhaustive shadow-rollout tuning. Do not build the full return-A V2.
- **Bounded conclusion:** Supervised task-conditioned transfer direction matters
  across two transfers in this fixture, but the extra three-step rollout adds no
  identified value over exact one-step and the comparator matrix was incomplete.
  This does not promote capacity migration or validate Pop's theorem.
- **Next discriminative test:** Only resume with a qualitatively new controller.
  First require it to beat exact one-step, fixed random, and a future-aware fixed
  split on untouched seeds before adding return-A, cue, or state-audit scope.
