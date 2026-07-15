# Multi-Batch Controller V2

**Decision:** **Demote** horizon-3 exhaustive shadow rollout as the V2
directional controller. Park further tuning of the shadow-rollout family unless
a qualitatively new mechanism or regime appears.

This protocol tests two supervised transfer decisions in the tiny synthetic MLX fixture. Even a passed gate would justify only iteration to a separate full V2; it cannot establish return-A behavior, unlabeled wake, physical-memory conservation, large-model behavior, compute efficiency, or Pop's theorem.

## Frozen gate

- PASS: `ten_frozen_evidence_seeds`
- FAIL: `complete_valid_control_matrix`
- PASS: `b_horizon3_vs_static_b_score_auc`
- PASS: `b_horizon3_vs_fixed_random_b_score_auc`
- FAIL: `b_horizon3_vs_b_exact_one_step_b_score_auc`
- PASS: `b_horizon3_vs_a_horizon3_wrong_task_b_score_auc`
- PASS: `b_horizon3_localizes_more_than_fixed_random`
- PASS: `site_oracle_reaches_full_b_alignment`

## Aggregate

| condition | n | B AUC | B final | B alignment | transfers | virtual grads |
|---|---:|---:|---:|---:|---:|---:|
| b_horizon3 | 10 | 0.4171406494442264 | 0.48021946249001185 | 0.6 | 2.0 | 186.0 |
| static | 10 | 0.23565269855917767 | 0.30228654512779485 | 0.4333333333333333 | 0.0 | 0.0 |
| fixed_random | 10 | 0.23574571481710893 | 0.3278440346018576 | 0.4333333333333333 | 2.0 | 0.0 |
| b_exact_one_step | 9 | 0.34146448206015606 | 0.48789162529832364 | 0.6666666666666666 | 2.0 | 61.0 |
| a_horizon3_wrong_task | 10 | 0.22994879290908982 | 0.29495512775013444 | 0.4333333333333333 | 2.0 | 184.5 |
| site_oracle | 10 | 0.3963599719548534 | 0.510114954981962 | 1.0 | 1.7 | 0.0 |

## Paired comparisons

| comparison | n | mean difference | 95% CI |
|---|---:|---:|---:|
| b_horizon3_vs_static_b_score_auc | 10 | 0.18148795088504874 | [0.07711596161477356, 0.3130467684191596] |
| b_horizon3_vs_fixed_random_b_score_auc | 10 | 0.18139493462711748 | [0.03601932734880008, 0.3232058670706056] |
| b_horizon3_vs_b_exact_one_step_b_score_auc | 9 | 0.08086013342185122 | [-0.05545697237452866, 0.2866178129269638] |
| b_horizon3_vs_a_horizon3_wrong_task_b_score_auc | 10 | 0.18719185653513665 | [0.06453817400814975, 0.3287427005468902] |
| b_horizon3_vs_fixed_random_b_final_alignment | 10 | 0.16666666666666666 | [0.03333333333333333, 0.3] |

## Failures

- `{'seed': 26, 'condition': 'b_exact_one_step', 'failure_type': 'FloatingPointError', 'message': 'non-finite value in training gradients'}`

## Interpretation

- **Observed:** Horizon-3 completed all 10 seeds, conserved active rank 6,
  committed both fixed-cadence strict-recycle transfers, restored every shadow
  checkpoint, and never observed hidden-site metadata.
- **Observed:** Horizon-3 beat static by `+0.1815`, fixed random by `+0.1814`,
  and the A-task horizon-3 control by `+0.1872`; every paired 95% interval was
  above zero. It also improved final B-site coverage over random by `+0.1667`,
  95% CI `[+0.0333, +0.3000]`.
- **Observed:** The admission gate failed. Horizon-3 versus exact one-step was
  `+0.0809` across nine finite pairs with CI `[-0.0555, +0.2866]`, and exact
  one-step seed 26 produced non-finite training gradients. The failure reproduced
  in a retained one-condition rerun.
- **Derived:** Task-conditioned direction matters under this protocol, but the
  additional three-step horizon has no identified advantage over exact one-step.
- **Inferred:** The multi-batch rollout may be more stable than the one-step path,
  but the protocol did not predeclare a stability comparison, so that remains
  exploratory only.

Selection compute was unequal by design: horizon-3 averaged `186` virtual
gradient evaluations, exact one-step `61`, and random/static `0`. The wrong-task
control used the same horizon and per-candidate work but averaged `184.5`
evaluations because divergent policies produced slightly different legal
candidate counts. No compute-efficiency claim is made.

The full `A -> B -> A` V2 remains blocked. Re-entry requires a qualitatively
different controller, a complete finite matrix, and separation from exact
one-step, fixed random, and future-aware fixed split on untouched seeds.

## Reproduction

The frozen seed-0 configuration reproduced protocol, trajectory, summary,
diagnostics, and interpretation files byte-for-byte. Time-bearing provenance was
excluded from byte comparison. The seed-26 exact-one-step numerical failure was
also reproduced independently.
