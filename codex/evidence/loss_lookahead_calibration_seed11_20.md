# Loss-Lookahead Calibration

**Decision:** **Demote** one-step loss lookahead as a directional rank-ordering
controller. Retain only an exploratory task-conditioned move-versus-don't-move
signal.

This test calibrates one transfer signal at the first B opportunity in the tiny synthetic MLX fixture. It cannot promote full capacity migration, cue-triggered wake, physical-memory conservation, large-model behavior, or Pop's theorem.

## Frozen gate

- PASS: `ten_frozen_evidence_seeds`
- PASS: `complete_finite_seed_branch_matrix`
- PASS: `paired_checkpoint_and_budget_invariants`
- FAIL: `predicted_gain_ranks_realized_gain`
- PASS: `predicted_best_vs_static`
- FAIL: `predicted_best_vs_predicted_worst`
- PASS: `predicted_best_vs_wrong_task_best`

## Primary result

- Mean seed-level Spearman(predicted gain, realized gain): 0.2456668779249424 (95% CI [-0.025505335837996922, 0.5159402011518945], n=10).

## Paired selected-branch comparisons

| comparison | n | mean difference | 95% CI |
|---|---:|---:|---:|
| predicted_best_vs_static | 10 | 0.15402367593563332 | [0.05590235426957557, 0.297245296785062] |
| predicted_best_vs_predicted_worst | 10 | 0.061690557823594325 | [-0.03127509432519468, 0.15138974821093928] |
| predicted_best_vs_wrong_task_best | 10 | 0.16318562027875233 | [0.06835933927058747, 0.30168131691093475] |
| predicted_best_vs_prediction_independent_random | 10 | 0.13046312062873358 | [0.03686346042376882, 0.26946139314990164] |

## Interpretation

- **Observed:** All 10 evidence seeds and every legal transfer branch were
  finite; every branch began from the exact same A-trained checkpoint, conserved
  active rank 6, and passed strict-recycle checks.
- **Observed:** Predicted-best beat static, prediction-independent random, and
  the A-loss-selected wrong-task control with paired seed-level intervals above
  zero.
- **Observed:** The candidate-level ordering was not calibrated: mean Spearman
  was `0.2457` with 95% CI `[-0.0255, 0.5159]`, and predicted-best versus
  predicted-worst was `+0.0617` with CI `[-0.0313, 0.1514]`.
- **Derived:** The probe contains task-conditioned information about whether a
  transfer can help, but this experiment does not support using its full
  one-step ordering to choose the transfer direction.
- **Inferred:** The one-step objective is probably too myopic or incomplete for
  12-update utility. A pure timing-only explanation is weakened, not eliminated.

The predeclared full gate failed. A V2 may re-enter only with a different
multi-batch or longer-horizon utility mechanism and a fixed-cadence,
prediction-independent falsification control.

## Reproduction

The frozen seed-0 configuration was repeated after the evidence run. Protocol,
raw result, summary, diagnostics, and interpretation files reproduced
byte-for-byte; time-bearing provenance was intentionally excluded from the byte
comparison.
