# Evidence Snapshots

This directory stores compact review artifacts for PopRank experiments. These
files are small summaries that let reviewers see the measured claim without
committing generated datasets, model checkpoints, adapter packs, logs, or raw
`out/` directories.

Evidence snapshots are not release-grade datasets. Check each snapshot's
source dataset and license metadata before using a result for product or
commercial claims.

## Current bounded evidence

- [Baseline diagnosis](../research/baseline-diagnostic-results.md) and
  [measurements](baseline_diagnostic_seed31_35.json): all ten dense references
  pass; all ten broken-pairing controls fail. This selects a factorized-baseline
  question and does not establish controller benefit.
- [Forward workspace](forward_workspace_20260905.json): twelve isolated synthetic
  records, including timing follow-ups. Temporary allocations fall; gated resident
  storage is unchanged and no consistent gated latency improvement is established.
- [Gradient-agreement development](gradient_agreement_development_seed31_35.json):
  45 finite runs, failed readiness and sufficient-capacity gates. Remains parked.
- [Compact SVD workspace](svd_workspace_20260905.json): bounded numerical and
  allocation evidence, without a downstream quality claim.

## Historical observations

The model-backed screens below retain the initialization and provenance
qualifications in [ADR-0012](../decisions.md).
Their old promotion fields are preserved; they do not satisfy today's evidence
requirements retroactively.

`fault_codes_paired_control_screen_seed42.json` was an earlier local diagnostic
artifact. It records a single-seed paired falsification screen with same-budget
random, shuffled, target-constant, and cross-domain controls. Its promotion gate
passed, but its evidence status remains diagnostic until repeated across seeds,
datasets, and base checkpoints.

`text_to_sql_paired_transfer_screen_seed42.json` is the reciprocal transfer
artifact. On 300 examples, the native Text-to-SQL map beat a byte-matched
fault-code transplant with a paired 95% interval excluding zero, while retaining
97.50% of the contextual fixed-r32 PPL gain at 44.16% of its size. Together the
two screens show narrow native-map wins in both directions; multi-seed transfer
evidence is still required.

`text_to_sql_fullscale_summary.json` records the completed 10,000-row
Text-to-SQL replication. It passed the original tradeoff gate; the random and
shuffled control candidates subsequently added to its spec remain unrun.

`capacity_migration_reference_seed0_9.{json,md}` records the first conserved
`A -> B -> A` reference run. Across ten unique deterministic orthogonal
teacher/student fixture seeds, counterfactual-gradient vault and recycle
conditions conserved active rank; beat static, future-aware fixed-split, and
same-timing shuffled-recipient controls with paired 95% fixture-seed intervals
excluding zero; and separated cue-triggered wake before a parameter update from
reset-slot relearning. This validates a routing upper bound and the reference
mechanics only; it is not evidence that a trained neural model naturally
discovers, localizes, forgets, or reallocates knowledge the same way.

`capacity_migration_learned_dense_seed1_10.{json,md}` records the first frozen
learned-MLX promotion attempt. A dense input-derived router and real-loss shadow
swaps produced a positive guided-versus-static interval, a conditional positive
B-AUC interval across nine finite random pairs, and a positive post-transfer
event-window interval. The localization-versus-random gate remained incomplete
and failed because the tenth random run was non-finite; another control also went
non-finite, guided did not separate from the future-aware fixed split, and the
joint-sufficient control narrowly missed its threshold. The A-return metric was
post-supervised-loss-probe/pre-update, not cue-triggered wake. `min_rank=1`
guaranteed at least one A-site component, but some seeds retained more and dense
routes could preserve A elsewhere. Strict-recycle cleanliness and the vault's
mean dormant value of `4.3` are provisional A-column lower bounds because B-only
learned state was not audited. This is partial learned capacity-movement evidence
and a recorded negative promotion result, not validation of forgetting,
physical-memory conservation, or the central thesis.

`loss_lookahead_calibration_seed11_20.{json,md}` records the frozen
first-B-opportunity calibration of V1's one-step loss-lookahead selector. Across
10 untouched fixture seeds, every legal branch was finite, checkpoint-paired,
rank-conserving, and strict-recycle clean. Predicted-best beat static,
prediction-independent random, and an A-loss-selected wrong-task control, but
the candidate ordering was not calibrated: mean seed-level Spearman was
`0.2457` with 95% CI `[-0.0255, 0.5159]`, and predicted-best did not beat
predicted-worst conclusively. The full gate failed, so one-step lookahead is
demoted as a directional controller; only a narrower exploratory move signal
survives.

`multibatch_controller_v2_seed21_30.{json,md}` records the frozen two-transfer
horizon-3 follow-up. Horizon-3 completed all ten seeds and beat static, fixed
random, and an A-task horizon-3 control on paired B-acquisition AUC; it also
localized more final rank at the hidden B site than random. The full admission
gate failed: it did not separate from exact one-step over nine finite pairs, and
the exact-one-step seed-26 run produced reproducible non-finite training
gradients. Horizon-3 is demoted as the V2 controller, further exhaustive
shadow-rollout tuning is parked, and the full return-A V2 remains blocked.
