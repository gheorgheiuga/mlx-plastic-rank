# Loss-Lookahead Calibration Hypothesis Ledger

## Research decision

- **Decision to enable:** Decide whether the learned V1 allocator's one-step
  loss lookahead is informative enough to retain as the controller signal for a
  V2 capacity-migration protocol.
- **Current best evidence:** On frozen seeds 1-10, guided recycling beat static
  B-acquisition AUC and had a positive matched event window, but that comparison
  does not establish that the selected transfer direction caused the benefit.
  The same-timing random comparison had only nine finite pairs.
- **Scope and constraints:** Tiny MLX dense-router fixture only; one B-phase
  transfer from a shared A-trained checkpoint; identical data, update count,
  active-rank budget, and strict-recycle semantics across branches. Development
  seed 0 is reserved for wiring. Evidence seeds 11-20 are frozen before their
  results are observed. Existing V1 outputs are not overwritten.

## Hypotheses

| ID | Status | Mechanism | Falsifiable prediction | Alternatives | Decisive test | Evidence | Decision |
|---|---|---|---|---|---|---|---|
| H1 | weakened | The one-step B-loss shadow swap ranks transfers by persistent B-learning utility. | Across evidence seeds, candidate predicted gain and realized 12-update B-score gain have positive seed-level rank correlation; the predicted-best candidate beats static, predicted-worst, and A-loss-selected controls. | H2, H3 | Exhaustively branch every legal transfer from the same checkpoint and compare prediction with realized gain. | Predicted-best beat static and A-loss-selected controls, but rank-correlation and predicted-best-versus-worst intervals crossed zero. | Demote as directional controller. |
| H2 | weakened | Rank movement at the allocation cadence acts mainly as a regularizer or optimization reset; transfer direction is not important. | Candidate prediction has no reliable relationship with realized gain, and predicted-best does not separate from random or predicted-worst legal transfers. | H1, H3 | Same exhaustive branch experiment with matched transfer timing and update budget. | Predicted-best did not separate from predicted-worst, but did beat prediction-independent random and static. | Retain as a surviving alternative, not a complete explanation. |
| H3 | supported | The lookahead is locally correct for its virtual one-step objective but does not predict utility after continued training. | Predicted one-step gains vary and select a best candidate, but rank correlation with 12-update realized gain is non-positive or the predicted-best branch does not beat static. | H1, H2 | Compare one-step prediction ordering with the frozen 12-update horizon. | Mean Spearman `0.2457`, 95% CI `[-0.0255, 0.5159]`; predicted-best versus predicted-worst `+0.0617`, CI `[-0.0313, 0.1514]`. | Best explanation under this protocol; mechanism remains bounded to the synthetic first-transfer fixture. |

## Research queue

| Priority | Test or method | Expected information gain | Cost | Dependency | Stop condition |
|---|---|---:|---:|---|---|
| 1 | Accumulated multi-batch recipient demand versus fixed-cadence random | High | Medium | New controller mechanism, not one-step retuning | Stop if it does not beat the matched fixed-cadence control on untouched seeds |
| 2 | Timing-matched direction falsification across multiple transfers | High | Low | Same stable fixture and full branch accounting | Stop if prediction-independent directions match the new controller |
| 3 | Offline 12-update utility predictor on held-out fixture seeds | Medium | High | Only if a learned controller remains strategically valuable | Stop if held-out calibration does not improve over accumulated gradients |
| 4 | Full V2 A/B state audit and unlabeled cue path | High | Medium | A controller passes its own directional gate | Stop if optimization remains non-finite after development-only calibration |

## Decision history

| Date | Decision | Evidence | Caveat | Re-entry condition |
|---|---|---|---|---|
| 2026-07-14 | Run H1/H2/H3 calibration before changing the V1 controller or scaling the benchmark. | V1 positive guided-versus-static event window lacks directionality identification. | Synthetic fixture and one transfer opportunity only. | A different controller mechanism or horizon is required if H1 is weakened. |
| 2026-07-14 | Demote one-step lookahead as a directional rank-ordering controller; retain a narrower exploratory move signal. | Full gate failed despite valid controls: ordering and best-versus-worst intervals crossed zero; best-versus-static/random/wrong-task remained positive. | First B opportunity, 12-update horizon, tiny dense-router fixture. | Re-enter only with a different multi-batch or longer-horizon utility mechanism that beats fixed-cadence random on untouched seeds. |
