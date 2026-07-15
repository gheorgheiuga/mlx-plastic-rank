# Loss-Lookahead Calibration Experiment Specification

## Objective

- **Decision enabled:** Retain, revise, or demote one-step loss lookahead as the
  V2 rank-transfer controller signal.
- **Claim under test:** At the first B-phase allocation opportunity, the V1
  predicted loss-gain ordering identifies legal transfers that produce better
  B learning over the next 12 matched updates.
- **Out of scope:** Full `A -> B -> A` promotion, cue-triggered wake, dormant
  state completeness, physical-memory conservation, large models, and theorem
  validation.

## Protocol

- **Hypothesis:** H1 in `hypothesis-ledger.md`.
- **Treatment:** The legal one-in/one-out strict-recycle transfer with maximum
  predicted B-probe loss gain.
- **Primary baseline:** Static, no-transfer branch from the identical A-trained
  checkpoint.
- **Additional baselines:** A deterministic prediction-independent legal
  transfer and the candidate selected by the A-loss probe at the B switch.
- **Ablation:** The legal transfer with minimum predicted B-probe loss gain.
- **Negative/broken control:** The minimum-predicted-gain transfer must not match
  the predicted-best treatment; failure to reject it makes the measurement
  non-discriminative.
- **Dataset and split:** Existing deterministic `tiny_mlx_dense_v1` synthetic
  train/probe/eval batches. Probe data selects and scores candidates; disjoint
  eval data measures realized B score.
- **Compute/parameter/data budget:** One guided A prefix per seed, exhaustive
  candidate scoring at the first B opportunity, and 12 B updates per candidate
  branch. All branches share the same initialized model, A checkpoint, batches,
  learning rate, active-rank budget 6, and physical rank 16.
- **Seeds:** smoke/development `0`; evidence `11-20` in canonical order. Evidence
  seeds are frozen before the evidence run and must be complete.
- **Primary metric:** Per-seed Spearman rank correlation between predicted B
  loss gain and realized 12-update B eval-score gain over static.
- **Secondary metrics:** Paired seed-level realized-gain differences for
  predicted-best versus static, predicted-worst, prediction-independent random,
  and A-loss-selected controls; budget violations; checkpoint fingerprints;
  non-finite branches.
- **Uncertainty and aggregation:** Experimental unit is the fixture seed.
  Report means and deterministic 95% percentile bootstrap intervals over the 10
  evidence seeds using 2,000 resamples. Candidate branches within a seed are
  paired counterfactual treatments, not independent replicates.

## Frozen gate

- **Pass criteria:** All 10 evidence seeds and all legal branches are finite;
  checkpoint and active-rank invariants hold; mean Spearman 95% interval lower
  bound is above zero; predicted-best minus static, predicted-worst, and
  A-loss-selected mean realized gains each have 95% interval lower bounds above
  zero.
- **Kill criteria:** Non-positive rank-correlation interval, predicted-best
  equivalence to predicted-worst, hidden budget mismatch, checkpoint mismatch,
  or a result driven by incomplete/non-finite branches.
- **Invalid-run criteria:** Any branch does not begin at the exact common
  checkpoint; active rank differs from 6; the candidate set differs between B
  and A scoring; train/probe/eval batches differ across branches; or any metric
  is non-finite.
- **Exclusions fixed in advance:** None. A failed branch fails completeness and
  remains in diagnostics.
- **Stopping rule:** Run seed 0 once for smoke, freeze implementation, run seeds
  11-20 once, then repeat seed 0 from the frozen implementation. Do not tune on
  evidence results.

## Predictions

| Outcome | Interpretation | Decision |
|---|---|---|
| Full frozen gate passes | The lookahead ordering is calibrated for short-horizon transfer utility in this fixture. | Iterate to a separately frozen multi-transfer V2; do not promote the thesis. |
| Correlation passes but selected-best comparisons fail | The signal contains ordering information but is too weak or noisy as a controller. | Iterate on aggregation or cadence using development-only seeds. |
| Immediate prediction varies but 12-update correlation fails | The proxy is myopic. | Demote one-step lookahead; require a different horizon or mechanism for re-entry. |
| Predicted-best and predicted-worst are equivalent | Transfer timing/reset dominates direction. | Demote V1 direction selection and test a fixed-cadence policy. |
| Broken/invariant controls fail | Benchmark is non-discriminative or invalid. | Park conclusions and repair measurement only. |

## Reproducibility

- **Entry command:** `.venv/bin/python scripts/loss_lookahead_calibration_benchmark.py --mode evidence --seeds 11-20 --output-dir out/capacity_migration/loss_lookahead_calibration_v1/evidence`
- **Expected outputs:** `protocol.json`, `provenance.json`, `raw_results.jsonl`,
  `summary.csv`, `summary.json`, `diagnostics.csv`, and `interpretation.md`.
- **Environment:** Python 3.13, project `.venv`, MLX/Metal on Apple Silicon.
- **Estimated runtime:** At most 10 minutes for smoke, evidence, and the frozen
  seed-0 repeat on the local Metal backend.
- **Resume/checkpoint strategy:** Each seed is an independent unit. The first
  implementation is intentionally non-resumable because the total budget is
  small; partial seed artifacts must still be retained on failure.

## Post-run record

- **Protocol deviations:** None after the successful seed-0 smoke. The first
  CLI smoke attempt failed only while serializing the MLX package version; the
  metadata lookup was fixed before artifacts were accepted or evidence seeds
  were run. No treatment, metric, horizon, seed, or gate changed.
- **Control behavior:** All 10 evidence seeds and all 27 or 32 legal branches per
  seed were finite. Every branch checkpoint matched, active rank remained 6,
  and strict-recycle checks passed. Predicted-best beat static, the
  prediction-independent random candidate, and the A-loss-selected wrong-task
  control. The predicted-worst broken control was not conclusively rejected.
- **Result:** Mean seed-level Spearman was `0.2457`, 95% CI
  `[-0.0255, 0.5159]`. Predicted-best minus static was `+0.1540`
  `[+0.0559, +0.2972]`; minus predicted-worst was `+0.0617`
  `[-0.0313, +0.1514]`; minus prediction-independent random was `+0.1305`
  `[+0.0369, +0.2695]`; and minus wrong-task-best was `+0.1632`
  `[+0.0684, +0.3017]`.
- **Decision:** **Demote** one-step loss lookahead as a directional rank-ordering
  controller. Keep only an exploratory task-conditioned move-versus-don't-move
  signal.
- **Bounded conclusion:** In this first-opportunity synthetic MLX fixture, the
  selected B-loss transfer outperformed several controls, but the full candidate
  ordering did not reliably predict 12-update utility. This does not promote
  capacity migration or validate Pop's theorem.
- **Next discriminative test:** Compare accumulated multi-batch recipient demand
  with a fixed-cadence, prediction-independent direction control across multiple
  transfers. Do not tune the demoted one-step objective on these evidence seeds.
