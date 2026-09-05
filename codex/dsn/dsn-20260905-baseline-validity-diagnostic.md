# DSN-20260905-05 — Diagnose baseline validity before controller work

- **Status:** Experimental; bounded diagnostic completed, all declared gates passed
- **Evidence status:** Ten dense references generalize on saved seeds 31–35; ten broken-pairing controls fail as expected. No controller or factorized-budget result.
- **Decision index:** ADR-0016 in `codex/decisions.md`
- **Scope parent:** DSN-20260905-04

## Question

Does the stored development data support a well-generalizing solution, or does
the observed readiness failure arise earlier than controller selection?

The joint-capacity control missed the declared 0.80 A/B gates. A high training
score and low held-out score at adjacent checkpoints on one seed motivates
checking generalization and conditioning, but does not identify the cause.
Another controller comparison would not resolve that uncertainty.

## Smallest proposed experiment

Use only the existing seed 31–35 arrays and their recorded routes, masks and
identities from
`out/capacity_migration/gradient_agreement_v1/development_20260905/inputs/`.
Verify their receipt hashes before reading results. Do not regenerate examples,
change splits, consume seeds 101–120 or run the nine-condition controller matrix.
Seed 0 may be used only for implementation smoke checks if its saved inputs are
available and verified. Missing inputs/receipts stop the diagnostic.

For each task A/B and each seed:

1. Form the dense routed design matrix by concatenating `route_s(x) * x` over
   every site. Fit the task's supervised output heads to **training rows only**.
   This matches the frozen routed substrate's linear hypothesis class without
   reading the teacher transform or oracle site labels.
2. Solve once by SVD least squares in float64, with singular values retained above
   `max(n_rows, n_columns) * eps(float64) * largest_singular_value`. Report matrix
   shape, numerical rank, singular values and condition estimates; no tuning of
   regularization or cutoff against held-out rows. NumPy is permitted here as an
   explicitly offline reference, not a runtime dependency change.
3. Evaluate that exact fitted checkpoint on both training and held-out arrays,
   using the existing loss masks and `1 - MSE / max(zero_output_MSE, 1e-12)` score.
   Report raw losses as well as scores. Do not include probe/eval targets in fitting.
4. Repeat the same fit with training target rows cyclically shifted by one as a
   deliberately broken pairing control. Preserve all results; do not try shifts
   until one produces a preferred answer.

This is ten ordinary reference fits and ten negative-control fits. It uses no
adaptive rank policy, task-specific oracle placement or joint A/B labels during
fitting. It is an unconstrained learnability diagnostic, **not** an equal-budget
competitor or evidence that rank six can learn both tasks sequentially.

## Decision rules to freeze before execution

Require complete, finite results and validated input identities. A healthy dense
reference is defined here as training **and** held-out scores at least 0.95 for
both tasks on all five seeds. This new diagnostic criterion does not replace the
historical 0.80 readiness/admission gates. Require every shifted-target control
to score below 0.50 on unshifted held-out targets; otherwise stop to inspect the
fixture/control or leakage before interpreting reference success.

| Observation | Next action |
| --- | --- |
| Dense reference passes; broken pairing control fails as expected | Investigate factorized optimization/allocation with a separately declared, matched-checkpoint diagnostic; do not attribute the gap solely to SGD |
| Training fit is good, held-out fit fails | Inspect coverage, conditioning and identifiability; record any fixture revision under a new protocol |
| Training fit fails despite full unconstrained reference | Check design/target/mask construction and solver correctness before more learning runs |
| Broken pairing succeeds, receipts mismatch, or any result is missing/non-finite | Stop interpretation and preserve the failure; diagnose the measurement first |

A reference pass does not unpark gradient agreement. Its original factorized
readiness and joint-capacity controls must still pass under a new, declared
version before a controller admission study can be considered.

## Deliverable and bounds

One small diagnostic script, focused tests for routed feature construction and
train/eval isolation, and one compact result table linked to exact input/source
hashes. Record solver/runtime versions, the fixed cutoff, per-seed outcomes and
elapsed time. Use a fresh output directory; update NOTICE and generator metadata
for the derived artifact. Bound execution to five minutes; an overrun produces
an incomplete result, not a silently reduced matrix.

Before the first development-data fit, freeze this declaration and the tested
source bytes in the result package. If implementation requires a design change,
revise this Proposed DSN before inspecting results. Publish no inferential
controller ranking and make no quality, theorem, erasure or memory-savings claim.


## Implementation declaration before the first retained-data fit

The runner is `uv run --locked python -m research.baseline_diagnostic --output-dir
out/baseline_diagnostic/<new-name>`. It pins the original completed development
receipt (`46e38b4684226981b05e1def0253671752454d7aff79cfdc6d0fa415656a5c6e`),
verifies every receipt entry, and copies the five verified input archives before
fitting. It reads only the declared training/held-out features, saved routes,
targets and binary head masks; teacher transforms and probe arrays are excluded.

The fit uses the supervised output columns, which gives the same mean squared
loss as the original binary mask. Negative-control `fit_train` scores use shifted
training targets; `train` and `heldout` scores always use the original targets.
Every fitted coefficient matrix, spectrum, cutoff and input/source identity is
retained. Condition numbers that are undefined for a singular design are null;
all fitted values and losses must be finite. The original gates are unchanged.

Unit tests use independent small artificial fixtures, including changed held-out
targets, irrelevant poisoned teacher arrays, tampered receipts and timeouts.
Source, protocol, lockfile and attribution are copied before fitting. The runner
refuses existing output directories and records partial failures and receipts.
No seed, cutoff, fit budget or gate will be changed after viewing the results.

## Recorded outcome — 2026-09-05

The first retained-data execution completed all twenty fits in 0.066 seconds.
Every reference passed the frozen 0.95 train/held-out gates; the lowest held-out
score was 0.999999999999925. Every shifted-target control scored below 0.50
(range −17.647 to −0.940). All 32 × 24 designs had numerical rank 24, with
condition numbers 19.56–74.49. The original input receipt and all 37 diagnostic
receipt entries verified; independent saved-coefficient measurements agree.

[Results and reproducibility](../research/baseline-diagnostic-results.md) link
the compact machine-readable evidence and exact artifact identities. The output
package retains the pre-run declaration and tested source bytes; this outcome
section was added afterwards. Inputs and original failed controller artifacts
remain unchanged. No reserved seed, new fixture, adaptive matrix or gate change
was used.

Follow the first decision-table branch: separately declare a factorized-baseline
diagnosis of representation, optimization and allocation at matched checkpoints.
An unconstrained fit does not establish rank-six feasibility or blame SGD alone.
The factorized readiness and joint-capacity failures remain unresolved, so
gradient agreement stays parked. No additional study has been run here.
