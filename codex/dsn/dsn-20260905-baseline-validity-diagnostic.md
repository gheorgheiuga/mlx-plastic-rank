# DSN-20260905-05 — Diagnose baseline validity before controller work

- **Status:** Proposed; not implemented or run
- **Evidence status:** No diagnostic result yet; motivated by the failed DSN-20260905-03 development gate
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
