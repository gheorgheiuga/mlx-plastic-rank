# Stored-data baseline diagnostic

DSN-20260905-05 passed every frozen gate on the existing development seeds 31–35.
All ten dense references achieved training and held-out scores above
0.9999999999999; every shifted-target control failed the 0.50 held-out gate.
No examples, splits, thresholds, reserved seeds or controller settings changed.

| Seed | Task | Reference train | Reference held-out | Broken pairing held-out | Condition number |
| --- | --- | ---: | ---: | ---: | ---: |
| 31 | A | 1.000000000000 | 1.000000000000 | -12.914690 | 47.21 |
| 31 | B | 1.000000000000 | 1.000000000000 | -5.210343 | 19.56 |
| 32 | A | 1.000000000000 | 1.000000000000 | -1.917855 | 32.19 |
| 32 | B | 1.000000000000 | 1.000000000000 | -8.939716 | 20.65 |
| 33 | A | 1.000000000000 | 1.000000000000 | -1.198267 | 19.67 |
| 33 | B | 1.000000000000 | 1.000000000000 | -8.648836 | 34.04 |
| 34 | A | 1.000000000000 | 1.000000000000 | -6.127349 | 40.64 |
| 34 | B | 1.000000000000 | 1.000000000000 | -0.939861 | 23.01 |
| 35 | A | 1.000000000000 | 1.000000000000 | -17.646994 | 26.36 |
| 35 | B | 1.000000000000 | 1.000000000000 | -13.190314 | 74.49 |

Scores are `1 - MSE / zero_output_MSE`; values rounded to twelve decimals can
appear exactly one. The lowest unrounded held-out reference score was
0.999999999999925. All ten 32 × 24 routed design matrices had numerical rank 24;
condition numbers ranged from 19.56 to 74.49. Each fit used float64 SVD least
squares and only training rows. The same coefficients supplied both reported
measurements. Negative controls used one fixed cyclic shift of training targets;
all reported held-out targets stayed unshifted. Total fitting-package runtime
was 0.066 seconds on this machine, within the five-minute bound.

## Interpretation and next decision

The saved routed data supports a well-generalizing unconstrained linear solution.
Coverage or conditioning alone does not explain why the earlier factorized
training schedule failed its gates. This result does not establish that rank six
is sufficient, that SGD is the sole problem, or that any controller is useful.
It is a diagnostic across five development fixtures, with no population inference.

The next proposed study should separate representational capacity from training
and allocation: assess whether a fixed-budget factorization can represent the
training-fitted dense reference, then compare a fixed allocation's train/held-out
results at identical checkpoints with matched initialization. Declare the exact
budget, controls, thresholds and stopping rules before that study. Do not tune a
controller or consume reserved seeds to resolve this baseline question. Gradient
agreement remains parked until its factorized readiness and joint-capacity
controls pass under a new declared protocol.

## Evidence and reproduction

The [compact machine-readable record](../evidence/baseline_diagnostic_seed31_35.json)
contains raw losses, scores, solver details, fitted-coefficient identities and
source/input hashes. The complete local package is
`out/baseline_diagnostic/stored_development_20260905/`, including all twenty fits,
input copies, singular spectra, the pre-run declaration and exact source bytes.
Its output receipt SHA-256 is
`0706a63a3bed5f9ca476a9f1996f4a79493128315d05c0a97010da4070873381`.
The original input receipt is
`46e38b4684226981b05e1def0253671752454d7aff79cfdc6d0fa415656a5c6e`.
All 81 original and 37 diagnostic receipt entries verified. An independent
calculation from the saved coefficients reproduced all forty training/held-out
measurements within 1e-10; tests also show that changing held-out data cannot
change fitted coefficients. No teacher or probe arrays enter fitting.

Use the [runbook](../runbook.md#research-handoff) for a fresh replay. The original
package remains unchanged; the current DSN includes this post-run result while
the package preserves the declaration frozen before fitting. Missing retained
inputs stop the runner rather than triggering regeneration. Source provenance
records an uncommitted working tree and exact snapshots, not a claim that the
base Git revision contains the new implementation.
