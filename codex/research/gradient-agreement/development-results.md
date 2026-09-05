# Gradient-agreement development: park before evidence

**2026-09-05 — Implementation verified; development validity gate failed.**

All 45 development runs (five seeds × nine conditions) completed with finite
values, matched inputs and correct rank/state handling. The setup failed its
predeclared initial A learning and sufficient-capacity A/B checks. Evidence seeds
101–120 were not generated or run. No controller efficacy or migration benefit
is claimed.

## Implementation

- [Controller and audited state](../../../src/mlx_plastic_rank/packs/gradient_agreement.py):
  analytic removal cost, recipient agreement, gradient-energy ablation, uniform
  clipped SGD, exact branch restoration and both-factor inactive-state checks.
- [Development orchestration](../../../src/mlx_plastic_rank/packs/gradient_admission.py):
  nine conditions, common A checkpoints for selector comparisons, separate A
  training for future-aware controls, fixed partitions and complete-matrix checks.
- [Entry point](../../../scripts/gradient_agreement_benchmark.py): source-byte
  snapshots, generated-array archives, journals, output receipts, time caps and
  refusal to reuse an output directory. Evidence mode is disabled.

Historical experiments retain their behavior. The inherited A-preparation probe
remains gate-only inside its virtual calculation, with clipping added in this
separate path. Its temporary inactive learned factors are permitted only inside
that historical shadow calculation. Restoration and every real checkpoint pass
the full audit. Actual transfers and B exact-one-step branches use strict recycle.

## Failed validity checks

Scores are normalized held-out loss improvements over the zero-output baseline.
These are means over seeds 31–35. Each required mean was declared to be at least
0.80 before observing results.

| Required measurement | Observed mean | Result |
|---|---:|---|
| Common A checkpoint, before B training | 0.6062 | Fail |
| Future-aware fixed split, A before B training | 0.4823 | Fail |
| Joint capacity, A at B end | 0.6682 | Fail |
| Joint capacity, B at B end | 0.5811 | Fail |

Joint-capacity final B scores ranged from 0.4328 to 0.7600; all five missed 0.80.
There were no missing or non-finite runs. No inferential intervals or controller
ranking claims are made for this validity-only stage.

Simply extending training may be insufficient. For example, joint-capacity seed
35 had an A training score near 0.930 before its last A update, but A held-out
score 0.224 after that update. These are adjacent checkpoints, not an exact
matched-checkpoint generalization measurement. This motivates checking
optimization, data coverage and conditioning separately; it does not establish
which explanation is correct.

## Verification and artifacts

- **325 tests passed**, including 14 new cases covering analytic derivatives,
  real donor removal, cross-batch signs, identical actual/virtual clipping,
  inactive B-only corruption, rollback after a broken candidate, invalid result
  matrices, unexpected static transfers and timeout preservation.
- Ruff and mypy passed, with 56 checked source files.
- The final seed-0 run and repeat used identical source snapshots and produced
  byte-identical trajectory, event, input-identity and A-preparation journals.
  Per-condition results match after excluding elapsed times.
- Independently verified **312 receipt entries** and rank totals/bounds in all
  **2,160 development trajectory rows**.
- Development took 133.2 seconds. Including the first pilot and two final smoke
  runs, total benchmark time was 211.9 seconds, within the 30-minute budget.
  These timings include audit overhead and do not establish controller efficiency.

After the first smoke, development-only instrumentation gained frozen base-weight
hashes and validation of zero transfers in static controls. No learning rate,
clipping, task generation, condition, threshold or seed partition changed. The
development run and final two smoke runs share the same benchmark source.

The [compact record](../../evidence/gradient_agreement_development_seed31_35.json)
contains all per-seed results, aggregate measurements, package identities and
repeatability checks. Full local packages remain at:

```text
out/capacity_migration/gradient_agreement_v1/smoke_20260905/
out/capacity_migration/gradient_agreement_v1/development_20260905/
out/capacity_migration/gradient_agreement_v1/smoke_final_20260905/
out/capacity_migration/gradient_agreement_v1/repeat_seed0_20260905/
```

Each package includes source copies and dependency/protocol hashes. Its
`freeze-receipt.json` explicitly identifies a development snapshot, not an evidence
freeze. Later documentation updates do not rewrite retained snapshots. Reproduce
into a **new**, absent output directory on Apple Silicon with Metal access:

```sh
uv sync --locked
uv run --locked python scripts/gradient_agreement_benchmark.py \
  --mode development \
  --output-dir out/capacity_migration/gradient_agreement_v1/development_reproduction \
  --require-valid
```

Exit status 2 is expected for this retained failed gate. Environment versions and
the original commands are in each package's `provenance.json`.

## Decision and next diagnostic

**Park before evidence.** Preserve the thresholds and reserved seeds. The
agreement hypothesis is unconfirmed; a failed fixture gate does not establish
that the mechanism is ineffective.

The next diagnostic should use the stored development tasks: fit a dense linear
reference to the fixed routed training features, then measure it on the same
held-out arrays. This separates whether the observed data support a good solution
from whether factorized SGD finds one. It is an unconstrained diagnostic, not a
fair rank-budget competitor. Specify that test separately before running it.
Any later training-data or update-budget change needs a new protocol with the
held-out arrays fixed before another adaptive-controller matrix.

An evidence runner, successful validity receipt and freeze, untouched-seed audit,
and the eight declared paired comparisons remain prerequisites for admission.
No confirmatory mode or promotion has been enabled.
