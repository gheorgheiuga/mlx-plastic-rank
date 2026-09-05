# Gradient-agreement controller admission protocol

**Status: Experimental, 2026-09-05. Implemented; development gate failed.**

All 45 development runs were finite and mechanically valid, but A readiness and
joint-capacity learning missed the declared thresholds. Final seed-0 runs
reproduced exactly. [Results and preserved artifacts](development-results.md).
Evidence remains disabled; the numeric thresholds below were not retuned.

This package defines the next bounded research test after the engineering review.
It does not reverse ADR-0011's failed gates. The decision is whether a controller
that uses agreement between batch gradients deserves a later, separately
specified `A -> B -> A` experiment. London/Gemma training, cue-triggered wake,
physical-memory savings, deletion, and Pop-theorem claims remain outside scope.

The numeric declaration is [protocol.json](protocol.json); the mechanism and
competing explanations are in [hypothesis-ledger.md](hypothesis-ledger.md).
The development runner reads and validates the JSON. Evidence mode is unavailable.

## Why this test

The retained learned experiment did not beat future-aware fixed split. The
first-transfer calibration failed its candidate-ordering gate. Horizon-3 then
failed to separate from exact one-step, and one comparator was non-finite.
Horizon-3 also omitted future-aware fixed split. Increasing the rollout horizon
would repeat a demoted line of work. These are observations from the retained
reports, not results obtained with September's implementation safeguards.

The proposed controller uses no candidate training rollouts. It chooses a donor
by immediate removal cost and a recipient by cross-batch agreement of the
recipient's prospective gradient. The hypothesis is that agreement rejects
batch-specific demand that gradient magnitude alone would reward. This is an
untested proxy, not a demonstrated estimate of long-term transfer value.

## Fixture and two comparisons

Use a **new** protocol ID, `tiny_mlx_gradient_agreement_v1`. Retain the dense
four-site synthetic task generator: hidden width 6, target rank 3, maximum rank 4
per site, minimum active rank 1, global active rank 6, physical rank 16, and
alpha 8 per adapter (scale 2, independent of active rank). Each task has 32 train,
16 selection-probe, and 32 evaluation examples. The router and task generation
constants are declared in the JSON. No downloaded data or model is needed.

All conditions use the same seed-derived tasks, initial `component-v1` factor
bank, model/router, actual training examples, and numerical policy. Initial
factors must match by component hash; a shared random seed alone is insufficient.
The future-aware controls also use physical rank 16; only their gates differ.

1. **Selector comparison:** Prepare one A checkpoint per seed, then copy its
   parameters and gates exactly to all seven conditions in the first group
   below. Start A at ranks `(2, 2, 1, 1)` and train for 72 full-batch updates.
   Retain the inherited `_prepare_a_checkpoint` A-allocation algorithm: the
   original one-step probe on A-probe at steps 0, 12, 24, 36, 48 and 60, including
   its no-transfer outcome and existing commit-seed formula. Implement the new
   initialization and numerical policy in a separate preparation path. This
   common setup is not evidence for the new B selector.
2. **Whole-strategy comparison:** Start future-aware fixed split and joint
   capacity from that same *untrained* factor bank, with their allocations set
   before the 72 A updates. Their A checkpoints are expected to differ from the
   common checkpoint. Pair these comparisons by fixture seed and initial bank,
   **not** by identical A checkpoint. Report A readiness before comparing B
   acquisition; do not attribute this contrast solely to B selection.

Every condition then receives 24 full-batch B updates. Transfers, where declared,
occur immediately before B updates 0 and 12. Evaluate after each update; record
the pre-B checkpoint separately. Use the same existing supervised A/B task
definitions and disjoint output-head masks. Selection-probe data is validation
data used by controllers, not held-out evidence. Evaluation arrays and hidden
task-site identifiers must not enter ordinary selector inputs.

| Condition | Setup | B allocation policy | Purpose |
|---|---|---|---|
| `agreement` | Common A checkpoint | Two transfers using the formula below | Treatment |
| `static` | Common A checkpoint | No transfers | Benefit beyond continued training |
| `fixed_random` | Common A checkpoint | Uniform seeded legal pair at each event | Direction beyond timing/reset |
| `exact_one_step` | Common A checkpoint | Exhaustive exact strict-recycle branches; one update on B-train rows 0–7; terminal B-probe loss | Required incumbent comparator |
| `gradient_energy` | Common A checkpoint | Same donor and batches as treatment; recipient uses mean squared gradient norm | Agreement-specific ablation |
| `wrong_task_agreement` | Common A checkpoint | Treatment algorithm using A-train examples and A loss mask | Supervised task-direction control |
| `site_oracle` | Common A checkpoint | Prioritize growing the hidden B site until rank 3 | Structural localization diagnostic |
| `future_fixed_split` | Same untrained bank; separate A training | Rank 2 at A and B sites, rank 1 elsewhere throughout A/B | Required whole-strategy comparator |
| `joint_capacity` | Same untrained bank; separate A training | Rank 3 at A and B sites, rank 1 elsewhere throughout A/B | Rank-8 solvability diagnostic only |

For exact one-step, every legal cross-site donor/recipient pair branches from
and restores the exact current checkpoint. Use the same recycle/reset bank as
the eventual real commit; no candidate-specific random draws. Choose minimum
terminal B-probe loss, even if no candidate improves the no-transfer branch.
Keep the no-transfer branch as a diagnostic, not an extra selectable action.

The five non-oracle transfer policies each commit exactly twice. For site oracle,
choose a donor outside B and an inactive B component, lexicographically, until
B has rank 3; no-op once it does. Assert rank 3 after the available opportunities.
It knows site identity, not useful factors, so its score is not an upper bound.

## Controller definition

At each event use three fixed, disjoint selection microbatches: train rows 0–7,
8–15 and 16–23. They are reused at the second event. All actual updates still
use all 32 train examples. No running history, learned predictor, moving average,
candidate rollout, or tunable mixture coefficient is introduced.

For each legal donor component `d`, calculate its contribution to current
predictions. With equal-size batches, define removal cost:

```text
D_d = mean_k [ L_k(prediction - contribution_d) - L_k(prediction) ]
```

Choose the donor with minimum cost, including negative costs. A donor's site
must remain at rank >= 1 and have at least one legal recipient in another site.
Use the donor-removed predictions to calculate recipient demand. This is a
read-only calculation: do not persist an intermediate rank-5 model or update any
parameters. It is exact removal cost for this additive routed fixture, not a
general formula for nonlinear networks.

For each eligible inactive recipient `c` at site `j`, let `b_c` be its current
clean input-factor row. Let `G_kj` be the gradient of batch loss with respect to
the site's effective dense weight matrix at the donor-removed prediction:

```text
E_k   = (prediction_without_donor - target) * loss_mask
G_kj  = 2 / (batch_size * sum(loss_mask)) * E_k.T @ (route_j * X_k)
g_kc  = adapter_scale * G_kj @ b_c.T

Q_c = (||sum_k g_kc||^2 - sum_k ||g_kc||^2) / (K * (K - 1)), K = 3
```

Here `route_j` multiplies each input row; weights have shape output × input.
`g_kc` is the prospective output-factor gradient of a zero-output recipient,
without the gate-zero gradient that ordinary autodiff would otherwise produce.
All formulas use fp32 arithmetic and the existing masked mean-squared loss.
Check analytic gradients against an ungated zero-output autodiff probe before
admitting the implementation. No hidden task site or teacher transform is used.

Select the legal recipient with maximum `Q_c`, even if every score is negative.
Break all ties by `(adapter_name, component_index)`. The ablation replaces `Q_c`
with `mean_k ||g_kc||^2`; donor cost, observations, event count and commit policy
remain identical. Algebraically, `Q_c` averages cross-batch gradient dot products.
Its interpretation as persistent demand is a hypothesis; the three fixed batches
are not independent observations of future training trajectories.

Commit the selected pair with strict recycle. Zero the donor output column and
replace its input row with the existing deterministic reset draw; activate only
a clean recipient. Match the existing B-event seed formula
`fixture_seed * 1_000_000 + step * 1_000 + 17` across conditions and branches.
Use `random.Random(fixture_seed * 1_000_003 + step * 101 + 29)` solely for the
uniform random pair over a lexicographically sorted candidate list.

## Numerical and state policy

The proposed change from the old experiment is **uniform global gradient-norm
clipping at 1.0**, with the original SGD learning rate 1.5 and no optimizer
momentum. Apply it to every actual A/B update and every virtual update in A
preparation and the exact-one-step comparator. Use the same operator:
`g_clipped = g * min(1, 1 / max(global_norm(g), 1e-12))`. Analytic selection scores
use unclipped gradients; they are observations, not updates. Record pre-clip
norms and clipping frequency. Check finite loss, gradient, masters and materialized
adapter factors before/after each update; clipping does not excuse a non-finite
input. A non-finite candidate probe also fails that condition; never discard the
candidate and choose a finite alternative. Keep fp32 masters and the existing
fp16 adapter materialization.

This is a new stabilized comparison, not a rerun of July's numerical protocol.
No learning-rate grid or seed-specific rescue is allowed. If development fails
numerically, park this draft and record a new proposed version before changing
the policy. Old evidence remains untouched.

Audit **both factors**, not only zero output columns. Every inactive component
must have zero A and B equal to its initial or most recent recorded reset row,
in masters and the materialized dtype. Selection must leave all parameters,
gates and reset metadata unchanged. Commit must preserve non-donor master
parameters exactly; active rank must equal 6 at every real checkpoint (8 only for
the explicitly labeled joint-capacity diagnostic). Log physical factor bytes,
master bytes, optimizer state, inactive state and temporary selector workspace
separately. There is no dormant vault in this admission test. None of these
checks establishes historical erasure, cue-triggered recall or total-memory
conservation.

## Stages, budget and frozen gate

The proposed partitions are smoke/repeat seed `0`, development seeds `31–35`,
and evidence seeds `101–120` in ascending order. The [local seed audit](seed-audit.json)
found no reserved-seed intersection in 32 retained JSON/JSONL artifacts. It
conservatively includes bootstrap seed fields too. This is a bounded local
inventory, not proof that the seeds were never used elsewhere. Recheck at freeze;
any known prior inspection of an evidence task/result disqualifies the seed
before evidence starts, and requires a documented revised partition.

1. Implement separate fixture/controller/runner paths and meaningful invariant
   checks. Do not alter frozen historical configurations or silently reuse their
   CLI commands. The separate development entry point is now implemented; see
   [commands and results](development-results.md).
2. Run seed 0 mechanics, then the complete nine-condition development matrix on
   31–35, then repeat seed 0. Cap this stage at 30 minutes of benchmark time.
   Require determinism, a finite matrix, state/data boundaries, and solvability
   checks below. Development is for validity, not ranking or tuning controllers.
   If it fails, stop before accessing evidence seeds.
3. Freeze source bytes (including dirty changes), dependencies, generators,
   reset/initialization identities and this protocol in a receipt. Record a
   runnable command and measured smoke time. Only then enable the evidence mode.
4. Run all 20 evidence seeds once, with a 60-minute hard benchmark budget, then
   repeat smoke. Preserve failures and partial output. No automatic new sweep,
   replacement seed, dropped row, optional extra seeds or post-result tuning.
   A timeout or numerical failure fails the complete-matrix gate. Crashes must
   preserve their condition/event and completed rows; reruns are diagnostic only.

No statistical power has been measured. The development matrix took 133.2 seconds;
all pilot/repeat runs together with it took 211.9 seconds, within the budget.
Twenty paired seeds is a bounded screening budget, not a power guarantee. These
resource caps authorize no cloud spend or large-model job.

**Primary metric:** Per-seed mean of 24 post-update B evaluation scores,
`1 - masked_MSE / zero_model_masked_MSE`, without score clipping. Steps,
candidates and examples are not independent replicates. Record B final score,
A scores before/after B, event choices, rank maps, hidden-B coverage
`min(rank_B / 3, 1)`, selection operations, virtual updates, actual updates,
wall time, peak memory and all failures as secondary diagnostics.

**Inference:** Use 20,000 paired percentile bootstrap resamples over the 20
fixture seeds, NumPy `Generator(PCG64(20260905))`, and the same resample-index
matrix for all contrasts. Report ordinary two-sided 95% intervals and
Bonferroni-adjusted 99.375% intervals for the eight declared comparisons below.
The adjusted quantiles are 0.003125 and 0.996875, using linear interpolation.
This conservative adjustment addresses the declared comparison family; it does
not establish power or exact small-sample coverage.

**Admission requires every condition below; no exclusions:**

- Complete, finite 20 × 9 matrix; matching inputs and appropriate checkpoint
  identities; exact conservation, event counts, restoration and two-factor
  recycle audits; repeated smoke deterministic in event choices and hashes.
- Treatment B-AUC exceeds each of `static`, `fixed_random`, `exact_one_step`,
  `future_fixed_split`, `gradient_energy` and `wrong_task_agreement`: each paired
  mean difference >= 0.02 and each adjusted lower confidence bound > 0. The 0.02
  threshold is a proposed minimum useful change in this normalized score, chosen
  before results rather than estimated from this fixture.
- Treatment final hidden-B coverage exceeds random with adjusted lower bound
  > 0; the site oracle reaches rank 3 at B in every seed.
- For the whole-strategy comparison, both the common-A and fixed-split mean A
  readiness scores must be >= 0.8. Common-A minus fixed-split A readiness must
  have adjusted lower bound > -0.02, the eighth comparison. This is a readiness
  safeguard, not identical-checkpoint evidence or a retention claim.
- Joint capacity has mean A and B scores >= 0.8 at B end. This tests whether the
  declared training schedule can solve both tasks with sufficient capacity. It
  contributes no same-budget improvement claim. Apply these absolute readiness
  and solvability checks in development too, without inferential gates there.

If fixed split is better or indistinguishable, do not advance adaptive migration.
If raw gradient energy is equivalent, demote the agreement-specific explanation.
If wrong-task agreement matches treatment, task direction is not identified.
If only quality or only localization improves, the full admission gate fails.
If numerical/solvability/audit checks fail, the fixture is not admissible;
preserve results and repair the benchmark under a new version. Even a complete
pass permits only design of a separate full return-A protocol; it does not
validate the thesis or permit a large-model experiment.

## Reproducibility and current readiness

The development runner produces `protocol.json`, `freeze-receipt.json`, `provenance.json`,
`trajectory.jsonl`, `events.jsonl`, `failures.jsonl`, `summary.json`, paired
interval status and an interpretation. Its freeze receipt is explicitly a
development source snapshot; no intervals are computed for development.
Bind source bytes, lockfile, all generated
arrays/masks, split identities, checkpoints, reset rows and output hashes.
Reject existing or mismatched output directories. Preserve incomplete output
without treating it as a completed experiment.

Use project-local Python 3.13 and the locked uv environment; no new dependencies
are proposed. All eventual commands use `uv run --locked`. A seed audit can be
repeated by parsing JSON/JSONL under the exact paths and keys in the accompanying
audit, hashing each source file and intersecting all discovered integers with the
reserved evidence set. It is an inventory, not a training command.

**Definition checks completed:** JSON partitions, comparison counts, confidence
quantiles, local links and all 32 inventory hashes were verified. An independent
float64 finite-difference check of the prospective-gradient formula had maximum
absolute error below `8e-11`; the cross-batch identity and equal/opposing-gradient
sign checks passed. These initial checks used an algebra-only fixture. Subsequent
MLX gradient, state and controller checks also passed, as recorded in the
development result; the reserved tasks remain unused.

**Current boundary:** The initialization/preparation path, shared clipped update,
analytic selector and ablation, both panels, factor audit, output receipts,
development matrix checks and smoke/failure tests are implemented. The learning
gate failed. An evidence freeze, confirmatory runner and paired inference remain
pending and must not be enabled for this failed protocol.
