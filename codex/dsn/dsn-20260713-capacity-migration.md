# Decision Support Note (DSN)

**ID:** DSN-20260713-02
**Title:** Test conserved low-rank capacity migration
**Date:** 2026-07-13
**Status:** Experimental
**Evidence Status:** Ten-seed counterfactual reference mechanics passed; first frozen learned-MLX promotion gate failed with partial movement/performance evidence
**Related Research Inbox Entry:** Neuroplastic rank growth, sleep, wake, and reallocation

---

## Context

Plastic Rank began as a reversible-capacity project. The first public README
described factors that could "grow, shrink, or wake as needed," and the original
runbook required pruned components to be parked for possible reactivation. Later
work made valuable progress on skill packs, static heterogeneous rank maps, and
quality per exported byte, but that proxy gradually displaced the founding
question:

> Can rank show capacity migrating as a model learns, specializes, forgets, and
> becomes able to learn again?

Existing results show that rank placement can matter locally. They do not show
capacity moving through time, that relinquishing one skill improves acquisition
of another, or that a recovered skill was erased rather than made inaccessible.

## Options Considered

1. Continue treating a smaller exported rank map as the primary outcome.
   - Pros: Reuses mature pack and bakeoff tooling with clear size and quality
     metrics.
   - Cons: Measures a final allocation, not migration, interference, forgetting,
     or renewed plasticity.
2. Allow every useful site to grow independently during sequential learning.
   - Pros: Tests whether utility signals can find useful sites.
   - Cons: Any improvement may come from extra capacity; there is no donor cost
     and therefore no migration claim.
3. Enforce a one-in/one-out global active-rank budget across sequential tasks.
   - Pros: Makes every growth event identify a donor, exposes trade-offs, and
     creates a falsifiable test of reallocation and recovery.
   - Cons: Requires a conserved allocator, explicit accounting for dormant
     factors, and sequential controls in one model instance.

## Decision

- Chosen option: 3.
- Restore conserved capacity migration as the canonical Pop Rank research
  thesis. Build the smallest sequential `A -> B -> A` benchmark that can falsify
  it before scaling to a language model or a London-navigation narrative.
- Keep static rank-map and quality-per-byte results as supporting evidence. The
  rank ledger is the measuring instrument; dynamic rank control is the actuator.
  Neither is itself proof of migration or quality benefit.
- Canonical decision record: `codex/decisions.md` ADR-0011.

## Terms That Must Not Be Conflated

- **Active capacity:** the declared effective-rank count currently participating
  in the forward pass. Its budget is fixed at `R` over the declared pool in the
  core experiment; this does not itself conserve physical parameter bytes,
  optimizer state, resident learned factors, or information.
- **Dormant capacity:** learned factors retained outside the active computation
  budget. It must have its own reported rank and byte ledger; it supports a
  "cannot currently remember" interpretation, not capacity erasure.
- **Recycled capacity:** donor slots are overwritten or reinitialized for the
  new task while active and resident learned-factor counts remain bounded. The
  reference student still preallocates every possible physical slot, so this
  condition does not establish parameter-memory conservation.
- **Forgetting:** an observed loss of task access or performance. The benchmark
  must identify whether this came from gating, retained latent traces,
  interference, or overwriting rather than treating all four as deletion.
- **Relearning:** recovery after task A returns, measured against immediate
  wake, fresh learning, and never-trained controls.

## Minimal Sequential Experiment

Use one small frozen teacher/student system with several equal-shaped low-rank
sites. Task A and task B require different hidden sites. Run all phases in one
process, with one model and one adapter substrate:

1. Establish the unadapted baseline.
2. Learn A under global active-rank budget `R`.
3. Switch to B without A rehearsal. Every rank unit given to a B site must be
   taken from another active site in the same allocation event.
4. Reintroduce A and separately measure unlabeled cue-triggered recall before any
   supervised loss probe, post-probe/pre-update access, and subsequent relearning.
   Learned V1 measured only the second of these and therefore did not test
   cue-triggered wake.

For this first benchmark, "global" means the complete declared pool of
equal-shaped `attn.q_proj` target sites; other adapter sites must be absent or
frozen. That restriction makes one rank unit comparable across donors. Later
experiments may conserve parameter bytes rather than raw rank across mixed
projection shapes.

The planned implementation surface is:

- `LoRAManager.active_rank_state(...)` for an auditable allocation snapshot;
- `LoRAManager.adjust_conserved_ranks(total_active_rank, min_rank=1,
  max_transfers=1, seed=0, target_suffix="attn.q_proj")` for atomic
  one-in/one-out transfers;
- `src/mlx_plastic_rank/packs/capacity_migration.py` for the sequential
  experiment and trajectory report;
- `scripts/capacity_migration_benchmark.py` for the reproducible command-line
  run; and
- `tests/research/test_capacity_migration.py` for budget and control invariants.

The learned bridge subsequently added
`src/mlx_plastic_rank/packs/learned_capacity_migration.py`,
`scripts/learned_capacity_migration_benchmark.py`, and
`tests/research/test_learned_capacity_migration.py`. It uses a frozen input-derived dense
router, real-loss shadow swaps, transactional strict recycling, and separate
active/physical/master ledgers plus a provisional A-column dormant-state ledger.

At minimum compare:

- conserved utility-guided migration;
- frozen A-only static allocation, whose B score is zero by construction;
- a future-aware fixed split that reserves equal active capacity for A and B;
- random one-in/one-out migration with the same event schedule;
- an oracle allocator that knows the hidden sites;
- extra capacity (`2R`) to show whether A and B are jointly learnable;
- a vault condition that retains inactive A factors and a recycle condition
  that overwrites the same slots;
- a never-trained-on-A reference for interpreting relearning savings.

Utility-based pruning must rank components by an explicit contribution measure,
such as `||A[:, j]|| * ||B[j, :]||`, or a documented alternative. Dropping a
fixed column prefix or suffix is not evidence that the least useful capacity
moved.

The current manager keeps `min_rank >= 1` so every managed site retains a live
gradient path. Unit-transfer allocations are experimental runtime states; odd
ranks must be rebalanced to a profile-supported rank map before pack export.
True rank-zero sleep requires an external demand probe or explicit wake policy
because a hard-zero gate cannot generate its own discovery gradient.

## Reference Mechanics Result — 2026-07-13

The pure-Python orthogonal teacher/student runner completed ten unique fixture
seeds, 0-9, with all declared reference-mechanics gates passing. Vault and
recycle receive idealized component-level counterfactual gradient demand derived
from the synthetic task targets. That signal exposes where loss could improve,
so it is a routing upper bound rather than leak-free learned discovery. The
oracle remains distinct because it reallocates directly and immediately from
the hidden site identifier.

- Active rank stayed exactly at `R=2` for every conserved condition and step.
- Vault and recycle both reached mean task-B AUC `0.9002`, versus `0.0000` for
  static allocation, `0.5083` for a future-aware fixed split, and `0.3623` for
  same-timing shuffled-recipient transfer.
- Paired 95% fixture-seed bootstrap intervals excluded zero for vault/recycle
  versus all three same-budget controls; the narrowest lower bound was `0.3633`.
  These intervals summarize variation in this constructed fixture, not
  learned-model uncertainty.
- Vault made A behaviorally inaccessible after B while retaining latent score
  `0.99998` and restoring it after an A-derived retrieval cue but before any
  parameter update. Its maximum resident rank was `4`, correctly exposing stored
  dormant capacity outside the active budget.
- Recycle retained resident rank `2`, had no immediate A return, and matched the
  never-trained-A control under the same transfer schedule exactly (relearning
  AUC advantage `0.0`).
- The extra-capacity `2R` control retained both A and B, confirming that the
  conflict under `R` was capacity-induced in this constructed system.

Compact evidence is recorded at
`codex/evidence/capacity_migration_reference_seed0_9.{json,md}`; the full JSONL
trajectory remains under `out/capacity_migration/reference_seed0_9/`.

This is a counterfactual reference-mechanics pass, not promotion of the central
thesis. The orthogonal teacher exposes unusually clean demand and does not show
that a trained neural model will localize or reallocate knowledge similarly.

## First Learned-MLX Result — 2026-07-13

Protocol `tiny_mlx_dense_v1` was developed on seed 0, frozen, and then run on
untouched confirmatory seeds 1–10. The allocator saw real loss, parameters, and
active masks, not task-site metadata. All route weights were nonzero, every
route design had rank four, A and B used distinct rank-three transforms on
disjoint output heads, and the joint task had an analytic exact solution.

The result was `learned_dense_capacity_migration_gate_failed`:

- Effective active rank remained exactly conserved. Physical rank 16, fp16
  adapter storage 384 bytes, float32 master storage 768 bytes, and zero-byte
  stateless optimizer state were reported separately.
- Guided recycle B-acquisition AUC was `0.5176`, versus `0.2792` static. The
  paired mean advantage was `+0.2384`, 95% fixture-seed interval
  `[+0.1822, +0.2974]`.
- Against nine finite random pairs, the conditional guided advantage was
  `+0.1268`, 95% interval `[+0.0631, +0.1802]`. Because the tenth random run was
  non-finite, the localization-versus-random gate was incomplete and failed.
- The first loss-guided transfer during the B phase preceded a mean 12-step
  advantage over its score-matched static twin of `+0.1431`, 95% interval
  `[+0.0807, +0.2052]`.
- Guided recycle did not beat the future-aware fixed split conclusively:
  `+0.0227`, 95% interval `[-0.0346, +0.0816]`.
- Random seed 2 and extra-capacity seed 3 produced non-finite gradients, so the
  complete finite seed/condition matrix failed.
- The nine finite joint-sufficient runs averaged A-after-B `0.7777` and B-final
  `0.7983`, narrowly below the frozen `0.8` gate.
- Strict recycle's mean A score was `0.2569` after a supervised A-loss probe and
  any resulting gate transfer but before a parameter update. This was not an
  unlabeled cue-triggered wake test. `min_rank=1` guaranteed at least one A-site
  component remained active, while some seeds retained more and dense routes
  could preserve A through other sites; the score therefore cannot be attributed
  to the floor alone.
- The vault's corresponding post-supervised-loss-probe/pre-update value was
  `0.5270`. Its reported mean dormant learned rank of `4.3`, and the strict-
  recycle cleanliness check, are provisional A-column lower bounds: the V1
  audit did not detect learned state stored only in B rows.

This is the first learned-model evidence that loss-guided effective-rank
movement can precede a matched performance benefit. It is also a negative
promotion result: optimization was not fully stable, a strong fixed reservation
remained competitive, the ten-seed localization-versus-random gate was
incomplete, and learned V1 did not test cue-triggered wake or fully audit dormant
factor state. The London experiment is not promoted and the Forgetting Machine
remains parked. Compact evidence is in
`codex/evidence/capacity_migration_learned_dense_seed1_10.{json,md}`.

## Loss-Lookahead Direction Calibration — 2026-07-14

The strongest remaining V1 ambiguity was whether the one-step shadow-swap
objective identified useful transfer directions or whether any move at the
matched cadence produced the observed event-window gain. Protocol
`loss_lookahead_calibration_v1` exhaustively branched every legal first-B-phase
transfer from one exact A-trained checkpoint, then trained each branch for 12
matched B updates. Seed 0 was used only for smoke; untouched evidence seeds
11–20 were frozen in advance.

The full gate failed despite a complete and valid measurement matrix:

- every legal branch across all 10 seeds was finite, checkpoint-paired,
  active-rank conserving, and strict-recycle clean;
- predicted-best beat static by `+0.1540`, 95% CI
  `[+0.0559, +0.2972]`, prediction-independent random by `+0.1305`
  (`[+0.0369, +0.2695]`), and an A-loss-selected wrong-task control by
  `+0.1632` (`[+0.0684, +0.3017]`);
- candidate predicted gain did not reliably rank realized 12-update gain:
  mean seed-level Spearman was `0.2457`, 95% CI
  `[-0.0255, +0.5159]`; and
- predicted-best did not beat predicted-worst conclusively: `+0.0617`, 95% CI
  `[-0.0313, +0.1514]`.

Demote one-step lookahead as a directional rank-ordering controller. A narrower
task-conditioned move-versus-don't-move signal survives as exploratory evidence,
because best-B selection beat static, random, and the wrong-task selector. This
does not rescue V1 or promote capacity migration. Re-entry requires a different
multi-batch or longer-horizon utility mechanism that beats a fixed-cadence,
prediction-independent direction control on untouched seeds. Compact evidence
is in `codex/evidence/loss_lookahead_calibration_seed11_20.{json,md}`; full raw
branches and provenance remain under
`out/capacity_migration/loss_lookahead_calibration_v1/evidence/`.

## Multi-Batch Controller V2 Admission Test — 2026-07-14

Protocol `multibatch_controller_v2` tested a genuinely different selector before
adding return-A scope. At B steps 0 and 12, the treatment applied exact strict
recycle in restored shadow branches, trained each legal candidate over three
fixed 8-example B microbatches, scored terminal B-probe loss, and committed the
best transfer. Static, fixed-random, exact-one-step, A-task horizon-3, and
hidden-B-site structural controls shared the same A checkpoint and 24 actual B
updates. Seed 0 was development-only; seeds 21–30 were frozen evidence.

The admission gate failed with a real partial signal:

- horizon-3 beat static B AUC by `+0.1815`, 95% CI
  `[+0.0771, +0.3130]`, fixed random by `+0.1814`
  (`[+0.0360, +0.3232]`), and A-task horizon-3 by `+0.1872`
  (`[+0.0645, +0.3287]`);
- final hidden-B-site coverage beat random by `+0.1667`, 95% CI
  `[+0.0333, +0.3000]`;
- horizon-3 did not separate from exact one-step over nine finite pairs:
  `+0.0809`, 95% CI `[-0.0555, +0.2866]`; and
- exact-one-step seed 26 produced non-finite actual-training gradients, so the
  complete finite matrix failed. A one-condition rerun reproduced the failure.

Demote horizon-3 as the V2 controller and park further exhaustive shadow-rollout
tuning. The narrower controlled result is that supervised task-conditioned
direction can matter across two transfers in this fixture; the extra rollout
horizon added no identified value. Selection compute was unequal by design, so
no efficiency claim is made. The full return-A V2 remains blocked. Compact
evidence is in `codex/evidence/multibatch_controller_v2_seed21_30.{json,md}`;
full artifacts remain under
`out/capacity_migration/multibatch_controller_v2/evidence/`.

## Evidence and Promotion Gate

All conditions must share initialization, samples, optimizer schedule, and
allocation opportunities where applicable. Report trajectories, not only final
scores: per-site active rank, dormant rank and bytes, transfer events, task loss,
and task accuracy at every phase.

The learned-model hypothesis advances beyond Experimental only if:

1. The global active-rank budget is exactly conserved after every transfer; any
   dormant store is separately counted.
2. After the A-to-B switch, utility-guided rank moves toward the hidden B site
   more reliably than random transfer across at least 10 seeds.
3. Utility-guided migration improves B acquisition AUC or steps-to-threshold
   over static, future-aware fixed-split, and random same-budget controls, with
   a paired 95% interval excluding zero.
4. Under the strict recycle condition, releasing A capacity improves B learning
   without increasing declared learned-factor bytes or optimizer state beyond
   matched controls. The preallocated reference fixture cannot satisfy this
   learned-model gate on its own.
5. After an A-derived retrieval cue but before any parameter update, the vault
   condition distinguishes immediate wake from recycled capacity. Subsequent A
   relearning is compared with the never-trained reference so residual savings
   are reported rather than called deletion.
6. Rank movement predicts or precedes corresponding performance movement; a
   changing ledger alone is insufficient.

## Kill or Demotion Criteria

Demote the capacity-migration mechanism, revise its controller, or reject the
strong thesis if any replicated result shows that:

- the conserved allocator exceeds `R` or hides material learned state outside
  the declared dormant-capacity ledger;
- random transfer performs equivalently to utility-guided transfer;
- rank moves without improving B acquisition or renewed A plasticity;
- B improves only in the extra-capacity condition, not by reallocating `R`;
- the effect disappears when component order is randomized or components are
  ranked by contribution before pruning;
- apparent forgetting is fully explained by a reversible gate while being
  presented as erasure; or
- a learned-model allocator receives direct task identity, target-support, or
  equivalent leakage. The current synthetic counterfactual-gradient fixture is
  explicitly an upper bound and cannot satisfy the learned-model promotion gate.

## Consequences

- The learned MLX recipient-demand bridge is complete as frozen V1. The immediate
  target is not another exhaustive shadow-rollout selector. One-step and
  horizon-3 both failed their frozen admission gates. Full V2 remains blocked;
  re-entry requires a qualitatively new controller that first beats exact
  one-step, fixed random, and future-aware fixed split in a complete frozen
  matrix—not a large London/Gemma experiment.
- Existing pack experiments remain valid local rank-placement evidence but no
  longer define the project's central success criterion.
- The Forgetting Machine external-vault prototype is parked separately because
  governed record deletion does not answer whether model capacity migrates.
- A public-facing city-navigation experiment is justified only after the
  synthetic phenomenon survives the promotion gate.

## Follow-ups

- [x] Implement and test atomic one-in/one-out transfers under a fixed budget.
- [x] Implement contribution-aware donor selection and a matched random
  transfer control.
- [x] Run the synthetic `A -> B -> A` matrix across at least 10 seeds.
- [x] Record a compact summary and paired statistical report; keep the full
  trajectory under ignored `out/` artifacts.
- [x] Add measured gradient/loss-probe recipient demand to one sequential MLX
  training process without task-site leakage; record its failed first promotion
  gate before revising the controller.
- [x] Calibrate one-step loss lookahead against every legal first-B transfer on
  untouched seeds; record the failed ordering gate and demote it as a directional
  controller.
- [x] Test a genuinely different horizon-3 selector across two fixed-cadence
  transfers; record its failed admission gate, demote it, and park further
  exhaustive shadow-rollout tuning.
- [ ] Before any full V2, require a qualitatively new controller to beat exact
  one-step, fixed random, and future-aware fixed split in a complete frozen
  matrix. Only then add rank-zero/A1-matched forgetting, an unlabeled cue path,
  the A/B baseline ledger, and full A-and-B state accounting.
- [ ] If the learned-model gate passes, design the `Where Does London Go?`
  audience benchmark.
- [x] Record the failed gate as a negative result before changing the allocator
  or task construction.

---

This DSN remains Experimental until the sequential evidence satisfies its
promotion gate. Mechanics or a visually compelling demo alone cannot promote
it.
