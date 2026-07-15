# Architectural Decisions

## ADR-0001 — Rebrand to `mlx-plastic-rank`
- **Date:** 2025-09-12
- **Status:** Accepted
- **Context:** The original name `poprank` no longer reflected the MLX focus or packaging direction.
- **Decision:** Rename the project in `pyproject.toml`, CLI banner, and tests. Module moves under `src/mlx_plastic_rank/` would follow incrementally.
- **Consequences:** Update documentation, packaging metadata, and CI references; no functional changes expected.

## ADR-0002 — Create `codex/` for Maintainer Docs
- **Date:** 2025-09-12
- **Status:** Accepted
- **Context:** We needed a lightweight home for runbooks, ADRs, DSNs, and research notes without overloading contributor guides.
- **Decision:** Introduce `codex/runbook.md` and `codex/decisions.md`, with a Research Inbox for citations. Non-code decisions live here instead of in standard contributor docs.
- **Consequences:** Maintainers have a durable space for plans and experiments; contributors remain focused on `AGENTS.md`/`CONTRIBUTING.md`.

## ADR-0003 — LoRA Skill Packs for GPT-style Models
- **Date:** 2025-09-20
- **Status:** Accepted
- **Context:** Global SVD compression hurt perplexity, pushing the team toward reversible, modular adapters that keep base checkpoints intact.
- **Decision:** Wrap fused attention slices with `LoRAFusedLinear`, exporting packs as `.lora.{A,B}` (fp16) plus `alpha` (fp32). Enforce size ≤10 MB, alpha = 2r, and one active pack at a time through the `packs` CLI.
- **Implementation update (2026-07-09):** Pack application now rejects incompatible base-model identifiers unless checkpoint hashes establish equivalence, verifies scalar fp32 alpha tensors against metadata, and stages a replacement fully before detaching the active pack.
- **Consequences:** Packs stay small (≈O(r·hidden·layers)), adapters swap at runtime, and evaluation tooling now reports PPL deltas, load time, and memory metrics. Path-B delta exports will reuse the same schema.

## ADR-0004 — Gemma 4 Industrial Pack Pilot
- **Date:** 2026-06-08
- **Status:** Accepted as implementation direction; evidence remains experimental
- **DSN:** `codex/dsn/dsn-20260608-gemma4-12b.md`
- **Context:** The project needed a larger multimodal-capable base and industrial-domain benchmark path after earlier text-only pack experiments showed mechanics without strong domain lift.
- **Decision:** Target Gemma 4 12B mxfp8 as the default pilot base for unified any-to-any packs, keep bf16 as reference, and use the fault-code/IndustryBench tooling as the first industrial evaluation surface.
- **Consequences:** The codebase now carries Gemma 4 smoke, extraction, and dataset helpers. The decision does not prove useful domain adaptation yet; real 12B pack training and quality lift remain validation gates.

## ADR-0005 — Rank Algebra Ledger Before Pop Rank Claims
- **Date:** 2026-06-09
- **Status:** Accepted as instrumentation; not accepted as theorem validation
- **DSN:** `codex/dsn/dsn-20260609-pop-rank-ledger.md`
- **Context:** The Pop Rank premise needed a way to measure rank behavior before making claims about theorem-guided advantage.
- **Decision:** Add `packs rank-ledger` to report effective rank, rank slack, composition rank, row/column overlap, and rank savings for LoRA pack operators.
- **Consequences:** The ledger gives a reproducible measurement surface for adapter rank algebra. It is not a proof that Pop Rank improves downstream quality; theorem or quality claims still require separate validation.

## ADR-0006 — Dynamic Pop Rank Gates
- **Date:** 2026-06-09
- **Status:** Accepted as experimental implementation
- **DSN:** `codex/dsn/dsn-20260609-dynamic-pop-rank.md`
- **Context:** Static rank ceilings could not test whether adapters should grow or shrink rank during training.
- **Decision:** Treat `--rank` as a ceiling, add active-rank gates, allow grow/shrink behavior from learned rank signals, and export only active columns.
- **Consequences:** Dynamic rank mechanics can now be tested behind CLI flags. The current signal is adapter-level utility from learned factor norms, not a validation-loss oracle; quality-per-MB benefit still needs benchmark evidence before stronger claims.

## ADR-0007 — Low-Spectrum Key Projection Rank Map
- **Date:** 2026-06-09
- **Status:** Accepted as local experimental direction; broader validation remains experimental
- **DSN:** `codex/dsn/dsn-20260609-low-spectrum-key-adaptation.md`
- **Context:** Pop polynomial probes showed generic matrix-polynomial identities are valid rank accounting but not direct rank selectors. The stronger local signal was that trained Gemma fault-code `k_proj` adapters, especially full-attention key projections, carried elevated low-spectrum energy while `q_proj` and `v_proj` stayed near baseline.
- **Decision:** Use spectral notch diagnostics to build same-budget rank-map candidates that promote high-lift full-attention `k_proj` adapters and compensate by reducing lower-signal `q_proj` ranks.
- **Consequences:** The first candidate, `spectral-key-candidate`, beat the current hetero map on held-out answer-token PPL/accuracy at essentially equal size and matched its generation overlap check. This validates the local rank-map direction for the fault-code pilot; it does not prove a general Pop Rank theorem advantage and should be repeated across seed/dataset/base before broader claims.

## ADR-0008 — Domain Pack Proof Reports
- **Date:** 2026-06-09
- **Status:** Accepted as productization gate; broader validation remains experimental
- **DSN:** `codex/dsn/dsn-20260609-domain-pack-proof.md`
- **Context:** The DLC-style pack claim was supported by separate training, eval, generation, and ledger artifacts, but the repo lacked a machine-checkable pass/fail report tying those artifacts together.
- **Decision:** Add `packs proof` to audit artifact-backed domain improvement claims: pack exists, training/eval data exist, base and base+pack eval rows match, attach changes logits, held-out metrics improve, and optional generation/ledger gates pass.
- **Evidence update:** A full-split Gemma 4 IT fault-code bakeoff on 2,700 train / 300 held-out rows shows the fixed r32 pack is the local quality ceiling (PPL 5.5622, token accuracy 0.6802, 54.16 MB), while the learned hetero rank-map pack is the best size/quality tradeoff (PPL 5.9406, token accuracy 0.6748, 23.73 MB) and passes `packs proof`.
- **Consequences:** The repo can now prove the local industrial fault-code base+pack workflow from artifacts. Newly created packs record training provenance in metadata. This validates a product-style proof path for one domain and a local size/quality tradeoff result, not a universal Pop Rank quality claim.

## ADR-0009 — Reproducible Pack Bakeoff Workflow
- **Date:** 2026-06-10
- **Status:** Accepted as orchestration workflow; local replication and reciprocal transfer passed, random/shuffled Text-to-SQL controls pending
- **DSN:** `codex/dsn/dsn-20260610-pack-bakeoff-workflow.md`
- **Context:** The proof path was useful but still required hand-running multiple commands and reading ignored local artifacts.
- **Decision:** Add `packs bakeoff --spec` to orchestrate create/eval/rank-ledger/proof phases from JSON specs, commit compact evidence snapshots under `codex/evidence/`, and use Apache-2.0 `gretelai/synthetic_text_to_sql` as the next large replication dataset.
- **Implementation update (2026-07-09):** Bakeoff specs can now generate seeded `random_same_budget` and `shuffled_discovered` controls in resumable rank-map preflight phases, preserve their provenance in summaries, and require the tradeoff candidate to beat included controls before promotion. Fixed batch schedules, separate training/dropout seeds, per-example metrics, and paired bootstrap intervals make those comparisons auditable.
- **Evidence update:** The original 10,000-row Text-to-SQL replication completed: its fresh heterogeneous map reached PPL 1.7600 at 23.92 MB, retained 94.05% of the fixed-r32 gain at 44.16% of its size, and passed the original promotion gate. The newly added random/shuffled candidates have not yet run under a paired schedule.
- **Consequences:** The repo now has quality-positive local results on fault codes and Text-to-SQL. This accepts the workflow and evidence hygiene, not a broader Pop Rank theorem or domain-specific quality claim.

## ADR-0010 — Paired Rank-Placement Falsification Screen
- **Date:** 2026-07-09
- **Status:** Accepted as diagnostic single-seed evidence; confirmatory replication pending
- **Specs:** `codex/bakeoffs/fault_codes_paired_control_screen_seed42.json`, `codex/bakeoffs/text_to_sql_paired_transfer_screen_seed42.json`
- **Evidence:** `codex/evidence/fault_codes_paired_control_screen_seed42.json`, `codex/evidence/text_to_sql_paired_transfer_screen_seed42.json`
- **Context:** The earlier heterogeneous-map win did not isolate rank placement from initialization, minibatch order, dropout, rank histogram, simple projection rules, or transferable cross-domain structure.
- **Decision:** Treat rank-placement benefit as promotable only when a fresh discovered-map run shares its stochastic schedule with controls, beats at least four of five same-budget random maps, beats all declared structured controls, and each paired 95% bootstrap interval excludes zero.
- **Result:** The fault-code map passed: PPL 5.8532 at 23.73 MB, 97.05% of the contextual fixed-r32 gain at 43.82% of its size. It beat 5/5 random maps, an exact-budget shuffled map, `q16/k16/v8`, and a normalized Text-to-SQL transplant. The cross-domain margin was only 1.12% (PPL difference -0.0662, 95% CI [-0.0849, -0.0485]).
- **Reciprocal evidence update (2026-07-12):** On 300 Text-to-SQL examples, a fresh native map reached PPL 1.7172 versus 1.7429 for a byte-matched fault-code transplant, a 1.47% relative advantage (paired difference -0.0256, 95% CI [-0.0300, -0.0217]). It retained 97.50% of the contextual fixed-r32 gain at 44.16% of its size and passed the paired transfer gate.
- **Consequences:** Same-budget rank placement is a real local signal. Both domain-native maps beat reciprocal transplants, but by only 1.12% and 1.47%, so domain specificity is consistent yet weak and single-seed. The next confirmatory test remains a multi-seed domain-by-map transfer matrix. These results do not validate Pop's matrix-polynomial theorem.

## ADR-0011 — Restore Conserved Capacity Migration as the Pop Rank Thesis
- **Date:** 2026-07-13
- **Status:** Experimental research direction; partial sequential signal observed, promotion failed
- **Evidence Status:** Ten-seed counterfactual reference mechanics passed; the first frozen dense learned-MLX promotion gate failed despite a positive guided-versus-static result, a conditional nine-pair guided-versus-random AUC result, and a positive event-window signal
- **DSN:** `codex/dsn/dsn-20260713-capacity-migration.md`
- **Context:** Pop Rank began as a reversible neuroplasticity experiment in which capacity could grow, shrink, sleep, and wake. Later static rank-map and quality-per-byte work produced useful evidence but displaced the original temporal question without testing it.
- **Decision:** Make capacity migration the canonical research thesis. The decisive experiment must keep a global active-rank budget fixed, pair every growth with a donor shrink, separately count dormant learned factors, and compare utility-guided migration with static, future-aware fixed-split, random, oracle, extra-capacity, vault, recycle, and never-trained controls in one sequential model run. ADR-0005 and ADR-0006 provide measurement and actuation; ADR-0007 through ADR-0010 remain supporting rank-placement evidence rather than the project's endpoint. The controlled spectral Forgetting Machine remains preserved but parked because external-record deletion does not test model-capacity migration.
- **Promotion Gate:** Across at least 10 seeds, utility-guided transfer must conserve the declared budget exactly, move rank toward the new task more reliably than random, beat static, future-aware fixed-split, and random controls on paired acquisition metrics with a 95% interval excluding zero, and show that strictly recycled A capacity improves B learning. Immediate cue-triggered wake and gradient-based relearning must be reported separately.
- **Kill Criteria:** Demote or revise the mechanism if it exceeds the budget, hides learned state outside the dormant ledger, matches random transfer, moves rank without a performance benefit, improves only when extra capacity is added, or describes reversible gating as erasure.
- **Reference result (2026-07-13):** The deterministic four-site teacher/student benchmark passed all declared counterfactual reference-mechanics gates across ten unique fixture seeds. Vault and recycle reached mean B AUC 0.900 versus 0.000 static, 0.508 future-aware fixed split, and 0.362 same-timing shuffled-recipient transfer while conserving active rank at every checkpoint. They use idealized component-level counterfactual gradient demand, a routing upper bound rather than learned discovery. Vault restored A after an A-derived cue and before a parameter update from separately counted resident state; recycle did not.
- **Learned result (2026-07-13):** A dense-router MLX bridge developed on seed 0 and frozen for seeds 1–10 failed its full promotion gate. Guided recycle beat static on B AUC by +0.2384 (95% CI [+0.1822, +0.2974]); across the nine finite random pairs it had a conditional +0.1268 advantage ([+0.0631, +0.1802]), but the localization-versus-random gate remained incomplete and failed because the tenth random run was non-finite. The first loss-guided transfer during the B phase preceded a +0.1431 matched event-window advantage ([+0.0807, +0.2052]). Guided did not separate from future-aware fixed split (+0.0227, [-0.0346, +0.0816]); an extra-capacity run also went non-finite; and joint-sufficient A/B means narrowly missed 0.8. The reported A-return value was post-supervised-loss-probe/pre-update, not cue-triggered wake. `min_rank=1` guaranteed at least one A-site component, but some seeds retained more and dense routes could preserve A elsewhere. Strict-recycle cleanliness and the vault's mean dormant value of 4.3 are provisional A-column lower-bound measurements because B-only learned state was not audited. Evidence: `codex/evidence/capacity_migration_learned_dense_seed1_10.{json,md}`.
- **Selector calibration (2026-07-14):** A frozen exhaustive branch test at the first B allocation opportunity ran every legal one-in/one-out transfer from the same A-trained checkpoint on untouched seeds 11–20. All branches were finite, checkpoint-paired, rank-conserving, and strict-recycle clean. Predicted-best beat static by `+0.1540` (95% CI `[+0.0559, +0.2972]`), prediction-independent random by `+0.1305` (`[+0.0369, +0.2695]`), and an A-loss-selected control by `+0.1632` (`[+0.0684, +0.3017]`). The full directional gate failed: mean candidate-level Spearman was `0.2457` (`[-0.0255, +0.5159]`) and predicted-best did not beat predicted-worst conclusively (`+0.0617`, `[-0.0313, +0.1514]`). Evidence: `codex/evidence/loss_lookahead_calibration_seed11_20.{json,md}`.
- **Multi-batch controller result (2026-07-14):** A separately frozen horizon-3 protocol made two fixed-cadence strict-recycle transfers over 24 B updates on untouched seeds 21–30. It beat static on B AUC by `+0.1815` (95% CI `[+0.0771, +0.3130]`), fixed random by `+0.1814` (`[+0.0360, +0.3232]`), and an A-task horizon-3 control by `+0.1872` (`[+0.0645, +0.3287]`); final B-site coverage also beat random by `+0.1667` (`[+0.0333, +0.3000]`). The admission gate failed because horizon-3 did not separate from exact one-step over nine finite pairs (`+0.0809`, `[-0.0555, +0.2866]`) and exact-one-step seed 26 produced reproducible non-finite gradients. Evidence: `codex/evidence/multibatch_controller_v2_seed21_30.{json,md}`.
- **Consequences:** Keep ADR-0011 Experimental. Demote horizon-3 as the V2 controller and park further exhaustive shadow-rollout tuning. The narrower controlled claim is that supervised task-conditioned direction can beat random/static/wrong-task controls across two transfers in this fixture; no added value from the longer horizon was identified. Do not build the full return-A V2 or a large London/Gemma experiment. Re-entry requires a qualitatively new controller that produces a complete finite matrix and beats exact one-step, fixed random, and future-aware fixed split on untouched seeds. Existing quality-per-byte results remain valid local evidence but no longer define project success.
