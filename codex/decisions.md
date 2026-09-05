# Architectural Decisions

## Current scope — 2026-09-05

ADR-0015 supersedes older default-demo and expansion priorities: maintain the
low-rank/pack substrate, park unvalidated controllers and larger pilots, and
prepare only the ADR-0016 baseline diagnostic. Earlier entries below are a
decision history, not a queue of currently endorsed experiments. Historical
quality claims retain the initialization/provenance qualifications in ADR-0012.

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
- **Status:** Parked Experimental under ADR-0015; historical local observations retained
- **DSN:** `codex/dsn/dsn-20260609-low-spectrum-key-adaptation.md`
- **Context:** Pop polynomial probes showed generic matrix-polynomial identities are valid rank accounting but not direct rank selectors. The stronger local signal was that trained Gemma fault-code `k_proj` adapters, especially full-attention key projections, carried elevated low-spectrum energy while `q_proj` and `v_proj` stayed near baseline.
- **Decision:** Use spectral notch diagnostics to build same-budget rank-map candidates that promote high-lift full-attention `k_proj` adapters and compensate by reducing lower-signal `q_proj` ranks.
- **Consequences:** The first candidate, `spectral-key-candidate`, recorded better held-out answer-token PPL/accuracy at essentially equal size and matched the generation overlap check. This is a historical local observation, not causal validation: later review found unmatched factor initialization and incomplete artifact provenance. Further rank-map discovery is parked under ADR-0015; any new comparison requires matched controls and training-seed replication.

## ADR-0008 — Domain Pack Proof Reports
- **Date:** 2026-06-09
- **Status:** Accepted as productization gate; broader validation remains experimental
- **DSN:** `codex/dsn/dsn-20260609-domain-pack-proof.md`
- **Context:** The DLC-style pack claim was supported by separate training, eval, generation, and ledger artifacts, but the repo lacked a machine-checkable pass/fail report tying those artifacts together.
- **Decision:** Add `packs proof` to audit artifact-backed domain improvement claims: pack exists, training/eval data exist, base and base+pack eval rows match, attach changes logits, held-out metrics improve, and optional generation/ledger gates pass.
- **Historical evidence:** A Gemma 4 IT fault-code bakeoff on 2,700 train / 300 held-out rows recorded fixed-r32 PPL 5.5622, accuracy 0.6802 and 54.16 MB; the hetero map recorded PPL 5.9406, accuracy 0.6748 and 23.73 MB. These describe the tested artifacts and their original report passes, not a general quality ceiling or a causal allocation result.
- **Consequences:** Pack reports provide a machine-checkable evaluation path. September's stricter content binding does not retroactively validate old reports; matched initialization and fresh provenance are required for new allocation comparisons. The tooling remains active, while broader rank-map studies are parked under ADR-0015.
- **Validation correction (2026-09-05):** Those passes used the original report gate, which admitted missing metrics and did not bind data, checkpoint, tokenizer, or pack contents. ADR-0012 replaces that gate. Historical metrics remain preserved; their old `passed` fields do not satisfy the current provenance requirements.

## ADR-0009 — Reproducible Pack Bakeoff Workflow
- **Date:** 2026-06-10
- **Status:** Accepted as orchestration workflow; local replication and reciprocal transfer passed, random/shuffled Text-to-SQL controls pending
- **DSN:** `codex/dsn/dsn-20260610-pack-bakeoff-workflow.md`
- **Context:** The proof path was useful but still required hand-running multiple commands and reading ignored local artifacts.
- **Decision:** Add `packs bakeoff --spec` to orchestrate create/eval/rank-ledger/proof phases from JSON specs, commit compact evidence snapshots under `codex/evidence/`, and use Apache-2.0 `gretelai/synthetic_text_to_sql` as the next large replication dataset.
- **Implementation update (2026-07-09):** Bakeoff specs can now generate seeded `random_same_budget` and `shuffled_discovered` controls in resumable rank-map preflight phases, preserve their provenance in summaries, and require the tradeoff candidate to beat included controls before promotion. Fixed batch schedules, separate training/dropout seeds, per-example metrics, and paired bootstrap intervals make those comparisons auditable.
- **Evidence update:** The original 10,000-row Text-to-SQL replication completed: its fresh heterogeneous map reached PPL 1.7600 at 23.92 MB, retained 94.05% of the fixed-r32 gain at 44.16% of its size, and passed the original promotion gate. The newly added random/shuffled candidates have not yet run under a paired schedule.
- **Consequences:** The repo now has quality-positive local results on fault codes and Text-to-SQL. This accepts the workflow and evidence hygiene, not a broader Pop Rank theorem or domain-specific quality claim.
- **Validation correction (2026-09-05):** File-existence resume checks did not establish that outputs matched the current inputs. ADR-0012 adds content-bound phase receipts and prevents promotion from unverified artifacts. The historical results above are preserved as measurements, not retrospectively certified by the new checks.

## ADR-0010 — Paired Rank-Placement Falsification Screen
- **Date:** 2026-07-09
- **Status:** Accepted as diagnostic single-seed evidence; confirmatory replication pending
- **Specs:** `codex/bakeoffs/fault_codes_paired_control_screen_seed42.json`, `codex/bakeoffs/text_to_sql_paired_transfer_screen_seed42.json`
- **Evidence:** `codex/evidence/fault_codes_paired_control_screen_seed42.json`, `codex/evidence/text_to_sql_paired_transfer_screen_seed42.json`
- **Context:** The earlier heterogeneous-map win did not isolate rank placement from initialization, minibatch order, dropout, rank histogram, simple projection rules, or transferable cross-domain structure.
- **Decision:** Treat rank-placement benefit as promotable only when a fresh discovered-map run shares its stochastic schedule with controls, beats at least four of five same-budget random maps, beats all declared structured controls, and each paired 95% bootstrap interval excludes zero.
- **Result:** The fault-code map passed: PPL 5.8532 at 23.73 MB, 97.05% of the contextual fixed-r32 gain at 43.82% of its size. It beat 5/5 random maps, an exact-budget shuffled map, `q16/k16/v8`, and a normalized Text-to-SQL transplant. The cross-domain margin was only 1.12% (PPL difference -0.0662, 95% CI [-0.0849, -0.0485]).
- **Reciprocal evidence update (2026-07-12):** On 300 Text-to-SQL examples, a fresh native map reached PPL 1.7172 versus 1.7429 for a byte-matched fault-code transplant, a 1.47% relative advantage (paired difference -0.0256, 95% CI [-0.0300, -0.0217]). It retained 97.50% of the contextual fixed-r32 gain at 44.16% of its size and passed the paired transfer gate.
- **Validation correction (2026-09-05):** The shared seed did not initialize shared factor rows identically when requested ranks differed. The observed native-configuration wins remain recorded, but initialization is an unresolved confound. The comparisons do not isolate rank placement or establish domain specificity. New comparisons must explicitly use `component-v1` initialization with shared schedules and untouched seeds. These results do not validate Pop's matrix-polynomial theorem.

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

## ADR-0012 — Correctness and Experiment Integrity
- **Date:** 2026-09-05
- **Status:** Implemented engineering safeguards; research validation remains Experimental
- **DSN:** `codex/dsn/dsn-20260905-correctness-and-experiment-integrity.md`
- **Context:** Review reproduced correctness defects in the plasticity path, weak proof validation, stale bakeoff reuse, and a rank-dependent initialization confound in older comparisons.
- **Decision:** Correct batch handling, sleeper identities, signed pruning, base freezing and alpha-zero handling. Require content-bound proof evidence and complete phase receipts. Provide explicit `component-v1` initialization for matched shared factors while retaining `legacy` as the historical default. Use the locked uv environment for CI and local checks.
- **Evidence:** Regression tests and a real tiny-model training-to-proof workflow pass locally. Type checking is clean. This is implementation evidence only; no new Gemma run or research promotion was performed.
- **Consequences:** Older reports and packs remain readable and preserved, but missing provenance or incomplete continuation history cannot establish a current proof. Unverified outputs cannot be resumed or promoted; prefer new output names to preserve them. September findings qualify the stronger interpretations in ADR-0008 through ADR-0010. ADR-0011 and its failed promotion gates remain unchanged.
- **Next validation:** The focused memory/structure work is recorded in ADR-0013. Require matched initialization, untouched seeds, complete finite controls and the declared held-out quality gate before promoting a scientific claim.

## ADR-0013 — Compact SVD and Shared Plasticity Operations
- **Date:** 2026-09-05
- **Status:** Implemented engineering changes; broader validation remains Experimental
- **DSN:** `codex/dsn/dsn-20260905-compact-svd.md`
- **Context:** Randomized SVD allocated full square singular-vector matrices, compression could silently choose full SVD on CPU/fallback, and sleeper storage was duplicated across pruning policies.
- **Decision:** Use reduced QR on both sides with a sketch-width SVD core and stabilized power iterations. Keep factorization behind the existing public functions in a dedicated module, preserve randomized CPU/fallback behavior, and centralize component parking without changing the pruning policies.
- **Evidence:** Eighteen isolated synthetic runs compare the exact committed baseline against source-identified updated code. Median process peaks fell by 50.24%, 49.63% and 29.01% on tall, wide and square cases; relative reconstruction errors differ by at most 2.39e-7. Regression tests cover numerical behavior, allocation requests, device paths, exported factors and sleep/wake behavior. Evidence: `codex/evidence/svd_workspace_20260905.json`.
- **Consequences:** The low-rank path avoids original-dimension square factors when the sketch is small. Near-full rank, input conversion, dense reconstruction and whole-checkpoint residency remain material costs. Old imports remain supported; numerical/RNG results can change. These measurements do not establish a whole-model device fit or reverse the failed research gates.
- **Next validation:** Any model-scale memory claim needs a complete workflow measurement. A new scientific experiment must independently meet ADR-0011/ADR-0012 and begin with a declared protocol; no large-model run was performed here.

## ADR-0014 — Gradient-Agreement Admission Test
- **Date:** 2026-09-05
- **Status:** Experimental; implementation verified, development validity failed
- **DSN:** `codex/dsn/dsn-20260905-gradient-agreement-admission.md`
- **Context:** The demoted horizon-3 controller did not beat exact one-step, its comparator matrix was incomplete, and it omitted the required future-aware fixed split. September's engineering repairs do not fill those scientific gaps.
- **Proposed decision:** Test cross-batch prospective-gradient agreement with removal-cost donors, without candidate training rollouts. Separate identical-A-checkpoint selector comparisons from the whole-strategy fixed-split comparison. Include random, exact one-step, plain gradient energy, wrong-task, static, site-oracle and sufficient-capacity controls under declared initialization, numerical and state-audit policies.
- **Evidence:** All 45 development runs on seeds 31–35 were finite, paired and mechanically valid. Common-A and fixed-split A-readiness means were 0.6062 and 0.4823; joint-capacity A/B end scores were 0.6682/0.5811, below the declared 0.8 gates. Final seed-0 trajectories and events repeated exactly. The compact record is `codex/evidence/gradient_agreement_development_seed31_35.json`; interpretation and commands are at `codex/research/gradient-agreement/development-results.md`. No evidence seed was used.
- **Gate:** Validity-only development on seeds 0 and 31–35 precedes source/content freeze. A complete 20-seed, nine-condition evidence matrix must meet all predeclared quality, localization, readiness and solvability gates, with no exclusions. Failed or incomplete controls prevent admission.
- **Consequences:** ADR-0011 and all failed historical gates remain unchanged. A successful admission would permit only a separate full return-A protocol design. Do not infer a quality advantage, whole-model memory benefit, cue-triggered wake, erasure or Pop-theorem result from this proposal.
- **Development decision:** Park before evidence. The new controller and failure-handling infrastructure are implemented, but the fixture cannot support admission under this training/data schedule. Keep the reserved seeds and thresholds unchanged. Before another adaptive matrix, separately specify a baseline diagnosis using the stored development data to distinguish reference learnability from factorized optimization and held-out generalization. Evidence enablement, freeze and paired inference remain unavailable.

## ADR-0015 — Consolidate the prototype around verified mechanics
- **Date:** 2026-09-05
- **Status:** Accepted as an engineering/scope decision; research quality remains Experimental
- **DSN:** `codex/dsn/dsn-20260905-prototype-consolidation.md`
- **Context:** Duplicate demos, unnecessary default dependencies and unvalidated research tracks obscured the useful low-rank/pack substrate. The default demo's arbitrary mean objective and rapid heuristic rank growth did not demonstrate conserved capacity.
- **Decision:** Keep tested factor/pack lifecycles, compact SVD, accounting and content-bound evidence. Replace the demo with a bounded prune/restore cycle; retire the banner and duplicate plasticity, generation and routing entry points. Remove unused SymPy, move dataset extraction to an optional extra, and retain explicit legacy imports without advertising parked APIs as the core.
- **Consequences:** Current documentation uses one demo and one pack CLI. Failed controllers, theorem selectors, large-model expansion and the external vault leave the active work plan while their tests/source/evidence remain. Historical priorities are superseded only in scope; failed thresholds and raw artifacts are preserved. The only proposed research next step is ADR-0016.
- **Test organization:** The default suite now follows core/pack/utility scope (259 cases); 71 parked regressions run explicitly. Two weak smoke checks are folded into stronger tests, and repetitive parser setup is consolidated. Research tests remain required when shared-mechanic changes affect them.

## ADR-0016 — Diagnose the baseline before another adaptive comparison
- **Date:** 2026-09-05
- **Status:** Proposed; not implemented or run
- **DSN:** `codex/dsn/dsn-20260905-baseline-validity-diagnostic.md`
- **Context:** Finite development runs still missed A-readiness and joint-capacity gates. Current evidence does not distinguish data coverage, conditioning, representation and optimization explanations.
- **Proposed decision:** Fit a dense routed reference on stored development training arrays only, measure the same fitted checkpoint on held-out rows, and include a shifted-target negative control. Freeze solver/measurement criteria before execution; use exact input identities, all five development seeds and a five-minute bound.
- **Consequences:** The result selects one follow-up diagnosis or fixture repair. It is not a fair rank-budget competitor and cannot admit a controller. Reserved evidence seeds, threshold changes, controller sweeps, return-A and larger-model work remain gated.
