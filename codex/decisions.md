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
- **Status:** Accepted as orchestration workflow; replication evidence pending
- **DSN:** `codex/dsn/dsn-20260610-pack-bakeoff-workflow.md`
- **Context:** The proof path was useful but still required hand-running multiple commands and reading ignored local artifacts.
- **Decision:** Add `packs bakeoff --spec` to orchestrate create/eval/rank-ledger/proof phases from JSON specs, commit compact evidence snapshots under `codex/evidence/`, and use Apache-2.0 `gretelai/synthetic_text_to_sql` as the next large replication dataset.
- **Consequences:** The repo can now reproduce the fault-code comparison and launch a second-dataset bakeoff from reviewable specs. This accepts the workflow and evidence hygiene, not a broader Pop Rank theorem or cross-domain quality claim.
