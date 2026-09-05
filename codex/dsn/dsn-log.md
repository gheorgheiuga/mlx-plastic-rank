# DSN Log

Track open and closed Decision Support Notes here for quick discovery. Link each entry to the corresponding markdown file under `codex/dsn/`.

Current scope follows DSN-20260905-04; older directions below are historical
unless explicitly resumed. Their numerical observations do not override the
September initialization/provenance qualifications or failed research gates.

- **DSN-20260905-05** — Diagnose baseline validity before controller work *(Proposed; not implemented or run)*
  - Summary: One bounded dense routed least-squares diagnostic on saved seeds 31–35, with train/held-out isolation and a broken-pairing control. Its result chooses the next fixture or optimization question; it does not admit a controller.
  - Link: `codex/dsn/dsn-20260905-baseline-validity-diagnostic.md`

- **DSN-20260905-04** — Consolidate around verified mechanics *(Accepted as engineering/scope decision; scientific benefit remains Experimental)*
  - Summary: One bounded lifecycle demo and one pack CLI; retire duplicate entry points, remove unused symbolic dependencies and make dataset tooling optional. Preserve failed evidence and park expansion. Prepare the baseline diagnostic before more controller work.
  - Test follow-through: Separate 259 default checks from 71 explicit research regressions; merge two weak smoke checks and consolidate parser scenarios. This changes maintenance scope, not research evidence.
  - Link: `codex/dsn/dsn-20260905-prototype-consolidation.md`

- **DSN-20260905-03** — Gradient-agreement controller admission test *(Experimental; implementation verified, development validity failed)*
  - Summary: Implements the controller and nine-condition benchmark with matched initialization, uniform clipping, full factor audits and source/output receipts. All 45 development runs were finite and mechanically valid, but A-readiness and joint-capacity scores missed their gates. Seed-0 results repeated exactly. Park before evidence; reserved seeds, full return-A and large-model work remain untouched.
  - Link: `codex/dsn/dsn-20260905-gradient-agreement-admission.md`

- **DSN-20260905-02** — Compact SVD workspace and focused structural cleanup *(Experimental; numerical/allocation behavior verified, no model-quality claim)*
  - Summary: Uses reduced QR and a small SVD core, preserves randomized CPU/fallback behavior, separates factorization from layer state, and shares sleeper handling. Eighteen synthetic runs measured 29–50% lower process peaks; whole-model fit and research gates remain unproven.
  - Link: `codex/dsn/dsn-20260905-compact-svd.md`

- **DSN-20260905-01** — Correctness and content-bound experiment evidence *(Experimental; engineering behavior verified, confirmatory quality pending)*
  - Summary: Corrects plasticity defects, adds strict proof identities and resume receipts, and introduces matched component initialization for new studies. Earlier report passes are historical; the rank-placement screens retain an initialization confound. No new research result is promoted.
  - Link: `codex/dsn/dsn-20260905-correctness-and-experiment-integrity.md`

- **DSN-20250912-01** — Python 3.13 default toolchain *(Accepted)*
  - Summary: Aligns local dev, CI, and MLX compatibility; future upgrades require DSN + ADR update.
  - Link: `codex/dsn/dsn-20250912-python313.md`

- **DSN-20250922-02** — Maintain LoRA guardrails for rank and pack size *(Proposed)*

- **DSN-20260608-01** — Target Gemma 4 12B mxfp8 for unified any-to-any packs *(Accepted; quality signal on fault-code pack, broader validation experimental)*
  - Summary: Uses mxfp8 as the default Gemma 4 runtime base, keeps bf16 for reference, makes `mlx-vlm`/`mlx-audio` the macOS modality stack, records IndustryBench as mechanics-positive but quality-negative, and selects `fault-codes-gemma4-it-answer-r32-300` as the first useful industrial pack candidate.
  - Link: `codex/dsn/dsn-20260608-gemma4-12b.md`

- **DSN-20260609-01** — Measure pack rank algebra before claiming theorem advantage *(Accepted; instrumentation only)*
  - Summary: Adds `packs rank-ledger` to measure effective rank, rank slack, composition rank, row/column overlap, and rank savings before claiming Pop-theorem rank selection benefits.
  - Link: `codex/dsn/dsn-20260609-pop-rank-ledger.md`

- **DSN-20260609-02** — Implement dynamic Pop Rank with gated active ranks *(Accepted; mechanics verified, quality experimental)*
  - Summary: Makes `--rank` a training ceiling via active-rank gates, grows/shrinks adapters by learned rank signal, and exports only active columns.
  - Link: `codex/dsn/dsn-20260609-dynamic-pop-rank.md`

- **DSN-20260609-03** — Test low-spectrum key-projection adaptation *(Parked Experimental; historical observations retained)*
  - Summary: Records local spectral and pack metrics. Unmatched initialization and incomplete provenance prevent causal placement conclusions. Further selector studies are parked under DSN-20260905-04 pending a separately declared matched comparison.
  - Link: `codex/dsn/dsn-20260609-low-spectrum-key-adaptation.md`

- **DSN-20260609-04** — Add artifact-backed domain pack proof reports *(Accepted as productization gate; broader validation experimental)*
  - Summary: Adds `packs proof` to turn pack, eval, generation, and rank-ledger artifacts into a pass/fail DLC-style domain improvement report; local fault-code proof reports pass, including the full-split 2,700/300 Gemma 4 IT bakeoff where the learned hetero rank map is the best size/quality tradeoff.
  - Link: `codex/dsn/dsn-20260609-domain-pack-proof.md`
  - September qualification: Original passes did not establish current artifact provenance; see DSN-20260905-01.

- **DSN-20260610-01** — Add reproducible pack bakeoff workflow *(Accepted as orchestration; two-domain local replication passed, broader validation experimental)*
  - Summary: Adds `packs bakeoff` specs for train/eval/rank-ledger/proof runs, commits compact evidence snapshots, reproduces the quality/size tradeoff on Apache-2.0 Text-to-SQL data, and records a diagnostic reciprocal native-versus-transplant screen.
  - Link: `codex/dsn/dsn-20260610-pack-bakeoff-workflow.md`
  - September qualification: Resume now requires content-bound receipts, and rank comparisons require matched factor initialization for causal interpretation; see DSN-20260905-01.

- **DSN-20260713-01** — Build a controlled spectral forgetting vault *(Parked Experimental; retained and runnable)*
  - Summary: Preserves the controlled-vault prototype and its explicit deletion boundary, but parks active development until a learned MLX capacity-migration result passes the learned-model promotion gate and needs an external-deletion comparison, or a new project-priority decision resumes the track.
  - Link: `codex/dsn/dsn-20260713-forgetting-machine.md`

- **DSN-20260713-02** — Test conserved low-rank capacity migration *(Experimental; reference mechanics passed, first learned-MLX promotion gate failed)*
  - Summary: Restores the founding Pop Rank question and implements fixed-budget `A -> B -> A` reference and learned-MLX benchmarks. The first learned run failed promotion, and a fully finite first-transfer calibration demoted one-step ordering. A new two-transfer horizon-3 controller beat static/random/wrong-task controls and localized B rank, but failed admission because it did not separate from exact one-step and that comparator had a reproducible non-finite seed. Horizon-3 is demoted, shadow-rollout tuning is parked, the full return-A V2 remains blocked, and London remains blocked.
  - Link: `codex/dsn/dsn-20260713-capacity-migration.md`
