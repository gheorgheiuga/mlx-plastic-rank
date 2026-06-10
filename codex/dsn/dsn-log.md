# DSN Log

Track open and closed Decision Support Notes here for quick discovery. Link each entry to the corresponding markdown file under `codex/dsn/`.

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

- **DSN-20260609-03** — Test low-spectrum key-projection adaptation *(Accepted as local experimental direction; broader validation experimental)*
  - Summary: Records the spectral-notch probe finding that trained Gemma fault-code `k_proj` adapters show elevated low-spectrum energy, and validates a same-budget spectral-key candidate that beats the current hetero map on held-out answer-token PPL/accuracy while matching generation overlap.
  - Link: `codex/dsn/dsn-20260609-low-spectrum-key-adaptation.md`

- **DSN-20260609-04** — Add artifact-backed domain pack proof reports *(Accepted as productization gate; broader validation experimental)*
  - Summary: Adds `packs proof` to turn pack, eval, generation, and rank-ledger artifacts into a pass/fail DLC-style domain improvement report; local fault-code proof reports pass, including the full-split 2,700/300 Gemma 4 IT bakeoff where the learned hetero rank map is the best size/quality tradeoff.
  - Link: `codex/dsn/dsn-20260609-domain-pack-proof.md`

- **DSN-20260610-01** — Add reproducible pack bakeoff workflow *(Accepted as orchestration; replication evidence pending)*
  - Summary: Adds `packs bakeoff` specs for train/eval/rank-ledger/proof runs, commits compact fault-code evidence snapshots, and selects Apache-2.0 Text-to-SQL data as the next large replication surface.
  - Link: `codex/dsn/dsn-20260610-pack-bakeoff-workflow.md`
