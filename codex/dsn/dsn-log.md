# DSN Log

Track open and closed Decision Support Notes here for quick discovery. Link each entry to the corresponding markdown file under `codex/dsn/`.

- **DSN-20250912-01** — Python 3.13 default toolchain *(Accepted)*
  - Summary: Aligns local dev, CI, and MLX compatibility; future upgrades require DSN + ADR update.
  - Link: `codex/dsn/dsn-20250912-python313.md`

- **DSN-20250921-01** — Enable LoRA training for Qwen3-4B quantized checkpoints *(Accepted)*
  - Summary: Patched geometry + LoRA wrapper to handle grouped projections, added eval batching, kept fp16 fallback.
  - Link: `codex/dsn/dsn-20250921-lora-qwen3.md`

- **DSN-20250922-02** — Maintain LoRA guardrails for rank and pack size *(Proposed)*

- **DSN-20260608-01** — Target Gemma 4 12B mxfp8 for unified any-to-any packs *(Accepted)*
  - Summary: Uses mxfp8 as the default Gemma 4 runtime base, keeps bf16 for reference, makes `mlx-vlm`/`mlx-audio` the macOS modality stack, records IndustryBench as mechanics-positive but quality-negative, and selects `fault-codes-gemma4-it-answer-r32-300` as the first useful industrial pack candidate.
  - Link: `codex/dsn/dsn-20260608-gemma4-12b.md`
