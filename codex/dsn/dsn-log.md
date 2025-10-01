# DSN Log

Track open and closed Decision Support Notes here for quick discovery. Link each entry to the corresponding markdown file under `codex/dsn/`.

- **DSN-20250912-01** — Python 3.13 default toolchain *(Accepted)*
  - Summary: Aligns local dev, CI, and MLX compatibility; future upgrades require DSN + ADR update.
  - Link: `codex/dsn/dsn-20250912-python313.md`

- **DSN-20250921-01** — Enable LoRA training for Qwen3-4B quantized checkpoints *(Accepted)*
  - Summary: Patched geometry + LoRA wrapper to handle grouped projections, added eval batching, kept fp16 fallback.
  - Link: `codex/dsn/dsn-20250921-lora-qwen3.md`

- **DSN-20250922-02** — Maintain LoRA guardrails for rank and pack size *(Proposed)*
