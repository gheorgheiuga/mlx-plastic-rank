# Decision Support Note (DSN)

**ID:** DSN-20250921-01  
**Title:** Enable LoRA training for Qwen3-4B quantized checkpoints  
**Date:** 2025-09-21  
**Status:** Accepted  
**Related Research Inbox Entry:** <none>

---

## Context
- Packs CLI currently targets Qwen3-style adapters (q/k/v slices) but fails on the 4-bit snapshot because projection weights are grouped (input dim 320 vs hidden 2560).
- Base evaluations succeed using the quantized checkpoint; LoRA training crashes inside `LoRAFusedLinear` during `adapter.delta` matmul.
- Need guidance: adapt code for grouped projections, switch to bf16 checkpoints, or narrow support to compatible models.

## Options Considered
1. Patch LoRA wrappers for grouped projections
   - Pros: Keeps quantized models usable; aligns with README examples.
   - Cons: Requires careful reshaping/splitting logic; increases complexity and test surface.
2. Use bf16 (full) checkpoints for training
   - Pros: Simplifies math; matches existing logic; minimal code changes.
   - Cons: Larger downloads/VRAM; documentation needs updates; quantized inference packs not produced.
3. Drop Qwen3 quantized support for training, revisit later
   - Pros: No immediate work; focus on other models.
   - Cons: Docs/demo lose alignment; less compelling for users with quantized pipelines.

## Decision
- Chosen option: Patch LoRA wrappers for grouped projections while retaining a guarded fp16 fallback.  
- Rationale: Preserves the quantized development workflow on Apple Silicon, keeps packs interoperable without swapping base checkpoints, and still provides an escape hatch for pathological geometries.

## Consequences
- Immediate impacts: LoRA geometry probing now understands grouped heads, adapters run on 4-bit Qwen3 bases, pack ranks adjust per slice, and eval batching is upgraded.  
- Risks/unknowns: Shape detection regressions, throughput drops from extra matmuls, pack size creep.  
- Mitigations: Strict asserts with logging, TPS/PPL checks in `packs eval-batch`, pack size gates, and fp16 fallback via `--train-fp16-fallback`.

## Follow-ups
- [x] Update README/AGENTS once decision lands.
- [x] Add regression tests covering chosen approach.
- [x] Document pack compatibility in `codex/runbook.md`.

---

Document outcomes in the runbook once implemented.
