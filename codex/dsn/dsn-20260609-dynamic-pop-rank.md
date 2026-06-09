# Decision Support Note (DSN)

**ID:** DSN-20260609-02  
**Title:** Implement dynamic Pop Rank with gated active ranks  
**Date:** 2026-06-09  
**Status:** Accepted  
**Evidence Status:** Mechanics verified; quality benefit experimental
**Related Research Inbox Entry:** Pop theorem / dynamic rank allocation

---

## Context
- Fixed-rank fault-code packs showed that more rank can improve task quality, but fixed rank spends capacity uniformly across all selected adapters.
- The next research question is whether rank can be allocated during training only where the base model needs help.
- Physical tensor resizing during MLX training would complicate optimizer state and make the first experiment fragile.

## Options Considered
1. Resize LoRA tensors during training
   - Pros: final rank is physically dynamic throughout training.
   - Cons: optimizer state and tensor shape changes are brittle.
2. Train max-rank tensors with hard active-rank gates
   - Pros: tensor shapes stay stable, active rank can grow/shrink, and export can write only active columns.
   - Cons: inactive columns do not receive gradients until opened, and the grow/shrink signal is heuristic.
3. Keep fixed-rank sweeps only
   - Pros: simplest baseline.
   - Cons: does not test the user's dynamic Pop Rank hypothesis.

## Decision
- Chosen option: Train max-rank tensors with hard active-rank gates.
- Rationale: It is the smallest reliable implementation that lets each adapter earn rank during training while keeping MLX training stable.
- Canonical decision record: `codex/decisions.md` ADR-0006.

## Implementation
- `SliceLoRA` now supports optional rank gates and exports only active columns.
- `LoRAManager.initialize_adapters(..., initial_active_rank=N)` starts adapters with an active rank prefix.
- `LoRAManager.adjust_dynamic_ranks(...)` grows or shrinks adapters along the allowed rank ladder based on learned rank signal.
- `TrainingConfig` and `packs create` expose `--dynamic-rank`, `--dynamic-initial-rank`, warmup/interval, grow threshold, and prune threshold.

## Consequences
- Immediate impacts: `--rank` can now mean maximum rank during training instead of final exported rank.
- Risks/unknowns: The current grow/shrink signal is adapter-level utility from learned factor norms, not a validation-loss oracle. Real quality-per-MB benefit still needs Gemma fault-code bakeoff.
- Mitigations: Keep fixed-rank baselines, use the rank ledger after training, and compare eval/generation metrics before claiming dynamic Pop Rank wins.
- Validation gate: Promote dynamic Pop Rank claims only when gated runs beat fixed-rank baselines on held-out quality metrics at equal or lower exported adapter size.

## Follow-ups
- [x] Add gated active-rank LoRA support.
- [x] Export only active rank columns.
- [x] Add CLI flags for dynamic rank training.
- [x] Add unit tests for gated deltas, dynamic grow, and active-rank export.
- [ ] Run `fault-codes-gemma4-it-answer-dynamic-r32-init4-300`.
- [ ] Compare dynamic result against fixed `r16/300` and `r32/300` with eval, generation check, and rank ledger.

---
