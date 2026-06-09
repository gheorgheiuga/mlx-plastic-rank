# Decision Support Note (DSN)

**ID:** DSN-20260609-01  
**Title:** Measure pack rank algebra before claiming theorem advantage  
**Date:** 2026-06-09  
**Status:** Accepted  
**Evidence Status:** Instrumentation verified; theorem and quality validation open
**Related Research Inbox Entry:** Pop theorem / rank-polynomial intuition

---

## Context
- The project now has a positive useful-pack signal on Gemma 4 IT QAT using industrial fault-code diagnostics.
- That result validates the pack/DLC architecture, but it does not prove that Pop's rank theorem is the reason the pack works.
- The next claim is different: rank algebra should explain or improve pack selection, composition, and overlap.

## Options Considered
1. Continue with fixed-rank sweeps only
   - Pros: easy to compare and already validated.
   - Cons: does not test the theorem intuition; rank remains a manual hyperparameter.
2. Move directly to theorem-selected training runs
   - Pros: faster path to a claim if it works.
   - Cons: hard to diagnose failures because there is no measurement layer for rank overlap or slack.
3. Add a rank ledger first
   - Pros: measures effective rank, rank slack, stable rank, row/column overlap, composition rank, and rank savings for real packs without loading the base model.
   - Cons: still instrumentation, not proof.

## Decision
- Chosen option: Add the rank ledger first.
- Rationale: It turns the theorem intuition into measurable pack accounting before we run expensive rank-selection bakeoffs or claim theorem advantage.
- Canonical decision record: `codex/decisions.md` ADR-0005.

## Consequences
- Immediate impacts: `packs rank-ledger` can inspect one pack or compare two packs from their SafeTensors factors.
- Risks/unknowns: The ledger measures low-rank LoRA operators, not matrix polynomials of the base model. It should be described as rank-algebra instrumentation, not as a direct theorem proof.
- Mitigations: Keep result language precise, then run fixed-rank vs stable-rank vs theorem-rank bakeoffs using the same ledger, eval, and generation metrics.
- Validation gate: Promote Pop Rank claims only if ledger metrics predict or explain held-out quality-per-MB improvements.

## First Readout
- `fault-codes-gemma4-it-answer-r32-300` has 136 adapters, declared rank 4352, effective rank 4352, zero rank slack, and about 13,041 bytes per effective rank.
- Compared with `fault-codes-gemma4-it-answer-r16-300`, the shared adapters compose additively: left effective rank 2176, right effective rank 4352, composition rank 6528, rank savings 0, row/column overlap 0.
- Mean absolute Frobenius cosine between matching adapters is about 0.0097, with maximum absolute cosine about 0.0356.
- Interpretation: the stronger rank-32 pack mostly adds new rank directions rather than duplicating the rank-16 pack.

## Follow-ups
- [x] Add `packs rank-ledger` command.
- [x] Run the ledger on the current best fault-code pack.
- [x] Compare rank-16 / 300 against rank-32 / 300.
- [ ] Run a controlled bakeoff: fixed rank, stable auto-rank, and theorem auto-rank with the same steps/data/eval.
- [ ] Add base-weight-aware ledger mode if we need direct base-vs-pack subspace accounting.

---
