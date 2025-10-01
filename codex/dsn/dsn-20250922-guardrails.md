# Decision Support Note (DSN)

**ID:** DSN-20250922-02  
**Title:** Maintain LoRA guardrails for rank and pack size  
**Date:** 2025-09-22  
**Status:** Proposed  

---

## Context
- Current packs CLI enforces `(2,4,8)` rank choices and <=6/12 MB size gates for Qwen3 4-bit.
- Pop-theorem rank selection occasionally returns values outside the set if unrestricted.
- We need clarity on whether to keep guardrails or allow flexible ranks/size.

## Options Considered
1. Keep existing guardrails (ALLOWED_RANKS and size limits)
   - Pros: predictable pack size, safe GPU footprint, grouped-head math remains intact, tests already cover it.
   - Cons: theorem may want intermediate ranks; users can’t request 1/6 etc.
2. Relax rank set to allow any value returned by theorem
   - Pros: respects raw output, more granular packs.
   - Cons: risk of k/v mismatch, larger packs, more testing overhead.
3. Remove both rank and size guardrails
   - Pros: maximum flexibility.
   - Cons: easy to produce oversized packs, potential runtime failures.

## Decision
- Chosen option: <TBD>
- Rationale: <fill after approval>

## Consequences
- Immediate impacts: <document once decision made>  
- Risks/unknowns: <enumerate>  
- Mitigations: <tests, gates, fallbacks>

## Follow-ups
- [ ] Update README/AGENTS with final stance.
- [ ] Adjust inspection/manager tests accordingly.
- [ ] Document behaviour in runbook + release notes.

---

