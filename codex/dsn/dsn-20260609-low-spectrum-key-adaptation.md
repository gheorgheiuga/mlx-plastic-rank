# Decision Support Note (DSN)

**ID:** DSN-20260609-03  
**Title:** Test low-spectrum key-projection adaptation  
**Date:** 2026-06-09  
**Status:** Accepted as local experimental direction  
**Evidence Status:** Quality verified on fault-code eval/generation; broader validation experimental  
**Related Research Inbox Entry:** Pop matrix-polynomial rank identity; spectral LoRA rank allocation

---

## Context
- The Pop/Negrescu theorem gives exact rank accounting for matrix polynomials, but it does not by itself identify which LoRA subspaces improve downstream quality.
- A local polynomial probe now verifies the theorem on chosen operators and measures adapter energy in spectral notch subspaces.
- Fault-code Gemma 4 runs show a useful compression target: `fault-codes-gemma4-it-answer-hetero-r32-init8-min4-map-600` is close to fixed `r32/600` quality at about half the adapter size.
- External context supports the general direction, not this specific claim: Pop and Negrescu frame the rank identity through Jordan/Frobenius rank arguments, while AdaLoRA and related LoRA literature motivate non-uniform rank allocation across matrices.

## Options Considered
1. Treat generic Pop polynomial identities as the rank selector
   - Pros: Directly follows the paper framing and is simple to explain.
   - Cons: Default and single-root polynomial pairs were non-discriminating on projected Gemma operators.
2. Use spectral notch probes as diagnostics for adapter subspace placement
   - Pros: Produces layer/target-specific signals and keeps the theorem as rank accounting rather than quality proof.
   - Cons: The signal is geometric and local; it still needs training and held-out evaluation.
3. Ignore the spectral result and keep only dynamic factor-norm gates
   - Pros: Already implemented and tested.
   - Cons: Misses a strong target-specific pattern in the trained packs.

## Decision
- Chosen option: Use spectral notch probes as diagnostics for fault-code rank-map allocation experiments.
- Rationale: The strongest local signal is not generic polynomial overlap. It is that trained `k_proj` adapters, especially full-attention key projections, put substantially more adapter energy into low-spectrum/null-ish directions of their own base row-Gram operator. A same-budget spectral-key candidate beat the current hetero map on held-out answer-token PPL/accuracy and matched its generation overlap.
- Canonical decision record: `codex/decisions.md` ADR-0007.

## Evidence
- Probe code: `src/mlx_plastic_rank/pop_polynomial_probe.py` and `scripts/pop_polynomial_probe.py`.
- Local artifacts:
  - `out/pop_poly_k_all_layers_low_mid_high_seed5.json`
  - `out/pop_poly_q_all_layers_low_mid_high_seed5.json`
  - `out/pop_poly_v_all_layers_low_mid_high_seed5.json`
  - `out/pop_poly_qkv_target_summary.json`
  - `out/pop_poly_spectral_key_rank_map_candidate.json`
  - `out/pop_poly_rank_map_spectral_auto_balanced.json`
- Five projection-seed sweep at projection dimension 256, notch size 8, expected random baseline `8/256 = 0.03125`:
  - `k_proj`: mean low-spectrum lift 1.455, full-attention `k_proj` mean low lift 2.222, sliding `k_proj` mean low lift 1.301.
  - `q_proj`: mean low-spectrum lift 1.066.
  - `v_proj`: mean low-spectrum lift 1.060.
- `k_proj` low lift has weak correlation with declared rank, so this is not just a larger-rank artifact.
- Fixed-rank pack comparison on selected `k_proj` layers suggests the low-spectrum key signal is a trained adapter/task signature, not unique to the dynamic-rank map.
- The first candidate map promotes full-attention `k_proj` layers 5, 11, and 35 to rank 32 and compensates by lowering nine low-signal `q_proj` ranks. It remains slightly under the source hetero map by the ledger parameter estimate.
- The `packs rank-map spectral` allocator reproduces that candidate automatically from the q/k/v probe JSON and source pack tensors; its generated `rank_map` and `alpha_map` exactly match the manual candidate.
- Candidate training artifact: `packs/spectral-key-candidate`.
- Candidate eval artifacts:
  - `out/fault_codes_gemma4_it_answer_spectral_key_candidate_eval_300.json`
  - `out/fault_codes_generation_spectral_key_candidate_8.json`
  - `out/fault_codes_rank_ledger_spectral_key_candidate.json`
  - `out/fault_codes_rank_compare_spectral_key_candidate_vs_hetero.json`
- Held-out 300-row answer-token eval:
  - `spectral-key-candidate`: 27.38 MB, PPL 5.7641, token accuracy 0.6786, PPL delta -62.65%.
  - Current hetero map: 27.39 MB, PPL 5.7811, token accuracy 0.6773, PPL delta -62.54%.
  - Fixed `r32/600`: 54.16 MB, PPL 5.6365, token accuracy 0.6802, PPL delta -63.47%.
- 8-example generation solution-overlap:
  - `spectral-key-candidate`: 0.3911.
  - Current hetero map: 0.3911.
  - Fixed `r32/600`: 0.4025.
- Rank ledger:
  - `spectral-key-candidate`: 136 adapters, declared/effective rank 2316, rank slack 0, 28,666,400 bytes, 12,377.55 bytes per effective rank.
  - Versus current hetero: left effective rank 2316, right effective rank 2288, composition rank 2422, rank savings 2182, composition efficiency 0.5261.

## Consequences
- Immediate impact: Future Pop Rank experiments should separate three claims:
  - The Pop identity is valid rank accounting on a chosen operator.
  - Spectral notch energy is a diagnostic of where adapters place capacity.
  - Quality benefit requires held-out evaluation after a rank-map or training change.
- Risk: Low-spectrum `k_proj` energy may be epiphenomenal. It could reflect base architecture or training path without being causally useful.
- Mitigation: Keep fixed `r16/r32`, freeze, and current hetero maps as baselines; only promote if an equal-size spectral-key-biased run beats them on answer-token accuracy/perplexity and generation checks.
- Validation gate: passed locally against `fault-codes-gemma4-it-answer-hetero-r32-init8-min4-map-600` on held-out answer-token PPL/accuracy at equal size and matched the generation overlap check.
- Falsification gate: If promoted key ranks do not improve quality or if the signal disappears across another dataset/base, keep this as a diagnostic only.

## Follow-ups
- [x] Generate a spectral-key-biased rank map with the same approximate byte budget as the current hetero pack.
- [x] Add a repeatable rank-map generation command for the spectral-key candidate.
- [x] Train/evaluate the proposed map against `r16/600`, `r32/600`, freeze, and current hetero baselines.
- [ ] Add a compact runbook command for the all-layer q/k/v spectral sweep if this line of work continues.
- [ ] Repeat with at least one additional seed or another industrial dataset before claiming a general Pop Rank advantage.
- [ ] Decide whether to keep the probe as research tooling or fold it into `packs rank-ledger`.

---
