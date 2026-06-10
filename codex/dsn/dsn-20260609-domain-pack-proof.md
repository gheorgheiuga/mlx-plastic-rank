# Decision Support Note (DSN)

**ID:** DSN-20260609-04  
**Title:** Add artifact-backed domain pack proof reports  
**Date:** 2026-06-09  
**Status:** Accepted as productization gate  
**Evidence Status:** Mechanics and local fault-code quality proof verified; broader validation experimental  
**Related Research Inbox Entry:** DLC-style LoRA pack workflow; base+pack domain improvement

---

## Context
- The project now has enough pieces to show the core product loop: keep a base model immutable, train a LoRA skill pack from domain data, attach it as a small DLC-style artifact, and evaluate base+pack against the base on held-out domain prompts.
- Existing evidence was scattered across eval JSON, generation JSON, rank-ledger JSON, pack metadata, README tables, and local commands.
- For open-source/product use, claims need to be machine-checkable from artifacts instead of prose.

## Options Considered
1. Keep proof in README tables
   - Pros: Simple and readable.
   - Cons: Easy to drift from artifacts; hard to fail CI or compare new packs.
2. Add a `packs proof` artifact audit
   - Pros: Reuses existing train/eval/generation/ledger outputs and emits explicit pass/fail requirements.
   - Cons: It audits completed artifacts; it does not itself train or evaluate a model.
3. Build a full `bakeoff` command immediately
   - Pros: One-command train/eval/proof loop.
   - Cons: Too much scope before the proof schema is stable.

## Decision
- Chosen option: Add `packs proof` as the productization gate for DLC-style domain improvement claims.
- Rationale: The command turns existing artifacts into an auditable report with requirement-level evidence: pack artifact exists, training/eval data exist, eval rows match the requested base and pack, attach changes logits, held-out answer metrics improve, optional generation improves, and optional rank ledger proves active adapter capacity.
- Canonical decision record: `codex/decisions.md` ADR-0008.

## Evidence
- Code:
  - `src/mlx_plastic_rank/packs/proof.py`
  - `src/mlx_plastic_rank/packs/cli.py` (`packs proof`)
  - `src/mlx_plastic_rank/packs/io.py` pack metadata training provenance fields for newly created packs.
- Tests:
  - `tests/test_domain_pack_proof.py`
  - `tests/test_packs_cli_parser.py`
  - `tests/test_pack_inspection.py`
- Local proof artifact:
  - `out/fault_codes_domain_pack_proof_spectral_key_candidate.json`
- Local proof result for `spectral-key-candidate`:
  - Status: `passed`.
  - Base model: `mlx-community/gemma-4-12B-it-qat-mxfp8`.
  - Domain: `industrial-fault-codes`.
  - Pack artifact: `packs/spectral-key-candidate`.
  - Attach evidence: `max_logit_diff=27.6875`.
  - Held-out answer-token eval: base PPL 15.4316 to pack PPL 5.7641; token accuracy 0.6155 to 0.6786.
  - Generation check: solution-keyword overlap 0.2723 to 0.3911.
  - Rank ledger: 136 adapters, effective rank 2316, rank slack 0.
- Full-split follow-up on `avneetsingla/industrial-fault-codes-sample`:
  - Data: 2,700 train rows and 300 held-out eval rows from all 3,000 usable
    source rows.
  - Base model: `mlx-community/gemma-4-12B-it-qat-mxfp8`.
  - Baseline held-out answer eval: PPL 15.4316, token accuracy 0.6155.
  - Fixed r32 pack, 600 steps: PPL 5.5622, token accuracy 0.6802,
    pack size 54.16 MB, declared/effective rank 4,352.
  - Fixed r16 pack, 600 steps: PPL 8.5677, token accuracy 0.6515,
    pack size 27.10 MB, declared/effective rank 2,176.
  - Learned hetero rank-map pack, 600 steps from the 150-step dynamic
    full-split rank map: PPL 5.9406, token accuracy 0.6748, pack size
    23.73 MB, declared/effective rank 1,984.
  - Hetero generation check on 16 deterministic held-out prompts: solution
    keyword overlap improves from 0.2440 to 0.3438; brand rate remains 1.0 and
    code rate remains 0.9375.
  - Hetero proof artifact: `out/fault_codes_full2700_domain_pack_proof_hetero_r32_init8_min4_map_600.json`
    with status `passed`.
  - Summary artifact: `out/fault_codes_full2700_fullscale_summary.json`.

## Consequences
- Immediate impact: The repo can now express the product claim as a verifiable report rather than an informal README table.
- New packs created by `packs create` persist training provenance in metadata, improving future proof strength.
- Existing packs without provenance can still be audited with explicit `--train-data`; their report should be read as external training-data evidence rather than metadata-level proof.
- The full-split bakeoff shows a useful size/quality tradeoff: fixed r32 is the
  best local quality result, but the learned hetero rank map retains most of the
  gain at less than half the r32 artifact size and beats fixed r16 at a smaller
  size.
- This still does not prove a general Pop Rank theorem advantage. It proves that one local domain pack can be attached and improve one held-out industrial-domain evaluation.

## Follow-ups
- [x] Add a one-command `packs bakeoff` workflow that trains, evaluates, ledgers, and proves strategies end-to-end. Generation checks remain domain-specific for now.
- [ ] Add a compact Markdown or HTML renderer for proof reports.
- [ ] Require metadata training provenance for release-grade packs after older local artifacts are regenerated.
- [ ] Repeat proof reports on at least one additional dataset/base before making a broader product claim.

---
