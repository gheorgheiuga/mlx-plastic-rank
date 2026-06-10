# Decision Support Note (DSN)

**ID:** DSN-20260610-01
**Title:** Add reproducible pack bakeoff workflow
**Date:** 2026-06-10
**Status:** Accepted as orchestration workflow
**Evidence Status:** Workflow implementation and compact fault-code snapshot added; second-domain validation pending
**Related Research Inbox Entry:** Replicable Gemma 4 pack proof across datasets

---

## Context
- The project has separate commands for training packs, evaluating base+pack,
  measuring rank ledgers, and proving artifact-backed domain improvement.
- The first full-scale fault-code result is promising, but raw `out/`, `data/`,
  and `packs/` artifacts are intentionally ignored and not reviewable in git.
- Product or open-source claims need a repeatable command path and small
  committed evidence snapshots instead of hand-assembled local summaries.

## Options Considered
1. Keep manual command sequences in README
   - Pros: No new code.
   - Cons: Easy to drift, hard to repeat, and weak for review.
2. Add `packs bakeoff` as an orchestration layer
   - Pros: Reuses existing CLI phases, writes logs, supports resumable runs,
     and creates a compact summary from artifacts.
   - Cons: It still relies on long-running local Gemma training for real proof.
3. Commit raw experiment outputs
   - Pros: Maximally detailed.
   - Cons: Pulls ignored generated data, logs, and pack artifacts into git.

## Decision
- Chosen option: Add `packs bakeoff --spec path.json` as the reproducible
  orchestration workflow.
- The command reads a JSON spec, runs `packs create`, `packs eval`,
  `packs rank-ledger`, and `packs proof` for each candidate, records phase logs,
  and writes compact JSON/CSV summaries.
- Add `codex/evidence/` for small review snapshots and `codex/bakeoffs/` for
  reproducible specs.
- Use `gretelai/synthetic_text_to_sql` as the second large replication dataset
  because the Hugging Face card exposes Apache-2.0 license metadata and large
  train/test splits.
- Canonical decision record: `codex/decisions.md` ADR-0009.

## Evidence
- Code:
  - `src/mlx_plastic_rank/packs/bakeoff.py`
  - `src/mlx_plastic_rank/packs/cli.py` (`packs bakeoff`)
  - `scripts/text_to_sql_extract.py`
- Specs and snapshots:
  - `codex/bakeoffs/fault_codes_gemma4_it_fullscale.json`
  - `codex/bakeoffs/text_to_sql_gemma4_it_fullscale.json`
  - `codex/evidence/fault_codes_full2700_fullscale_summary.json`
- Tests:
  - `tests/test_bakeoff.py`
  - `tests/test_text_to_sql_extract.py`
  - `tests/test_packs_cli_parser.py`

## Consequences
- The repo now has a repeatable productization path for comparing fixed-rank,
  dynamic discovery, and fresh heterogeneous rank-map packs.
- The full-split fault-code result is reviewable from committed compact
  evidence without committing raw local artifacts.
- Broader Pop Rank claims remain experimental until the Text-to-SQL spec, or
  another permissive large dataset, completes and supports the same tradeoff.
- Promotion criteria are explicit: the hetero/rank-map candidate must pass
  proof, beat fixed r16, retain at least 90% of fixed r32 improvement over base,
  and stay under 60% of fixed r32 adapter bytes.

## Follow-ups
- [ ] Run the Text-to-SQL full-scale bakeoff.
- [ ] Repeat the best candidate with at least two seeds.
- [ ] Add generation or task-specific exact-match gates when a domain has a
  stable non-PPL task metric.
