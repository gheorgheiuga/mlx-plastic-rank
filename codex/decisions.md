# Decisions (ADRs)

ADR-0001 Rebrand to `mlx-plastic-rank`
- Date: 2025-09-12
- Status: Accepted
- Context: Original name `poprank` conflicted with broader scope and upcoming packaging. Align naming with MLX focus.
- Decision: Rename project in `pyproject.toml`, CLI banner, and tests. Keep module filenames for now; future move to `src/mlx_plastic_rank/`.
- Consequences: Update docs, test expectations, and any packaging metadata. No functional change.

ADR-0002 Create `codex/` folder for runbook and ADRs
- Date: 2025-09-12
- Status: Accepted
- Context: Need a lightweight place for plans, notes, and research pointers that do not belong in contributor guidelines.
- Decision: Add `codex/runbook.md` and `codex/decisions.md`. Use runbook “Research Inbox” for citations and summaries; avoid committing large binaries.
- Consequences: Central place for evolution; reduces churn in `AGENTS.md`.

