# Architectural Decisions

## ADR-0001 — Rebrand to `mlx-plastic-rank`
- **Date:** 2025-09-12
- **Status:** Accepted
- **Context:** The original name `poprank` no longer reflected the MLX focus or packaging direction.
- **Decision:** Rename the project in `pyproject.toml`, CLI banner, and tests. Module moves under `src/mlx_plastic_rank/` would follow incrementally.
- **Consequences:** Update documentation, packaging metadata, and CI references; no functional changes expected.

## ADR-0002 — Create `codex/` for Maintainer Docs
- **Date:** 2025-09-12
- **Status:** Accepted
- **Context:** We needed a lightweight home for runbooks, ADRs, DSNs, and research notes without overloading contributor guides.
- **Decision:** Introduce `codex/runbook.md` and `codex/decisions.md`, with a Research Inbox for citations. Non-code decisions live here instead of in standard contributor docs.
- **Consequences:** Maintainers have a durable space for plans and experiments; contributors remain focused on `AGENTS.md`/`CONTRIBUTING.md`.

## ADR-0003 — LoRA Skill Packs for GPT-style Models
- **Date:** 2025-09-20
- **Status:** Accepted
- **Context:** Global SVD compression hurt perplexity, pushing the team toward reversible, modular adapters that keep base checkpoints intact.
- **Decision:** Wrap fused attention slices with `LoRAFusedLinear`, exporting packs as `.lora.{A,B}` (fp16) plus `alpha` (fp32). Enforce size ≤10 MB, alpha = 2r, and one active pack at a time through the `packs` CLI.
- **Consequences:** Packs stay small (≈O(r·hidden·layers)), adapters swap at runtime, and evaluation tooling now reports PPL deltas, load time, and memory metrics. Path-B delta exports will reuse the same schema.
