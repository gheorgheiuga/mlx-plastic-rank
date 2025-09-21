# Contributing

Thank you for helping improve **mlx-plastic-rank**. This document summarizes local setup, workflow expectations, and quality gates for pull requests.

## Local Setup
- Python 3.13 is required. Create an optional virtual environment with `uv venv` then `source .venv/bin/activate`.
- Install the project in editable mode: `uv pip install -e .`. Add compression extras when needed: `uv pip install -e '.[compress]'`.
- Run `uv run python main.py` and `uv run python plastic_rank.py --steps 10` to confirm MLX works on your machine.

## Development Workflow
- Keep changes focused and incremental. Include before/after logs or CSV snippets for demos, benchmarks, or pack workflows.
- Add or adapt tests whenever behavior changes. The suite lives under `tests/`; prefer deterministic seeds for MLX ops.
- For design trade-offs or research-driven work, author a Decision Support Note (DSN) in `codex/dsn/` using `DSN-TEMPLATE.md`, and cross-link it from the PR description.
- Avoid committing large binaries. Store generated packs and datasets outside the repo or ensure they remain git-ignored.

## Quality Gates
- Tests: `uv run pytest -q`
- Targeted checks: `uv run pytest -q -k rank_layer` (rank heuristics) and `uv run pytest -q -k manager_adapters` (LoRA wrapping)
- Static analysis: `uv run ruff check`
- Type checking: `uv run mypy`

Run these commands before opening a pull request. If CI uncovers issues unique to Apple Silicon or MLX, document the resolution in the PR thread.

## Commit & PR Guidelines
- Write imperative commit subjects with a scope (e.g. `feat(rank): add prune threshold`, `fix(packs): guard alpha mismatch`).
- Summaries should explain the rationale and highlight risks or follow-up work.
- Pull requests should describe intent, link issues, and reference any supporting DSNs or research notes in `codex/`.
- Confirm the demos (`main.py`, `plastic_rank.py`) and tests pass locally prior to requesting review.

## Release Checklist
- Update the version in `pyproject.toml`.
- Ensure CI is green on the latest macOS/MLX runner.
- Tag and publish: `git tag vX.Y.Z && git push origin vX.Y.Z`.
