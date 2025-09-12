# Contributing

Thank you for helping improve mlx-plastic-rank!

## Why Plastic Rank?

Most approaches to model compression prune or distill, permanently cutting capacity and often losing subtle behaviors. Plastic rank takes inspiration from neuroplasticity: instead of discarding, weights are re-expressed in adaptive low-rank factors that can grow, shrink, or wake as needed. This makes compression reversible, transparent, and tunable — a closer analogy to how brains reorganize rather than amputate. The result is not just smaller models, but flexible ones that adjust rank dynamically while preserving knowledge.

Setup
- Use Python 3.13. Create an env with `uv venv` (optional) and `uv pip install -e .`.
- Run tests: `uv run pytest -q`.
- Lint and type check: `ruff .` and `mypy .` (configs provided).

Development guidelines
- Prefer small, focused PRs. Include before/after logs for demos where relevant.
- Add or update tests for new behavior.
- For nontrivial changes or tradeoffs, include a DSN (Decision Support Note) in `codex/dsn/` and link it from the PR description. A DSN template is provided.
- Avoid committing large binaries; use SafeTensors for model artifacts when necessary.

Release checklist
- Ensure `pyproject.toml` version is updated.
- Confirm CI is green on macOS (MLX).
- Tag the release: `git tag vX.Y.Z && git push origin vX.Y.Z`.
