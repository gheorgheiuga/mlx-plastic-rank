# Repository Guidelines

## Project Structure & Module Organization
- Entry points: `main.py` (CLI banner), `plastic_rank.py` (MLX: `RankLayer`, `PlasticBlock`, `PlasticityManager` + demo loop)
- Metadata: `pyproject.toml` (Python ≥ 3.13), `.python-version` (pyenv)
- New code under `src/mlx_plastic_rank/`; tests in `tests/` (e.g., `tests/test_rank_layer.py`)

## Build, Test, and Development Commands
- Create env (uv): `uv venv` (optional) and `source .venv/bin/activate` or use `uv run` without activating
- Install deps: `uv pip install mlx`; dev: `uv pip install pytest`
- Demo: `uv run python plastic_rank.py` (mini training loop; rank/sleep stats)
- Sanity: `uv run python main.py`
- Tests: `uv run pytest -q` or `uv run pytest -q -k rank_layer`

## Coding Style & Naming Conventions
- Python 3.13, 4‑space indentation, UTF‑8
- Naming: snake_case (functions/vars/modules), PascalCase (classes)
- Imports: standard → third‑party → local; remove unused
- Docs/types: concise docstrings and practical type hints for public APIs
- Formatter: none configured; keep style-only diffs minimal

## Testing Guidelines
- Framework: pytest
- Naming: files `tests/test_*.py`; functions `test_*`
- Focus: `RankLayer`, pruning/waking paths, shape/dtype behavior; fix seeds for determinism with MLX ops
- Commands: `pytest -q` for CI-like output; `pytest -vv -k rank_layer` to target specifics

## Commit & Pull Request Guidelines
- Commits: imperative and scoped (e.g., `feat(rank): add prune threshold`, `fix(demo): guard None bias`)
- PRs: include description, rationale, linked issues, and before/after logs or screenshots; note local demo/test results
- Readiness: run `python plastic_rank.py` and `pytest -q` before requesting review; keep changes focused and documented

## Security & Configuration Tips
- Use `.python-version` (3.13) and a local `.venv` for isolation
- MLX requires Apple Silicon/macOS; follow MLX install docs
- Do not commit large artifacts or credentials; respect `.gitignore`
