# Repository Guidelines

## Project Structure & Module Organization
Entry points live at `main.py` (CLI banner) and `plastic_rank.py` (demo loop combining `RankLayer`, `PlasticBlock`, and `PlasticityManager`). Core modules extend under `src/mlx_plastic_rank/`—notably `lowrank.py`, `plasticity_manager.py`, and the `packs/` toolkit. Tests mirror features in `tests/` (for example `tests/test_rank_layer.py`, `tests/test_manager_adapters.py`). Support assets reside in `data/` (sample corpora), `packs/` (generated skill packs; ignored in git), and `scripts/` (benchmarks, CLI helpers).

## Build, Test, and Development Commands
- Environment (optional): `uv venv` then `source .venv/bin/activate`; otherwise call `uv run …`.
- Install project: `uv pip install -e .`; enable compression extras with `uv pip install -e '.[compress]'`.
- Core demos: `uv run python main.py` (banner) and `uv run python plastic_rank.py --steps 10` (rank/sleep telemetry).
- LoRA CLI: `uv run packs create --name domain-demo --base qwen3-4b-2507-mlx-4bit --layers attn.q_proj,attn.k_proj,attn.v_proj --rank-strategy theorem --target-compression 0.9 --steps 1000 --batch-size 2 --learning-rate 5e-5 --data data/domain_prompts.jsonl`, then `uv run packs apply --name domain-demo --base qwen3-4b-2507-mlx-4bit --dry-run`, and `uv run packs eval --base qwen3-4b-2507-mlx-4bit --pack domain-demo --data-path data/domain_prompts.jsonl --csv results.csv`.
- Quantized training stays on the 4-bit base; k/v slices down-rank automatically. Use `--train-fp16-fallback` if a projection trips geometry checks.
- Tests: `uv run pytest -q` or target rank logic with `uv run pytest -q -k rank_layer`.

## Coding Style & Naming Conventions
Use Python 3.13 with 4-space indentation and UTF-8 files. Apply snake_case to functions, variables, and modules; reserve PascalCase for classes. Structure imports standard → third-party → local and remove unused lines. Public APIs should include concise docstrings plus pragmatic type hints. No autoformatter is enforced—avoid style-only churn in diffs.

## Testing Guidelines
- Pytest drives the suite; place cases in `tests/test_*.py` with functions `test_*`. Fix MLX seeds when asserting numeric tolerances, especially around pruning/waking flows. Run `uv run pytest -q` before pushing; capture failure logs for new adapters or CLI paths. Consider adding focused tests (`-k rank_layer`, `-k manager_adapters`) when modifying rank heuristics or pack wiring. Zero-impact tests on quantized Qwen (alpha=0) must continue to pass within `1e-6`.

## Commit & Pull Request Guidelines
Write imperative, scoped commit messages (e.g. `feat(rank): add prune threshold`, `fix(packs): guard alpha mismatch`). PRs should describe intent, link issues, and include before/after logs or CSV excerpts for demos. Verify `uv run python plastic_rank.py` and `uv run pytest -q` succeed prior to review. Keep changes focused; document trade-offs or research context in `codex/dsn/` and reference them from the PR.

## Security & Configuration Tips
Pin Python via `.python-version` (3.13) and prefer a local `.venv` for isolation. MLX targets Apple Silicon—follow upstream install guidance. Do not commit checkpoints, large datasets, or secrets; respect `.gitignore` and export artifacts via SafeTensors when needed.
