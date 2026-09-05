# Repository Guidelines

## Project Structure & Module Organization
The entry points are `plastic_rank.py` (bounded factor lifecycle demo) and the `packs` CLI. Core modules live under `src/mlx_plastic_rank/`, particularly `lowrank.py`, `factorization.py` and `packs/`. Tests mirror features in `tests/`. Generated data and packs stay ignored. DSN-20260905-04 defines current scope; parked controllers remain for reproducibility, not as an active implementation backlog.

## Build, Test, and Development Commands
- Environment/install: `uv sync --locked` into the project-local `.venv`; no activation or global installs.
- Demo: `uv run --locked python plastic_rank.py --steps 10` checks a conserved four-component prune/restore cycle.
- Pack tools: `uv run --locked --extra packs packs …`; follow `codex/runbook.md` with an existing compatible checkpoint and disjoint training/held-out data. Start from fixed rank, and use `--initialization component-v1` for new allocation comparisons. `gram_energy` is the descriptive heuristic name; `theorem` is a legacy alias.
- Optional utilities: add `--extra compress` for checkpoint downloading/compression or `--extra data` for extractors using Hugging Face `datasets`.
- Checks: `uv run --locked pytest -q`, `uv run --locked ruff check`, `uv run --locked mypy`. Default tests cover core, packs and utilities. Use `uv run --locked pytest -q tests/research` for parked research, or `uv run --locked pytest -q tests` for every suite. Include research checks when changing its code or shared mechanics it exercises.

## Coding Style & Naming Conventions
Use Python 3.13 with 4-space indentation and UTF-8 files. Apply snake_case to functions, variables, and modules; reserve PascalCase for classes. Structure imports standard → third-party → local and remove unused lines. Public APIs should include concise docstrings plus pragmatic type hints. No autoformatter is enforced—avoid style-only churn in diffs.

## Testing Guidelines
- Pytest drives the suites under `tests/core`, `tests/packs`, `tests/tools` and `tests/research`. Place tests by behavior; use parameterized cases instead of repeated setup. Fix MLX seeds for numeric assertions, especially pruning/waking. Run the default checks before review. All suites must work without downloaded checkpoints or optional dataset/model loaders. Zero-impact tests on quantized adapters (alpha=0) must continue to pass within `1e-6`.

## Commit & Pull Request Guidelines
Write imperative, scoped commit messages (e.g. `feat(rank): add prune threshold`, `fix(packs): guard alpha mismatch`). PRs should describe the resulting behavior and relevant validation. Verify the locked demo, tests and static checks before review. Keep changes focused and reference design trade-offs from `codex/dsn/`.

## Research & Decision Records
Treat `codex/decisions.md` as the canonical decision index. When adding or changing a DSN in `codex/dsn/`, update `codex/dsn/dsn-log.md` and either add a matching entry in `codex/decisions.md` or keep the DSN status as `Proposed`/`Experimental`. Do not mark a DSN `Accepted` when it only proves scaffolding, mechanics, or a hypothesis; record the evidence status separately and name the validation or falsification test that would promote it. For Pop Rank/theorem work, distinguish implementation instrumentation from proof of quality benefit, and keep unresolved validation gaps visible in the decision record.

## License & Attribution
When adding or changing datasets, generated JSONL samples, model checkpoints, copied snippets, or benchmark artifacts, update `NOTICE.md` and any local generator metadata. Keep generated data under `data/` ignored by default, and use `data/README.md` to document source/license expectations for regenerated files.

## Security & Configuration Tips
Pin Python via `.python-version` (3.13) and prefer a local `.venv` for isolation. MLX targets Apple Silicon—follow upstream install guidance. Do not commit checkpoints, large datasets, or secrets; respect `.gitignore` and export artifacts via SafeTensors when needed.
