# Contributing

The active prototype consists of low-rank mechanics, pack tools and experiment
integrity. [DSN-20260905-04](codex/dsn/dsn-20260905-prototype-consolidation.md)
defines the scope; [the runbook](codex/runbook.md) contains operator commands.

## Local setup and checks

Use Python 3.13 via `uv`, the pinned lockfile and the project-local `.venv`.
Activation and global installs are unnecessary. Select optional dependencies
with `--extra packs`, `--extra compress` or `--extra data` only when needed.

```sh
uv sync --locked
uv run --locked python plastic_rank.py --steps 10
uv run --locked ruff check
uv run --locked mypy
uv run --locked pytest -q
```

The demo checks a bounded prune/restore cycle. Tests use synthetic models and
local artifacts; passing them does not establish a real-model quality result.
Use deterministic MLX seeds for numerical regressions. The default command runs
`tests/core`, `tests/packs` and `tests/tools`. Parked research is explicit:
`uv run --locked pytest -q tests/research`; use `uv run --locked pytest -q tests`
for every suite. Run research checks when changing the research or shared
mechanics it exercises. See [suite guidance](tests/README.md).

## Changes and decisions

Keep changes focused, explain the resulting behavior, and include validation
and material limitations. Use scoped imperative commit subjects. Avoid adding
parallel entry points or generic frameworks for one-off experiments.

Design/research changes need a DSN in `codex/dsn/`, a DSN-log update and a matching
entry in `codex/decisions.md`, or an explicit Proposed/Experimental status.
Separate an accepted engineering decision from evidence for scientific benefit.
Name the test that would falsify or promote a hypothesis. The current next step
is [baseline diagnosis](codex/dsn/dsn-20260905-baseline-validity-diagnostic.md);
parked controllers are not an automatic implementation backlog.

Preserve failed runs and content identities. Never overwrite historical outputs
or reinterpret shared-seed results as matched initialization. Keep generated
data, packs and checkpoints ignored; update `NOTICE.md` and generator metadata
when changing artifact sources. See [data guidance](data/README.md).

Release/version changes and publication are separate decisions after review.
