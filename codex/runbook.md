# Prototype runbook

Current scope: [DSN-20260905-04](dsn/dsn-20260905-prototype-consolidation.md).
The default workflow demonstrates low-rank mechanics. Model training is an
optional workflow, and research admission remains blocked by failed controls.

## Environment and checks

Use the repository root, `.python-version`, `pyproject.toml`, `uv.lock` and a
project-local `.venv`. Do not use a global Python installation or shell activation.

```sh
uv sync --locked
uv run --locked python plastic_rank.py --steps 10
uv run --locked ruff check
uv run --locked mypy
uv run --locked pytest -q
```

The demo should report active ranks `4, 2, 4`, then hold at `4`, with dormant
counts `0, 2, 0`. It measures restoration error against the residual, checks
finite outputs and an unchanged backbone, and conserves four total components.
Quantized restoration is approximate. This is not a learning benchmark.

The default test command covers core, packs and utilities. Parked research runs
only when requested with `uv run --locked pytest -q tests/research`. Use
`uv run --locked pytest -q tests` for every suite, especially after changes to
shared numerical/state machinery. [Suite guidance](../tests/README.md).

Tests exercise synthetic models and local artifacts without downloading a base
checkpoint. MLX execution needs access to Apple Metal; a sandbox failing to
initialize Metal is an environment failure, not a numerical test result.

## Pack workflow

Use an existing compatible local checkpoint, choose its loader and projection
names, and prepare disjoint JSONL splits. These example paths are placeholders;
this workflow is not run by the default demo. Keep training/evaluation loss mode,
sequence length and chat-template settings aligned.

```sh
uv run --locked --extra packs packs create \
  --name domain-demo --base /path/to/checkpoint --loader auto \
  --layers attn.q_proj,attn.k_proj,attn.v_proj --rank 8 \
  --initialization component-v1 --seed 42 --steps 100 \
  --data data/domain_train.jsonl
uv run --locked packs inspect --name domain-demo
uv run --locked --extra packs packs apply \
  --name domain-demo --base /path/to/checkpoint --dry-run
uv run --locked --extra packs packs eval \
  --base /path/to/checkpoint --pack domain-demo \
  --data-path data/domain_eval.jsonl --out out/domain_eval.json
uv run --locked packs proof \
  --base /path/to/checkpoint --pack domain-demo --domain domain-demo \
  --train-data data/domain_train.jsonl --eval-data data/domain_eval.jsonl \
  --eval-report out/domain_eval.json --out out/domain_proof.json
```

A proof pass is conditional on measured improvement and the report's other
checks; this recipe does not promise one. See [integrity requirements](../docs/experiment_integrity.md).
Use `packs bakeoff --help` for orchestration and `packs route --help` for optional
domain routing. The CLI is the single pack entry point. A fresh `uv run` without
an extra returns the environment to the minimal dependency set; specify `--extra
packs` on each model-backed invocation.

## Optional utilities

| Need | Invocation |
| --- | --- |
| Model loaders/training | `uv run --locked --extra packs packs …` |
| Checkpoint download/compression | `uv run --locked --extra compress python scripts/compress_llm_mlx.py …` |
| Dataset extraction through `datasets` | `uv run --locked --extra data python scripts/text_to_sql_extract.py …` |
| Fault-code extraction through `datasets` | `uv run --locked --extra data python scripts/fault_codes_extract.py …` |
| Rank/byte accounting on local packs | `uv run --locked packs rank-ledger --help` |

The fault-code dataset-viewer path uses the standard library and can run without
the data extra. Generated examples preserve source/license metadata; see
[the data README](../data/README.md). No symbolic algebra dependency is required.

## Research handoff

Start with the [proposed baseline diagnostic](dsn/dsn-20260905-baseline-validity-diagnostic.md),
not another controller sweep. Its output should decide whether the next change
belongs in data coverage, numerical conditioning or factorized optimization.
Keep this diagnosis distinct from an equal-budget controller comparison.

Historical development replay is documented with its artifacts in
[gradient-agreement results](research/gradient-agreement/development-results.md).
Its validity failure is expected; do not lower thresholds to make it pass.
Snapshots preserve the old source and lockfile. Replaying today's source is a
new run, not a byte-identical reproduction of a historical environment.

The [parking lot](parking-lot.md) names return triggers. Do not resume a parked
track or consume reserved seeds merely because its script remains runnable.
For new scope, update the DSN, [DSN log](dsn/dsn-log.md) and
[canonical decision index](decisions.md) together.
