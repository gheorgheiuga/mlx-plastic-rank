# mlx-plastic-rank

Local Apple MLX tools for low-rank layers and portable LoRA adapter packs.
The maintained prototype covers factor pruning/restoration, compact SVD, adapter
training and export, and evaluation with traceable artifacts.

## Start locally

Use Apple Silicon and `uv`. Python 3.13 and dependencies are pinned; setup uses
a project-local `.venv` and requires no shell activation.

```sh
uv sync --locked
uv run --locked python plastic_rank.py
uv run --locked pytest -q
```

The demo creates four residual components, parks two and restores them. It checks
component counts, an unchanged base and quantization error. No model or dataset
download is needed. Restoration is approximate; the demo is not a learning result.

## Repository layout

| Location | Purpose |
| --- | --- |
| `src/mlx_plastic_rank/` | Low-rank and pack implementation |
| `plastic_rank.py` | Bounded local lifecycle demo |
| `tests/core/`, `tests/packs/`, `tests/tools/` | Default regression suite |
| `tests/research/` | Parked experiment regressions, run explicitly |
| `codex/` | Decisions, evidence and research history |

See [test commands](tests/README.md) and [contributor guidance](CONTRIBUTING.md).

## Work with packs

The [runbook](codex/runbook.md#pack-workflow) covers training, inspection,
application, held-out evaluation and proof reports using an existing compatible
checkpoint. Begin with fixed rank and separate training/evaluation data.

Model-backed commands use `uv run --locked --extra packs packs …`. Dataset
extraction and checkpoint compression use the optional `data` and `compress`
extras. The default environment contains MLX, NumPy and SafeTensors.
See [data guidance](data/README.md) and [artifact integrity](docs/experiment_integrity.md).

## Research status

Adaptive capacity migration and theorem-based rank selection remain unvalidated.
Their source and evidence are retained in the [parked research register](codex/parking-lot.md).
The next proposed experiment is one [baseline diagnostic](codex/dsn/dsn-20260905-baseline-validity-diagnostic.md)
on existing data; further controller and large-model work depends on its outcome
and the relevant research gates.

The [consolidation DSN](codex/dsn/dsn-20260905-prototype-consolidation.md) defines
current scope. The [decision index](codex/decisions.md) preserves the history.
Dataset/model attribution and redistribution expectations are in [NOTICE.md](NOTICE.md).
