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
| `src/mlx_plastic_rank/` | Installed low-rank and fixed-allocation pack implementation |
| `research/` | Repository-only experiments; excluded from the installed package |
| `plastic_rank.py` | Bounded local lifecycle demo |
| `tests/core/`, `tests/packs/`, `tests/tools/` | Default regression suite |
| `tests/research/` | Parked experiment regressions, run explicitly |
| `codex/` | Decisions, evidence and research history |

See [test commands](tests/README.md) and [contributor guidance](CONTRIBUTING.md).

## Work with packs

The [runbook](codex/runbook.md#pack-workflow) covers training, inspection,
application, held-out evaluation and proof reports using an existing compatible
checkpoint. Specify a fixed rank or explicit rank map and separate training/evaluation
data. New packs use matched `component-v1` initialization by default.

Model-backed commands use `uv run --locked --extra packs packs …`. Dataset
extraction and checkpoint compression use the optional `data` and `compress`
extras. The default environment contains MLX, NumPy and SafeTensors.
See [data guidance](data/README.md) and [artifact integrity](docs/experiment_integrity.md).

## Research status

Pop's papers provide exact subspace accounting for polynomials of one common
operator. They do not supply a learning rule or establish useful rank allocation.
The [mathematical contract](codex/dsn/dsn-20260905-pop-mathematical-contract.md)
defines that boundary and the controls a future learning claim would need.

The [dense baseline diagnostic](codex/research/baseline-diagnostic-results.md)
passed on all five stored development seeds; all broken-pairing controls failed.
The next question concerns factorized capacity, optimization and allocation.
Adaptive controllers and selector studies remain in the
[parked research register](codex/parking-lot.md), with source and replay guidance
in [research/](research/README.md).

The [consolidation DSN](codex/dsn/dsn-20260905-prototype-consolidation.md) defines
current scope. The [decision index](codex/decisions.md) preserves the history.
Dataset/model attribution and redistribution expectations are in [NOTICE.md](NOTICE.md).
