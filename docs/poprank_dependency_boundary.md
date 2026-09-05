# PopRank dependency boundary

The default environment contains MLX, NumPy and SafeTensors. Model loaders,
checkpoint-download helpers and Hugging Face dataset extraction belong to the
`packs`, `compress` and `data` extras respectively. SymPy is unused and removed.
See [DSN-20260905-04](../codex/dsn/dsn-20260905-prototype-consolidation.md).

## Runtime tensor work

Use MLX arrays for training, evaluation, adapter inspection, rank algebra and
factorization. Where needed, compact QR/SVD runs on the MLX CPU stream; that is
still MLX, not a NumPy fallback. Keep metadata, hashes, byte accounting and
rank-map validation in the standard library where tensors are unnecessary.

NumPy remains at the SafeTensors serialization boundary in `packs/io.py` and
`packs/manager.py`. Convert loaded arrays into MLX for runtime measurements.
Offline historical research/reference utilities and small synthetic tests may
use NumPy explicitly. They do not make NumPy conversion appropriate inside the
training path. The proposed dense baseline diagnostic is such an offline
reference, not a new runtime implementation.

## Enforced scope

`tests/core/test_dependency_boundary.py` parses imports in named core modules,
including factorization and provenance, and fails if any listed module is
missing. It also verifies that importing the advertised core and constructing
the pack CLI does not load optional model/dataset packages or parked controllers.
The two SafeTensors boundaries and the legacy polynomial probe carry local
NumPy justifications.

This is a focused regression guard, not an assertion that every historical
research module is MLX-only. Keep optional imports lazy so ordinary tests,
local pack inspection and the lifecycle demo work after `uv sync --locked`.
