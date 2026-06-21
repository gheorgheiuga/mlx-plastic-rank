# PopRank Dependency Boundary

PopRank runtime and research artifacts should stay MLX-native.

## Runtime Tensor Work

Use `mlx.core` / MLX arrays for tensor operations in:

- training and evaluation;
- adapter inspection and dynamic rank discovery;
- rank ledgers and spectral/rank probes;
- causal ablation scoring;
- runtime or MLX memory measurement.

Do not convert MLX arrays to NumPy for convenience in these paths. If MLX SVD,
QR, or eigensolvers are needed, use the explicit CPU stream where required by
MLX rather than switching libraries.

## Metadata And Accounting

Use Python standard library types for non-tensor work:

- rank-budget formulas;
- rank-map validation;
- device-profile checks;
- JSON, Markdown, and CSV artifacts;
- byte, slack, and summary-table calculations.

These paths do not need NumPy.

## NumPy Exceptions

NumPy is allowed only for:

- tests with small synthetic arrays;
- `safetensors.numpy` pack serialization/deserialization boundaries;
- optional offline utilities that are clearly marked non-runtime;
- temporary migration code with a local TODO explaining why it remains.

The current pack IO helpers use `safetensors.numpy`, so NumPy arrays are still
the file-format boundary. Core tensor math should convert from that boundary
into MLX before doing measurements or rank algebra.
