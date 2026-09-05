# Compact SVD and measured memory use

The factorization implementation now lives in
`src/mlx_plastic_rank/factorization.py`. Existing imports from `lowrank` and the
package root remain supported. Layer state and quantization remain in `lowrank`;
both pruning methods share one sleeper-storage operation.

## Allocation defect and replacement

On the locked MLX 0.31.2 runtime, an SVD of a 4096 × 40 sketch returned a
4096 × 4096 left factor. The second SVD also returned a square right factor along
the other input dimension. Slicing afterwards did not prevent those allocations.
The benchmark records the actual requested/output shapes.

The replacement uses reduced QR bases on both sides and applies SVD only to a
`k × k` core, where `k = min(r + p, m, n)` for nonnegative oversampling `p`.
Power iterations reorthogonalize between multiplications. QR and SVD run on CPU;
projections can run on CPU or GPU. The library interfaces are documented by
[MLX QR](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.qr.html)
and [MLX SVD](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.linalg.svd.html);
the shape and allocation claims here were verified on the installed version.

For a genuinely low-rank sketch, extra algorithmic workspace is
`O((m+n)k + k²)`. That excludes the original matrix, any float32 conversion,
allocator caches and a later dense reconstruction. If the sketch spans the
smaller dimension, compact exact SVD avoids redundant random projections.
Full-rank requests still need workspace proportional to the full smaller dimension.

## Behavior and compatibility

- Public factor shapes remain `U: (m,r)`, `S: (r,)`, `Vh: (r,n)`.
- Real inputs are computed in float32, including float16 and bfloat16 inputs.
  Complex input is rejected instead of silently dropping imaginary values.
- Small matrices use exact compact SVD. The automatic large-matrix path remains
  approximate; its accuracy depends on the spectrum and sampled subspace.
- `chunk_k` controls projection width per multiplication. It is not a total
  memory limit or a guarantee against a device timeout.
- Compression with `--svd randomized --device cpu` now honors the randomized
  algorithm. A GPU failure retries that same algorithm on CPU. Previously both
  paths could silently switch to a full decomposition.
- `--svd full` still requests exact reduced NumPy SVD on CPU.
- New QR bases, stabilized iterations and the full-width shortcut change the
  numerical/RNG protocol. Historical compressed outputs are not promised to be
  byte-identical. This change does not retrain or rewrite any historical pack.

## Local benchmark

The comparison uses rank 32, oversampling 8, one power iteration, and seeds
42–44. Every run is a separate process using a locally generated float32 Gaussian
matrix and CPU projections. Paired input hashes match. The table shows medians
over the three seeds; memory is captured before dense reconstruction.

| Matrix | Process peak before → after | RSS high-water increase before → after | Time before → after |
| --- | ---: | ---: | ---: |
| 4096 × 512 | 133.67 → 66.52 MiB | 70.64 → 3.83 MiB | 133.92 → 7.31 ms |
| 512 × 4096 | 132.38 → 66.67 MiB | 69.44 → 3.88 MiB | 76.01 → 8.22 ms |
| 2048 × 2048 | 116.77 → 82.89 MiB | 37.94 → 4.31 MiB | 55.23 → 8.81 ms |

Total process peaks decreased by 50.24%, 49.63%, and 29.01%, respectively. The
largest absolute difference in relative Frobenius reconstruction error was
`2.39e-7`. Separate controlled-spectrum tests compare accuracy against optimal
truncated SVD and check orthogonality, rank-deficient/zero matrices, near-full
rank, low-precision input and both projection devices.

RSS high-water increase is a difference between process maxima, not a direct
workspace allocation measurement. Process peaks include Python and imported
libraries. Timings are local observations on these synthetic matrices, not a
general speed guarantee. This does not establish that an entire LLM checkpoint,
rank-selection prepass, reconstruction or training run fits a device budget.

The [18-run evidence snapshot](../codex/evidence/svd_workspace_20260905.json)
contains all measurements, decomposition shapes, input identities, source hashes,
the locked dependency hash and environment versions. The baseline is commit
`1651a62c22c526427f0e890698dad912008f43be`; the updated implementation is identified
by its exact working-tree source hash.

Reproduce one current case:

```bash
uv run --locked python scripts/bench_svd_workspace.py \
  --m 4096 --n 512 --rank 32 --seed 42 \
  --out out/svd_current_tall_42.json
```

To compare the committed baseline in another fresh process:

```bash
mkdir -p out/svd_baseline
git show 1651a62c22c526427f0e890698dad912008f43be:src/mlx_plastic_rank/lowrank.py > out/svd_baseline/lowrank.py
uv run --locked python scripts/bench_svd_workspace.py \
  --implementation-file out/svd_baseline/lowrank.py \
  --m 4096 --n 512 --rank 32 --seed 42 \
  --out out/svd_baseline/tall_42.json
```

Repeat with seeds 43/44 and the other shapes to reproduce the comparison matrix.
Use `scripts/bench_memory.py` for factor-storage estimates; its byte counts are
not runtime process-memory measurements.
