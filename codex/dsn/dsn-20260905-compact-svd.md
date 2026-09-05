# Decision Support Note

**ID:** DSN-20260905-02  
**Title:** Compact SVD workspace and focused structural cleanup  
**Date:** 2026-09-05  
**Status:** Experimental  
**Evidence Status:** Local numerical and allocation behavior verified; no model-quality claim  
**Canonical decision:** ADR-0013 in `codex/decisions.md`

## Problem

The randomized path called full-vector MLX SVD on both a tall sketch and a wide
projection. On the locked runtime this allocated square factors along the
original dimensions, defeating the intended workspace bound. CPU compression
and GPU error recovery could also switch a requested randomized decomposition
to full SVD. Factorization, layer state and duplicated sleeper handling lived
together in `lowrank.py`.

## Implementation

Reduce both sides with compact QR and decompose only the sketch-width square
core. Reorthogonalize each power-iteration multiplication, compute in float32,
and use compact exact SVD when the sketch spans the smaller dimension. Preserve
the existing public imports while moving decomposition into `factorization.py`.
Keep randomized compression randomized on CPU and during GPU recovery. Share
one component-parking implementation between the two pruning policies.

## Evidence

`codex/evidence/svd_workspace_20260905.json` records 18 isolated processes:
three matrix shapes × three paired seeds × old/new implementations. Rank 32,
oversampling 8 and one power iteration were fixed. Input hashes match; source,
generator and dependency hashes are captured. Median process peak RSS decreased
by 50.24% for 4096 × 512, 49.63% for 512 × 4096, and 29.01% for 2048 × 2048.
Relative reconstruction errors differed by at most 2.39e-7. This is a small
synthetic engineering comparison, not a whole-model memory or speed guarantee.

Regression tests constrain SVD allocation requests and exercise numerical
accuracy, orthogonality, high-scale rank-deficient input, zero/low-precision
input, full rank, GPU projections and simulated GPU failure recovery. Existing
export and non-LIFO sleep/wake tests protect compatibility. The full pytest,
Ruff, mypy and documented demo checks remain required before handoff.

## Limits and next validation

Near-full rank still needs workspace proportional to the smaller dimension;
float32 input conversion, dense reconstruction and allocator caches consume
additional memory. QR changes the numerical/RNG protocol, so historical
compressed tensors are not promised to be byte-identical. Details and exact
reproduction commands are in `docs/svd_factorization.md`.

Keep this DSN Experimental. A model-scale fit claim requires an end-to-end
checkpoint measurement including rank selection, reconstruction and residency.
A Pop Rank quality claim still requires the independent controls and untouched
seeds defined in ADR-0011/ADR-0012. Lower factorization memory does not reverse
the failed research promotion gates or authorize a new large-model study.
