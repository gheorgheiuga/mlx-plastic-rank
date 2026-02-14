"""Rank selection heuristics and stability metrics.

This module purposely uses NumPy for singular-value computations so rank
selection never triggers large MLX/Metal allocations. The actual low‑rank
reconstruction during compression can still use MLX.
"""
from typing import Tuple, Union

import mlx.core as mx  # for type annotations only
import numpy as np

ArrayLike = Union[np.ndarray, mx.array]

def _to_numpy_f32(A: ArrayLike) -> np.ndarray:
    if isinstance(A, np.ndarray):
        return A.astype(np.float32, copy=False)
    # Fall back to array protocol
    return np.array(A, dtype=np.float32)


def stable_rank(A: ArrayLike, eps: float = 1e-6) -> float:
    """Compute the stable rank of a matrix A.

    stable_rank(A) = ||A||_F^2 / (sigma_max(A)^2 + eps)

    - Uses NumPy SVD without computing U/V for efficiency.
    - Adds a small `eps` to guard degenerate cases.
    """
    A_np = _to_numpy_f32(A)
    s = np.linalg.svd(A_np, compute_uv=False)
    fro2 = float(np.square(A_np).sum())
    denom = float(s[0] * s[0] + eps)
    return float(fro2 / denom)


def theorem_guided_rank(
    A: ArrayLike, target_compression: float = 0.9, eps: float = 1e-6, tol: float = 1.0
) -> Tuple[int, float]:
    """Select a rank guided by spectral energy and reconstruction residual.

    Procedure
    - Compute SVD once and pick r0 as the smallest rank whose cumulative
      spectral energy reaches ``target_compression``.
    - Reconstruct ``A_r`` and compute a relative Frobenius residual:
      ``res = ||A - A_r||_F / (||A||_F + eps)``.
    - If ``res > tol``, increment ``r`` until residual is within tolerance or
      full rank is reached.

    Returns ``(r, residual)`` where ``residual`` is the final relative error.
    """
    A_np = _to_numpy_f32(A)
    if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
        raise ValueError("theorem_guided_rank expects a square 2D matrix")

    U, S, Vh = np.linalg.svd(A_np, full_matrices=False)
    s2 = S * S
    total = float(s2.sum())
    cdf = np.cumsum(s2) / (total + 1e-12) if total != 0 else np.array([], dtype=np.float32)
    if cdf.size == 0:
        r = 1
    else:
        cvals = cdf.tolist()
        r0 = next((i + 1 for i, v in enumerate(cvals) if v >= float(target_compression)), min(A_np.shape))
        r = max(1, min(int(r0), min(A_np.shape)))

    full = min(A_np.shape)
    residual = float("inf")
    while r <= full:
        A_r = (U[:, :r] * S[:r][None, :]) @ Vh[:r, :]
        num = float(np.linalg.norm(A_np - A_r, ord="fro"))
        den = float(np.linalg.norm(A_np, ord="fro")) + float(eps)
        residual = num / den
        if residual <= tol:
            break
        r += 1

    return r, float(residual)


def _energy_rank(A: ArrayLike, target_compression: float) -> int:
    A_np = _to_numpy_f32(A)
    s = np.linalg.svd(A_np, compute_uv=False)
    s2 = s * s
    total = float(s2.sum())
    if total == 0:
        return 1
    cdf = np.cumsum(s2) / (total + 1e-12)
    vals = cdf.tolist()
    r0 = next((i + 1 for i, v in enumerate(vals) if v >= float(target_compression)), min(A_np.shape))
    return max(1, min(int(r0), min(A_np.shape)))


def choose_rank(
    A: ArrayLike,
    target_compression: float,
    strategy: str,
    eps: float = 1e-6,
) -> Tuple[int, float]:
    """Choose a rank according to a strategy.

    Returns a tuple `(r, residual)`.
    - For `strategy="stable"`, `r` is chosen by energy cutoff and `residual` is `-1.0` as a sentinel.
    - For `strategy="theorem"`, `r` and `residual` come from `theorem_guided_rank`.
    """
    if strategy == "stable":
        r = _energy_rank(A, target_compression)
        return r, -1.0
    elif strategy == "theorem":
        return theorem_guided_rank(A, target_compression, eps)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
