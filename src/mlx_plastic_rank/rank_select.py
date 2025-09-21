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


def _numerical_rank(M: ArrayLike, eps: float = 1e-6) -> int:
    s = np.linalg.svd(_to_numpy_f32(M), compute_uv=False)
    if s.size == 0:
        return 0
    thr = float(s[0] * eps)
    return int((s > thr).sum())


def theorem_guided_rank(
    A: ArrayLike, target_compression: float = 0.9, eps: float = 1e-6, tol: float = 1.0
) -> Tuple[int, float]:
    """Select a rank guided by a simple polynomial rank identity.

    Steps
    - Compute singular values of A and choose r0 as the smallest r s.t.
      sum_{i<=r} s_i^2 / sum s_i^2 >= target_compression.
    - Evaluate polynomials via Horner on the rank-r truncated matrix A_r:
        f(x)=x, g(x)=x^2-x, D(x)=x, M(x)=x(x-1).
    - Compute numerical ranks with tolerance eps and residual
      res = |(rf + rg) - (rd + rm)|.
    - If res > tol, increment r until residual <= tol or r hits full rank.

    Returns (r, residual).
    """
    A_np = _to_numpy_f32(A)
    if A_np.ndim != 2 or A_np.shape[0] != A_np.shape[1]:
        raise ValueError("theorem_guided_rank expects a square 2D matrix")

    # singular values of A
    s = np.linalg.svd(A_np, compute_uv=False)
    s2 = s * s
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
        # NumPy truncated SVD for reconstruction on CPU
        U, S, Vh = np.linalg.svd(A_np, full_matrices=False)
        A_r = (U[:, :r] * S[:r][None, :]) @ Vh[:r, :]
        I = np.eye(A_np.shape[0], dtype=np.float32)
        # Horner-evaluated polynomials
        fA = A_r
        gA = A_r @ (A_r - I)  # x(x-1)
        DA = A_r
        MA = A_r @ (A_r - I)

        rf = _numerical_rank(fA, eps)
        rg = _numerical_rank(gA, eps)
        rd = _numerical_rank(DA, eps)
        rm = _numerical_rank(MA, eps)
        residual = abs((rf + rg) - (rd + rm))
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
