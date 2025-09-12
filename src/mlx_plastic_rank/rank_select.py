"""Rank selection heuristics and stability metrics."""
from typing import Tuple, Union
import mlx.core as mx
from .lowrank import svd_lowrank

ArrayLike = Union[mx.array]


def stable_rank(A: ArrayLike, eps: float = 1e-6) -> float:
    """Compute the stable rank of a matrix A.

    stable_rank(A) = ||A||_F^2 / (sigma_max(A)^2 + eps)

    - Uses MLX SVD without computing U/V for efficiency.
    - Adds a small `eps` to guard degenerate cases.
    """
    s = mx.linalg.svd(A, compute_uv=False, stream=mx.cpu)
    fro2 = mx.square(A).sum()
    denom = s[0] * s[0] + eps
    return float(fro2 / denom)


def _numerical_rank(M: mx.array, eps: float = 1e-6) -> int:
    s = mx.linalg.svd(M, compute_uv=False, stream=mx.cpu)
    if s.size == 0:
        return 0
    thr = s[0] * eps
    return int(mx.sum(s > thr))


def theorem_guided_rank(
    A: mx.array, target_compression: float = 0.9, eps: float = 1e-6, tol: float = 1.0
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
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("theorem_guided_rank expects a square 2D matrix")

    # singular values of A
    s = mx.linalg.svd(A, compute_uv=False, stream=mx.cpu)
    s2 = s * s
    total = mx.sum(s2)
    cdf = mx.cumsum(s2) / (total + 1e-12)
    if cdf.size == 0:
        r = 1
    else:
        cvals = cdf.tolist()
        r0 = next((i + 1 for i, v in enumerate(cvals) if v >= float(target_compression)), min(A.shape))
        r = max(1, min(int(r0), min(A.shape)))

    full = min(A.shape)
    residual = float("inf")
    while r <= full:
        A_r = svd_lowrank(A, r)
        I = mx.eye(A.shape[0])
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


def _energy_rank(A: mx.array, target_compression: float) -> int:
    s = mx.linalg.svd(A, compute_uv=False, stream=mx.cpu)
    s2 = s * s
    total = mx.sum(s2)
    if total == 0:
        return 1
    cdf = mx.cumsum(s2) / (total + 1e-12)
    vals = cdf.tolist()
    r0 = next((i + 1 for i, v in enumerate(vals) if v >= float(target_compression)), min(A.shape))
    return max(1, min(int(r0), min(A.shape)))


def choose_rank(
    A: mx.array,
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
