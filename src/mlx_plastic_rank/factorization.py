"""Compact exact and randomized low-rank matrix factorizations.

QR reductions keep SVD workspace bounded by the smaller dimension or sketch
width. Projection and reconstruction are separate from layer state and factor
quantization; public functions remain re-exported from ``lowrank``.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Tuple

import mlx.core as mx


def _matrix_shape(A: mx.array, r: int) -> Tuple[int, int]:
    """Validate the shared real-matrix/rank contract before conversion."""
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix")
    m, n = A.shape
    if not (1 <= r <= min(m, n)):
        raise ValueError(f"r must be in [1, {min(m, n)}], got {r}")
    if mx.issubdtype(A.dtype, mx.complexfloating):
        raise ValueError("Factorization requires real-valued input")
    return m, n


def _cpu_stream():
    """Return a CPU stream context if MLX exposes streams; else no-op.

    In MLX, streams are context managers (constructors like ``mx.array`` do
    not accept a ``stream=...`` kwarg). Wrap compute ops in this context to
    keep execution on CPU when desired.
    """
    return mx.stream(mx.cpu) if hasattr(mx, "stream") else nullcontext()


def _svd_topk_direct(A: mx.array, r: int) -> Tuple[mx.array, mx.array, mx.array]:
    """Compute top-``r`` factors using a direct SVD of ``A``.

    Returns ``(U_r, s_r, Vh_r)`` where shapes are ``(m,r)``, ``(r,)``,
    and ``(r,n)`` respectively.
    """
    # MLX SVD returns full square U and Vh. Reduce the longer dimension
    # first so those allocations are at most min(m, n) squared.
    with _cpu_stream():
        m, n = A.shape
        if m > n:
            Q, core = mx.linalg.qr(A)
            U, s, Vh = mx.linalg.svd(core)
            U_r, Vh_r = Q @ U[:, :r], Vh[:r, :]
        elif n > m:
            Q, core = mx.linalg.qr(A.T)
            U, s, Vh = mx.linalg.svd(core.T)
            U_r, Vh_r = U[:, :r], Vh[:r, :] @ Q.T
        else:
            U, s, Vh = mx.linalg.svd(A)
            U_r, Vh_r = U[:, :r], Vh[:r, :]
        s_r = s[:r]
        mx.eval(U_r, s_r, Vh_r)
    return U_r, s_r, Vh_r


def _orthonormal_basis(A: mx.array) -> mx.array:
    """Materialize a reduced QR basis on the CPU."""
    with _cpu_stream():
        Q, _ = mx.linalg.qr(A)
        mx.eval(Q)
    return Q


def _matmul_columns(A: mx.array, B: mx.array, chunk: int) -> mx.array:
    """Project a skinny matrix in bounded column chunks."""
    blocks = []
    for start in range(0, B.shape[1], chunk):
        block = A @ B[:, start : start + chunk]
        mx.eval(block)
        blocks.append(block)
    result = mx.concatenate(blocks, axis=1)
    mx.eval(result)
    return result


def randomized_svd(
    A: mx.array,
    r: int,
    p: int = 8,
    q: int = 1,
    device_stream=None,
    chunk_k: int | None = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Randomized SVD for memory-friendly top-``r`` factorization.

    Parameters
    - A: real 2D MLX array of shape (m, n); computation uses float32
    - r: target rank (1..min(m, n))
    - p: oversampling (default 8)
    - q: power iterations (default 1); set to 0 to skip

    Returns
    - U_r, S_r, Vh_r such that ``A ≈ U_r @ diag(S_r) @ Vh_r``.

    Algorithm
    1) Project Gaussian probes into a sketch with ``k=min(r+p,m,n)`` columns.
    2) Find its reduced QR basis and reorthogonalize after each multiplication
       in the ``q`` power iterations to avoid amplification overflow.
    3) Project ``B = Q.T @ A`` and QR-factor ``B.T``. Only its ``k×k`` core
       needs an SVD; lift the resulting factors through the two QR bases.

    Additional workspace is O((m+n)k + k²), excluding the input, an optional
    fp32 input conversion, and backend allocator caches. QR/SVD run on CPU;
    ``device_stream`` selects the projection/lifting device. ``chunk_k`` caps
    projection columns per multiplication, not total memory or kernel duration.
    """
    m, n = _matrix_shape(A, r)

    k = int(min(m, n, max(r, 1) + max(p, 0)))
    ctx = device_stream or _cpu_stream()
    chunk = min(k, 64 if chunk_k is None or chunk_k <= 0 else int(chunk_k))
    with ctx:
        A = A.astype(mx.float32)
        # A full-width sketch cannot reduce the problem. Exact compact SVD
        # avoids pointless random projections and repeated full-width QR.
        if k == min(m, n):
            return _svd_topk_direct(A, r)
        blocks = []
        for start in range(0, k, chunk):
            omega = mx.random.normal((n, min(k - start, chunk)))
            block = A @ omega
            mx.eval(block)
            blocks.append(block)
        sketch = mx.concatenate(blocks, axis=1)
        mx.eval(sketch)
    del blocks, block, omega
    Q = _orthonormal_basis(sketch)
    del sketch
    for _ in range(max(0, q)):
        with ctx:
            projection = _matmul_columns(A.T, Q, chunk)
        right_basis = _orthonormal_basis(projection)
        del projection
        with ctx:
            projection = _matmul_columns(A, right_basis, chunk)
        Q = _orthonormal_basis(projection)
        del projection, right_basis
    with ctx:
        B = _matmul_columns(A.T, Q, chunk).T
    Ub, s, Vh = _svd_topk_direct(B, r)
    with ctx:
        U = Q @ Ub
        mx.eval(U, s, Vh)
    return U, s, Vh


def factorized_lowrank(A: mx.array, r: int) -> Tuple[mx.array, mx.array, mx.array]:
    """Return the top-``r`` factorization ``(U, S, Vh)`` of ``A``.

    Uses a direct SVD for small matrices and randomized SVD above a size
    threshold to reduce temporary workspace and avoid large GPU allocations.
    """
    m, n = _matrix_shape(A, r)

    # Heuristic: switch to rSVD when either dimension is large.
    # This keeps tests deterministic on small shapes while improving stability
    # and memory behavior for large matrices.
    threshold = 1024
    if max(m, n) >= threshold:
        U_r, s_r, Vh_r = randomized_svd(A, r, p=8, q=1)
    else:
        with _cpu_stream():
            U_r, s_r, Vh_r = _svd_topk_direct(A.astype(mx.float32), r)
    return U_r, s_r, Vh_r


def svd_lowrank_randomized(
    A: mx.array,
    r: int,
    n_oversamples: int = 8,
    n_iter: int = 1,
    device_stream=None,
    chunk_k: int | None = None,
) -> mx.array:
    """Return ``A``'s rank-``r`` approximation using randomized SVD.

    Convenience wrapper that reconstructs ``A_r`` directly. If ``device_stream``
    is provided, heavy ops execute inside that context; otherwise a CPU stream
    is used when available.
    """
    U, S, Vh = randomized_svd(
        A,
        r,
        p=n_oversamples,
        q=n_iter,
        device_stream=device_stream,
        chunk_k=chunk_k,
    )
    return (U * S[None, :]) @ Vh


def svd_lowrank(A: mx.array, r: int) -> mx.array:
    """Reconstruct ``(U * S) @ Vh`` using the automatic factorization path.

    The direct path is optimal in Frobenius norm; the randomized path is an
    approximation whose accuracy depends on the spectrum and sampled subspace.
    """
    U_r, s_r, Vh_r = factorized_lowrank(A, r)
    A_approx = (U_r * s_r[None, :]) @ Vh_r
    return A_approx
