"""Low-rank layers and SVD-based approximations.

This module provides two SVD paths for top-``r`` approximations:

- Direct SVD via ``mx.linalg.svd`` (good for small to medium matrices).
- Randomized SVD (rSVD) that bounds working memory and plays nicer with
  MLX streams by reducing the size of the expensive SVD to a skinny matrix.

The selection is automatic in :func:`factorized_lowrank` based on a size
threshold. All heavy linear-algebra ops run inside a CPU stream context
when available to avoid large Metal allocations on macOS GPUs.
"""
from __future__ import annotations
from typing import Tuple, Dict
from contextlib import nullcontext
import math
import mlx.core as mx
import mlx.nn as nn

from .utils import quantise, dequantise


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
    with _cpu_stream():
        U, s, Vh = mx.linalg.svd(A)
    return U[:, :r], s[:r], Vh[:r, :]


def randomized_svd(
    A: mx.array,
    r: int,
    p: int = 8,
    q: int = 1,
    device_stream=None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Randomized SVD for memory-friendly top-``r`` factorization.

    Parameters
    - A: 2D MLX array of shape (m, n)
    - r: target rank (1..min(m, n))
    - p: oversampling (default 8)
    - q: power iterations (default 1); set to 0 to skip

    Returns
    - U_r, S_r, Vh_r such that ``A ≈ U_r @ diag(S_r) @ Vh_r``.

    Algorithm
    1) Draw a Gaussian test matrix ``Omega ∈ R^{n×k}``, where ``k=r+p``.
    2) Form ``Y = A @ Omega`` and (optionally) perform ``q`` power iterations
       ``Y = A @ (A.T @ Y)`` for spectral polishing.
    3) Orthonormalize columns of ``Y`` via a thin SVD; let ``Q`` be its left
       singular vectors with ``k`` columns.
    4) Compute ``B = Q.T @ A`` (size ``k×n``) and SVD it as ``U_b, S, Vh``.
    5) Lift left singular vectors: ``U = Q @ U_b`` and slice top-``r``.
    """
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix")
    m, n = A.shape
    if not (1 <= r <= min(m, n)):
        raise ValueError(f"r must be in [1, {min(m, n)}], got {r}")

    k = int(min(m, n, max(r, 1) + max(p, 0)))

    ctx = device_stream or _cpu_stream()
    # 1) random projection and optional power iterations on chosen device
    with ctx:
        Omega = mx.random.normal((n, k))
        Y = A @ Omega  # (m, k)
        for _ in range(max(0, q)):
            Y = A @ (A.T @ Y)
    # 3) SVD(Y) on CPU to avoid GPU unsupported op / huge workspace
    with _cpu_stream():
        Uy, _, _ = mx.linalg.svd(Y)
    with ctx:
        Q = Uy[:, :k]  # (m, k)
        # 4) project and do SVD on CPU
        B = Q.T @ A  # (k, n)
    # 4b) SVD(B) on CPU
    with _cpu_stream():
        Ub, s, Vh = mx.linalg.svd(B)
    with ctx:
        # 5) lift
        U = Q @ Ub  # (m, k)

    return U[:, :r], s[:r], Vh[:r, :]


def factorized_lowrank(A: mx.array, r: int) -> Tuple[mx.array, mx.array, mx.array]:
    """Return the top-``r`` factorization ``(U, S, Vh)`` of ``A``.

    Uses a direct SVD for small matrices and randomized SVD above a size
    threshold to reduce temporary workspace and avoid large GPU allocations.
    """
    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix")
    m, n = A.shape
    if not (1 <= r <= min(m, n)):
        raise ValueError(f"r must be in [1, {min(m, n)}], got {r}")

    # Heuristic: switch to rSVD when either dimension is large.
    # This keeps tests deterministic on small shapes while improving stability
    # and memory behavior for large matrices.
    threshold = 1024
    if max(m, n) >= threshold:
        U_r, s_r, Vh_r = randomized_svd(A, r, p=8, q=1)
    else:
        U_r, s_r, Vh_r = _svd_topk_direct(A, r)
    return U_r, s_r, Vh_r


def svd_lowrank_randomized(
    A: mx.array,
    r: int,
    n_oversamples: int = 8,
    n_iter: int = 1,
    device_stream=None,
) -> mx.array:
    """Return ``A``'s rank-``r`` approximation using randomized SVD.

    Convenience wrapper that reconstructs ``A_r`` directly. If ``device_stream``
    is provided, heavy ops execute inside that context; otherwise a CPU stream
    is used when available.
    """
    U, S, Vh = randomized_svd(A, r, p=n_oversamples, q=n_iter, device_stream=device_stream)
    return (U * S[None, :]) @ Vh


def svd_lowrank(A: mx.array, r: int) -> mx.array:
    """Reconstruct the best rank-`r` approximation of `A` via SVD.

    Uses `factorized_lowrank` and returns `U.T @ diag(S) @ Vh`.
    """
    U_r, s_r, Vh_r = factorized_lowrank(A, r)
    # U from MLX is returned as (k, m) when sliced as above; ensure shapes
    # We expect A ≈ (U_r^T * diag(s_r)) @ Vh_r
    A_approx = (U_r * s_r[None, :]) @ Vh_r
    return A_approx


class RankLayer(nn.Module):
    """Linear layer with reversible low-rank residual factors.

    W = W0 + U @ diag(S) @ V

    - `W0` is the frozen backbone.
    - `U`, `S`, `V` are the learnable low-rank factors, reversible and prunable.
    - Pruned components are stored in `sleep_dict` as quantized tuples.
    """

    def __init__(self, weight: mx.array, bias: mx.array | None = None):
        super().__init__()
        self.W0 = mx.array(weight)
        out, inn = self.W0.shape
        self.max_rank = out
        self.U = mx.zeros((0, out))
        self.S = mx.zeros((0,))
        self.V = mx.zeros((0, inn))
        self.bias = mx.array(bias) if bias is not None else None
        self.sleep_dict = {}

    @property
    def rank(self) -> int:
        return self.S.shape[0]

    def __call__(self, x: mx.array) -> mx.array:
        W = self.W0
        if self.rank > 0:
            # Compose low-rank update as sum_i s_i u_i v_i
            # Shapes: U (r, out), V (r, inn) -> U.T @ (V * S[:, None]) => (out, inn)
            W = W + self.U.T @ (self.V * self.S[:, None])
        b = self.bias if self.bias is not None else 0
        return x @ W.T + b

    # ---------- plastic ops ----------
    def add_rank(self, k: int):
        k = min(k, self.max_rank - self.rank)
        if k <= 0:
            return
        out, inn = self.W0.shape
        U = mx.random.normal((k, out)) * (1 / math.sqrt(out))
        V = mx.random.normal((k, inn)) * (1 / math.sqrt(inn))
        S = mx.ones(k) * 1e-3
        self.U = mx.concatenate([self.U, U])
        self.V = mx.concatenate([self.V, V])
        self.S = mx.concatenate([self.S, S])

    def prune_rank(self, tol: float = 1e-4):
        if self.rank == 0:
            return
        U, S, V = self.U, self.S, self.V
        keep = S > tol
        sleep_mask = ~keep
        if sleep_mask.sum() == 0:
            return
        # store sleepers as quantized tuples
        for idx in mx.where(sleep_mask)[0].tolist():
            u, s, v = U[idx], S[idx], V[idx]
            q_u, mn_u, sc_u = quantise(u)
            q_v, mn_v, sc_v = quantise(v)
            self.sleep_dict[len(self.sleep_dict)] = (
                q_u,
                mn_u,
                sc_u,
                float(s),
                q_v,
                mn_v,
                sc_v,
            )
        # trim kept components
        self.U = U[keep]
        self.S = S[keep]
        self.V = V[keep]

    def wake_rank(self, idx: int):
        q_u, mn_u, sc_u, s, q_v, mn_v, sc_v = self.sleep_dict.pop(idx)
        u = dequantise(q_u, mn_u, sc_u)
        v = dequantise(q_v, mn_v, sc_v)
        self.U = mx.concatenate([self.U, u[None]])
        self.S = mx.concatenate([self.S, mx.array([s])])
        self.V = mx.concatenate([self.V, v[None]])

    def prune_to_rank(self, target_rank: int):
        """Prune smallest-|S| components until rank equals target_rank.

        Pruned components are stored in the sleep_dict via quantization.
        """
        if target_rank >= self.rank:
            return
        k_drop = int(self.rank - target_rank)
        # indices of k smallest |S|
        order = mx.argsort(mx.abs(self.S))
        drop_idx = order[:k_drop].tolist()
        mask_list = [True] * self.rank
        for idx in drop_idx:
            mask_list[idx] = False
        keep_mask = mx.array(mask_list)
        for idx in drop_idx:
            u, s, v = self.U[idx], self.S[idx], self.V[idx]
            q_u, mn_u, sc_u = quantise(u)
            q_v, mn_v, sc_v = quantise(v)
            self.sleep_dict[len(self.sleep_dict)] = (
                q_u,
                mn_u,
                sc_u,
                float(s),
                q_v,
                mn_v,
                sc_v,
            )
        keep_idx = [i for i, flag in enumerate(mask_list) if flag]
        if keep_idx:
            self.U = mx.concatenate([self.U[i][None] for i in keep_idx])
            self.S = mx.concatenate([self.S[i][None] for i in keep_idx])
            self.V = mx.concatenate([self.V[i][None] for i in keep_idx])
        else:
            out, inn = self.W0.shape
            self.U = mx.zeros((0, out))
            self.S = mx.zeros((0,))
            self.V = mx.zeros((0, inn))


class PlasticBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        # MLX expects (dim, num_heads)
        self.attn = nn.MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            RankLayer(mx.random.normal((2048, d_model)), mx.zeros(2048)),
            nn.ReLU(),
            RankLayer(mx.random.normal((d_model, 2048)), mx.zeros(d_model)),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.attn(x, x, x)[0]
        x = self.norm1(x + y)
        x = self.norm2(x + self.ff(x))
        return x


def _quantise_rows(X: mx.array, bits: int = 8):
    """Row-wise uniform quantization returning (q, mins, scales).

    - X: (rows, cols)
    - q: uint8 matrix same shape
    - mins, scales: shape (rows,)
    """
    x_min = X.min(axis=1)
    x_max = X.max(axis=1)
    denom = (2 ** bits - 1)
    scale = (x_max - x_min) / denom
    scale = mx.maximum(scale, mx.array(1e-12))
    q = ((X - x_min[:, None]) / scale[:, None]).round().astype(mx.uint8)
    return q, x_min.astype(mx.float32), scale.astype(mx.float32)


def _dequantise_rows(q: mx.array, mins: mx.array, scales: mx.array) -> mx.array:
    return q.astype(mx.float32) * scales[:, None] + mins[:, None]


def quantize_factors(U: mx.array, S: mx.array, Vh: mx.array, bits: int = 8) -> Dict[str, Tuple[mx.array, mx.array, mx.array]]:
    """Quantize SVD factors with row-wise quantization for U and Vh.

    Returns a dict with entries:
    - "U": (q_U, mins_U, scales_U)
    - "S": (q_S, min_S, scale_S)
    - "Vh": (q_Vh, mins_Vh, scales_Vh)

    S is quantized with a single (min, scale) over the vector.
    """
    # Ensure 2D factors
    if U.ndim != 2 or Vh.ndim != 2 or S.ndim != 1:
        raise ValueError("Shapes must be U:(m,r), S:(r,), Vh:(r,n)")

    qU, minU, scU = _quantise_rows(U, bits)
    qVh, minVh, scVh = _quantise_rows(Vh, bits)

    s_min = S.min()
    s_max = S.max()
    denom = (2 ** bits - 1)
    s_scale = (s_max - s_min) / denom
    s_scale = mx.maximum(s_scale, mx.array(1e-12))
    qS = ((S - s_min) / s_scale).round().astype(mx.uint8)

    return {
        "U": (qU, minU, scU),
        "S": (qS, s_min.astype(mx.float32), s_scale.astype(mx.float32)),
        "Vh": (qVh, minVh, scVh),
    }


def dequantize_factors(packed: Dict[str, Tuple[mx.array, mx.array, mx.array]]):
    """Inverse of quantize_factors: returns (U, S, Vh)."""
    qU, minU, scU = packed["U"]
    qS, s_min, s_scale = packed["S"]
    qVh, minVh, scVh = packed["Vh"]

    U = _dequantise_rows(qU, minU, scU)
    Vh = _dequantise_rows(qVh, minVh, scVh)
    S = qS.astype(mx.float32) * s_scale + s_min
    return U, S, Vh
