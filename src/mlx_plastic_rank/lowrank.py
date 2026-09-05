"""Reversible low-rank layers and factor quantization.

Matrix factorization functions are re-exported for compatibility. Their
implementation lives in :mod:`mlx_plastic_rank.factorization`.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple, cast

import mlx.core as mx
import mlx.nn as nn

from .factorization import (
    factorized_lowrank as factorized_lowrank,
)
from .factorization import (
    randomized_svd as randomized_svd,
)
from .factorization import (
    svd_lowrank as svd_lowrank,
)
from .factorization import (
    svd_lowrank_randomized as svd_lowrank_randomized,
)
from .utils import dequantise, quantise


class RankLayer(nn.Module):
    """Linear layer with reversible low-rank residual factors.

    W = W0 + U.T @ diag(S) @ V

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
        self.sleep_dict: Dict[
            int,
            Tuple[mx.array, float, float, float, mx.array, float, float],
        ] = {}
        self.freeze(recurse=False, keys=["W0", "sleep_dict"])

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
        sleep_mask = ~(mx.abs(self.S) > tol)
        mask_raw = cast(List[Any], sleep_mask.tolist())
        self._park_components([i for i, flag in enumerate(mask_raw) if bool(flag)])

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
        drop_raw = cast(List[Any], order[:k_drop].tolist())
        self._park_components([int(i) for i in drop_raw])

    def _park_components(self, indices: List[int]) -> None:
        """Store selected factors once, preserving live order and sleeper IDs."""
        if not indices:
            return
        next_id = max(self.sleep_dict, default=-1) + 1
        for offset, idx in enumerate(indices):
            u, s, v = self.U[idx], self.S[idx], self.V[idx]
            q_u, mn_u, sc_u = quantise(u)
            q_v, mn_v, sc_v = quantise(v)
            self.sleep_dict[next_id + offset] = (
                q_u,
                mn_u,
                sc_u,
                float(s),
                q_v,
                mn_v,
                sc_v,
            )
        dropped = set(indices)
        keep_idx = [i for i in range(self.rank) if i not in dropped]
        if keep_idx:
            live = mx.array(keep_idx)
            self.U = mx.take(self.U, live, axis=0)
            self.S = mx.take(self.S, live, axis=0)
            self.V = mx.take(self.V, live, axis=0)
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
        y = self.attn(x, x, x)
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
