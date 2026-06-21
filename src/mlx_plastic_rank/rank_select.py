"""Rank selection heuristics and stability metrics using MLX tensors."""

from __future__ import annotations

from typing import Any, Tuple

import mlx.core as mx

ArrayLike = Any
CPU_DEVICE = mx.Device(mx.cpu)


def _to_mx_f32(value: ArrayLike) -> mx.array:
    return mx.array(value, dtype=mx.float32)


def _fro_norm(value: mx.array) -> float:
    return float(mx.sqrt(mx.sum(value * value)).item())


def _singular_energy_values(singulars: mx.array) -> list[float]:
    return [
        float((singulars[index] * singulars[index]).item())
        for index in range(int(singulars.shape[0]))
    ]


def _first_energy_rank(singulars: mx.array, target_compression: float, full_rank: int) -> int:
    energies = _singular_energy_values(singulars)
    total = sum(energies)
    if total <= 0.0:
        return 1
    running = 0.0
    threshold = float(target_compression)
    for index, energy in enumerate(energies):
        running += energy
        if running / (total + 1e-12) >= threshold:
            return max(1, min(index + 1, full_rank))
    return max(1, full_rank)


def stable_rank(A: ArrayLike, eps: float = 1e-6) -> float:
    """Compute the stable rank of a matrix with MLX SVD."""

    data = _to_mx_f32(A)
    if len(data.shape) != 2:
        raise ValueError("stable_rank expects a 2D matrix")
    _, singulars, _ = mx.linalg.svd(data, stream=CPU_DEVICE)
    if int(singulars.shape[0]) == 0:
        return 0.0
    top = float(singulars[0].item())
    if top <= 0.0:
        return 0.0
    fro2 = float(mx.sum(data * data).item())
    return float(fro2 / (top * top + float(eps)))


def theorem_guided_rank(
    A: ArrayLike,
    target_compression: float = 0.9,
    eps: float = 1e-6,
    tol: float = 1.0,
) -> Tuple[int, float]:
    """Select a rank from MLX spectral energy and reconstruction residual."""

    data = _to_mx_f32(A)
    if len(data.shape) != 2 or data.shape[0] != data.shape[1]:
        raise ValueError("theorem_guided_rank expects a square 2D matrix")

    U, singulars, Vh = mx.linalg.svd(data, stream=CPU_DEVICE)
    full = min(int(data.shape[0]), int(data.shape[1]))
    r = _first_energy_rank(singulars, target_compression, full)

    denominator = _fro_norm(data) + float(eps)
    residual = float("inf")
    while r <= full:
        approx = mx.matmul(U[:, :r] * singulars[:r], Vh[:r, :])
        residual = _fro_norm(data - approx) / denominator
        if residual <= tol:
            break
        r += 1

    return r, float(residual)


def _energy_rank(A: ArrayLike, target_compression: float) -> int:
    data = _to_mx_f32(A)
    if len(data.shape) != 2:
        raise ValueError("_energy_rank expects a 2D matrix")
    _, singulars, _ = mx.linalg.svd(data, stream=CPU_DEVICE)
    return _first_energy_rank(singulars, target_compression, min(int(data.shape[0]), int(data.shape[1])))


def choose_rank(
    A: ArrayLike,
    target_compression: float,
    strategy: str,
    eps: float = 1e-6,
) -> Tuple[int, float]:
    """Choose a rank according to the named strategy."""

    if strategy == "stable":
        r = _energy_rank(A, target_compression)
        return r, -1.0
    if strategy == "theorem":
        return theorem_guided_rank(A, target_compression, eps)
    raise ValueError(f"Unknown strategy: {strategy}")
