"""Rank-algebra ledger for LoRA packs.

The ledger treats each LoRA adapter as a low-rank linear operator and records
how much rank is actually active, wasted, shared, or newly introduced. This is
not a proof of Pop's matrix-polynomial theorem; it is the measurement layer we
need before making that stronger claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import load_pack, load_pack_metadata


def _relative_rank(singular_values: np.ndarray, rank_tol: float) -> int:
    if singular_values.size == 0:
        return 0
    top = float(singular_values[0])
    if top <= 0.0:
        return 0
    threshold = max(float(rank_tol), 0.0) * top
    return int(np.sum(singular_values > threshold))


def _stable_rank_from_singulars(singular_values: np.ndarray) -> float:
    if singular_values.size == 0:
        return 0.0
    top = float(singular_values[0])
    if top <= 0.0:
        return 0.0
    fro2 = float(np.sum(np.square(singular_values)))
    return fro2 / (top * top)


def _lowrank_singular_values(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return non-zero singular values of ``left @ right`` using a small core."""

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Low-rank factors must be rank-2 arrays")
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            f"Low-rank factor mismatch: left columns {left.shape[1]} != right rows {right.shape[0]}"
        )
    if left.size == 0 or right.size == 0:
        return np.array([], dtype=np.float32)

    _, r_left = np.linalg.qr(left, mode="reduced")
    _, r_right_t = np.linalg.qr(right.T, mode="reduced")
    core = r_left @ r_right_t.T
    return np.linalg.svd(core, compute_uv=False).astype(np.float32, copy=False)


def _factor_rank(left: np.ndarray, right: np.ndarray, rank_tol: float) -> tuple[int, np.ndarray]:
    singulars = _lowrank_singular_values(left, right)
    return _relative_rank(singulars, rank_tol), singulars


def _basis(matrix: np.ndarray, rank_tol: float) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.float32)
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = _relative_rank(s, rank_tol)
    return u[:, :rank].astype(np.float32, copy=False)


def _column_basis(left: np.ndarray, right: np.ndarray, rank_tol: float) -> np.ndarray:
    if right.size == 0:
        return np.zeros((left.shape[0], 0), dtype=np.float32)
    u_right, s_right, _ = np.linalg.svd(right, full_matrices=False)
    rank = _relative_rank(s_right, rank_tol)
    if rank == 0:
        return np.zeros((left.shape[0], 0), dtype=np.float32)
    return _basis(left @ u_right[:, :rank], rank_tol)


def _row_basis(left: np.ndarray, right: np.ndarray, rank_tol: float) -> np.ndarray:
    if left.size == 0:
        return np.zeros((right.shape[1], 0), dtype=np.float32)
    _, s_left, vh_left = np.linalg.svd(left, full_matrices=False)
    rank = _relative_rank(s_left, rank_tol)
    if rank == 0:
        return np.zeros((right.shape[1], 0), dtype=np.float32)
    row_samples = vh_left[:rank, :] @ right
    return _basis(row_samples.T, rank_tol)


def _union_rank(left_basis: np.ndarray, right_basis: np.ndarray, rank_tol: float) -> int:
    if left_basis.shape[1] == 0:
        return int(right_basis.shape[1])
    if right_basis.shape[1] == 0:
        return int(left_basis.shape[1])
    singulars = np.linalg.svd(
        np.concatenate([left_basis, right_basis], axis=1),
        compute_uv=False,
    )
    return _relative_rank(singulars, rank_tol)


def _fro_inner(
    left_a: np.ndarray,
    left_b: np.ndarray,
    right_a: np.ndarray,
    right_b: np.ndarray,
) -> float:
    return float(np.trace((left_a.T @ right_a) @ (right_b @ left_b.T)))


def _target_from_key(key: str) -> str:
    parts = key.split(".", 2)
    return parts[2] if len(parts) == 3 else key


def _group_lora_tensors(tensors: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for key, value in tensors.items():
        if ".lora." not in key:
            continue
        prefix, suffix = key.split(".lora.", 1)
        grouped.setdefault(prefix, {})[suffix] = value
    return grouped


def _scaled_factors(
    key: str,
    tensors: dict[str, np.ndarray],
    rank_map: dict[str, int],
    alpha_map: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    if "A" not in tensors or "B" not in tensors or "alpha" not in tensors:
        missing = sorted({"A", "B", "alpha"} - set(tensors))
        raise ValueError(f"Missing LoRA tensor(s) for {key}: {missing}")
    left = np.asarray(tensors["A"], dtype=np.float32)
    right = np.asarray(tensors["B"], dtype=np.float32)
    fallback_rank = int(left.shape[1] if left.shape[1] is not None else 0)
    declared_rank = int(rank_map[key]) if key in rank_map else fallback_rank
    alpha = float(alpha_map.get(key, np.asarray(tensors["alpha"]).reshape(()).item()))
    if declared_rank <= 0:
        raise ValueError(f"Invalid declared rank for {key}: {declared_rank}")
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(f"LoRA tensors for {key} must be matrices")
    if left.shape[1] != right.shape[0]:
        raise ValueError(
            f"LoRA rank mismatch for {key}: A rank {left.shape[1]} != B rank {right.shape[0]}"
        )
    scale = alpha / float(declared_rank)
    return left * scale, right, alpha, declared_rank


def _load_update_factors(
    pack_dir: Path,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray, float, int, int]]]:
    metadata = load_pack_metadata(pack_dir / "meta.json")
    tensors = load_pack(pack_dir / "pack.safetensors")
    grouped = _group_lora_tensors(tensors)
    updates: dict[str, tuple[np.ndarray, np.ndarray, float, int, int]] = {}
    for key in sorted(grouped):
        left, right, alpha, declared_rank = _scaled_factors(
            key,
            grouped[key],
            metadata.rank_map,
            metadata.alpha_map,
        )
        tensor_bytes = sum(int(np.asarray(value).nbytes) for value in grouped[key].values())
        updates[key] = (left, right, alpha, declared_rank, tensor_bytes)
    return metadata.to_dict(), updates


def pack_rank_ledger(pack_dir: Path, rank_tol: float = 1e-5) -> dict[str, Any]:
    """Build a rank ledger for one pack."""

    metadata, updates = _load_update_factors(pack_dir)
    adapters: list[dict[str, Any]] = []
    by_target: dict[str, dict[str, Any]] = {}
    total_declared_rank = 0
    total_effective_rank = 0
    total_bytes = 0
    total_fro2 = 0.0

    for key, (left, right, alpha, declared_rank, tensor_bytes) in updates.items():
        effective_rank, singulars = _factor_rank(left, right, rank_tol)
        fro_norm = float(np.sqrt(np.sum(np.square(singulars)))) if singulars.size else 0.0
        spectral_norm = float(singulars[0]) if singulars.size else 0.0
        stable_rank = _stable_rank_from_singulars(singulars)
        target = _target_from_key(key)
        params = int(left.size + right.size)
        row = {
            "adapter": key,
            "target": target,
            "declared_rank": declared_rank,
            "effective_rank": effective_rank,
            "rank_slack": max(0, declared_rank - effective_rank),
            "rank_efficiency": effective_rank / float(declared_rank),
            "alpha": alpha,
            "scale": alpha / float(declared_rank),
            "shape": [int(left.shape[0]), int(right.shape[1])],
            "params": params,
            "bytes": tensor_bytes,
            "fro_norm": fro_norm,
            "spectral_norm": spectral_norm,
            "stable_rank": stable_rank,
            "singular_values": [float(value) for value in singulars[: min(8, singulars.size)]],
        }
        adapters.append(row)
        total_declared_rank += declared_rank
        total_effective_rank += effective_rank
        total_bytes += tensor_bytes
        total_fro2 += fro_norm * fro_norm

        target_row = by_target.setdefault(
            target,
            {
                "target": target,
                "adapters": 0,
                "declared_rank": 0,
                "effective_rank": 0,
                "rank_slack": 0,
                "params": 0,
                "bytes": 0,
                "fro_norm": 0.0,
            },
        )
        target_row["adapters"] += 1
        target_row["declared_rank"] += declared_rank
        target_row["effective_rank"] += effective_rank
        target_row["rank_slack"] += max(0, declared_rank - effective_rank)
        target_row["params"] += params
        target_row["bytes"] += tensor_bytes
        target_row["fro_norm"] = float(np.sqrt(float(target_row["fro_norm"]) ** 2 + fro_norm * fro_norm))

    summary: dict[str, Any] = {
        "adapter_count": len(adapters),
        "declared_rank": total_declared_rank,
        "effective_rank": total_effective_rank,
        "rank_slack": max(0, total_declared_rank - total_effective_rank),
        "rank_efficiency": (
            total_effective_rank / float(total_declared_rank) if total_declared_rank else 0.0
        ),
        "bytes": total_bytes,
        "fro_norm": float(np.sqrt(total_fro2)),
        "bytes_per_effective_rank": total_bytes / float(max(total_effective_rank, 1)),
    }

    return {
        "kind": "pack_rank_ledger",
        "rank_tol": rank_tol,
        "pack_dir": str(pack_dir),
        "metadata": metadata,
        "summary": summary,
        "by_target": sorted(by_target.values(), key=lambda row: row["target"]),
        "adapters": adapters,
    }


def compare_pack_rank_ledgers(
    left_pack_dir: Path,
    right_pack_dir: Path,
    rank_tol: float = 1e-5,
) -> dict[str, Any]:
    """Compare rank algebra between two packs."""

    left_meta, left_updates = _load_update_factors(left_pack_dir)
    right_meta, right_updates = _load_update_factors(right_pack_dir)
    shared = sorted(set(left_updates) & set(right_updates))
    left_only = sorted(set(left_updates) - set(right_updates))
    right_only = sorted(set(right_updates) - set(left_updates))

    pairs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "shared_adapter_count": len(shared),
        "left_only_count": len(left_only),
        "right_only_count": len(right_only),
        "left_effective_rank": 0,
        "right_effective_rank": 0,
        "composition_rank": 0,
        "rank_savings": 0,
        "column_overlap_rank": 0,
        "row_overlap_rank": 0,
    }

    for key in shared:
        left_a, left_b, _, left_declared, _ = left_updates[key]
        right_a, right_b, _, right_declared, _ = right_updates[key]
        left_rank, left_singulars = _factor_rank(left_a, left_b, rank_tol)
        right_rank, right_singulars = _factor_rank(right_a, right_b, rank_tol)
        sum_rank, sum_singulars = _factor_rank(
            np.concatenate([left_a, right_a], axis=1),
            np.concatenate([left_b, right_b], axis=0),
            rank_tol,
        )

        left_col = _column_basis(left_a, left_b, rank_tol)
        right_col = _column_basis(right_a, right_b, rank_tol)
        column_union = _union_rank(left_col, right_col, rank_tol)
        column_overlap = max(0, left_rank + right_rank - column_union)

        left_row = _row_basis(left_a, left_b, rank_tol)
        right_row = _row_basis(right_a, right_b, rank_tol)
        row_union = _union_rank(left_row, right_row, rank_tol)
        row_overlap = max(0, left_rank + right_rank - row_union)

        left_fro = float(np.sqrt(np.sum(np.square(left_singulars)))) if left_singulars.size else 0.0
        right_fro = float(np.sqrt(np.sum(np.square(right_singulars)))) if right_singulars.size else 0.0
        inner = _fro_inner(left_a, left_b, right_a, right_b)
        cosine = inner / max(left_fro * right_fro, 1e-12)
        rank_savings = max(0, left_rank + right_rank - sum_rank)
        row = {
            "adapter": key,
            "target": _target_from_key(key),
            "left_declared_rank": left_declared,
            "right_declared_rank": right_declared,
            "left_effective_rank": left_rank,
            "right_effective_rank": right_rank,
            "composition_rank": sum_rank,
            "rank_savings": rank_savings,
            "composition_efficiency": (
                sum_rank / float(left_rank + right_rank) if (left_rank + right_rank) else 0.0
            ),
            "column_union_rank": column_union,
            "column_overlap_rank": column_overlap,
            "row_union_rank": row_union,
            "row_overlap_rank": row_overlap,
            "fro_cosine": float(max(-1.0, min(1.0, cosine))),
            "composition_singular_values": [
                float(value) for value in sum_singulars[: min(8, sum_singulars.size)]
            ],
        }
        pairs.append(row)
        summary["left_effective_rank"] += left_rank
        summary["right_effective_rank"] += right_rank
        summary["composition_rank"] += sum_rank
        summary["rank_savings"] += rank_savings
        summary["column_overlap_rank"] += column_overlap
        summary["row_overlap_rank"] += row_overlap

    total_pair_rank = summary["left_effective_rank"] + summary["right_effective_rank"]
    summary["composition_efficiency"] = (
        summary["composition_rank"] / float(total_pair_rank) if total_pair_rank else 0.0
    )

    return {
        "kind": "pack_rank_comparison",
        "rank_tol": rank_tol,
        "left_pack_dir": str(left_pack_dir),
        "right_pack_dir": str(right_pack_dir),
        "left_metadata": left_meta,
        "right_metadata": right_meta,
        "summary": summary,
        "left_only": left_only,
        "right_only": right_only,
        "pairs": pairs,
    }


def ledger_rows_for_csv(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in report.get("adapters", []):
        csv_row = dict(row)
        csv_row.pop("singular_values", None)
        csv_row["shape"] = "x".join(str(dim) for dim in row.get("shape", []))
        rows.append(csv_row)
    return rows


def comparison_rows_for_csv(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in report.get("pairs", []):
        csv_row = dict(row)
        csv_row.pop("composition_singular_values", None)
        rows.append(csv_row)
    return rows
