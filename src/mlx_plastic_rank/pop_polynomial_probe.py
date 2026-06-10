"""Matrix-polynomial rank probes for Pop Rank research.

This module verifies Pop's matrix-polynomial rank identity on a chosen
operator, then optionally checks whether LoRA adapter subspaces overlap with
the polynomial image spaces. Those overlap numbers are diagnostics, not proof
that adapter quality follows from the theorem.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .packs.io import load_pack, load_pack_metadata


@dataclass(frozen=True)
class PolynomialPair:
    name: str
    f: tuple[float, ...]
    g: tuple[float, ...]
    gcd: tuple[float, ...]
    lcm: tuple[float, ...]
    note: str
    spectral_roots: tuple[float, ...] = ()


def default_pairs(shift: float = 1.0) -> list[PolynomialPair]:
    """Return simple polynomial pairs with known gcd/lcm relations.

    Coefficients are stored in ascending order, so ``(-shift, 1.0)`` means
    ``x - shift``.
    """

    return [
        PolynomialPair(
            name="x_vs_x_minus_a",
            f=(0.0, 1.0),
            g=(-float(shift), 1.0),
            gcd=(1.0,),
            lcm=(0.0, -float(shift), 1.0),
            note="rank M + rank(M-aI) = n + rank(M(M-aI))",
        ),
        PolynomialPair(
            name="x_minus_1_vs_x_plus_1",
            f=(-1.0, 1.0),
            g=(1.0, 1.0),
            gcd=(1.0,),
            lcm=(-1.0, 0.0, 1.0),
            note="rank(M-I) + rank(M+I) = n + rank(M^2-I)",
        ),
        PolynomialPair(
            name="x_vs_x2_minus_1",
            f=(0.0, 1.0),
            g=(-1.0, 0.0, 1.0),
            gcd=(1.0,),
            lcm=(0.0, -1.0, 0.0, 1.0),
            note="rank M + rank(M^2-I) = n + rank(M^3-M)",
        ),
    ]


def _safe_label(value: str) -> str:
    return (
        value.lower()
        .replace(":", "_")
        .replace(".", "p")
        .replace("-", "neg")
        .replace("+", "")
    )


def _eigenvalue_at_fraction(eigenvalues: np.ndarray, fraction: float) -> float:
    index = int(round(float(fraction) * float(eigenvalues.size - 1)))
    index = max(0, min(index, int(eigenvalues.size - 1)))
    return float(eigenvalues[index])


def _spectral_value(eigenvalues: np.ndarray, spec: str) -> tuple[str, float]:
    label = _safe_label(spec)
    value = spec.strip().lower()
    if not value:
        raise ValueError("Empty spectral shift spec")
    if value == "min":
        return label, float(eigenvalues[0])
    if value == "max":
        return label, float(eigenvalues[-1])
    if value in {"median", "med"}:
        return label, _eigenvalue_at_fraction(eigenvalues, 0.5)
    if value.startswith("p"):
        percentile = float(value[1:]) / 100.0
        if not 0.0 <= percentile <= 1.0:
            raise ValueError(f"Spectral percentile must be in [0, 100], got {spec}")
        return label, _eigenvalue_at_fraction(eigenvalues, percentile)
    if value.startswith("q"):
        quantile = float(value[1:])
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Spectral quantile must be in [0, 1], got {spec}")
        return label, _eigenvalue_at_fraction(eigenvalues, quantile)
    if value.startswith("index:"):
        index = int(value.split(":", 1)[1])
        if index < 0:
            index = int(eigenvalues.size) + index
        if not 0 <= index < eigenvalues.size:
            raise ValueError(f"Spectral index out of range for {eigenvalues.size} values: {spec}")
        return label, float(eigenvalues[index])
    if value.startswith("value:"):
        return label, float(value.split(":", 1)[1])
    raise ValueError(f"Unsupported spectral shift spec: {spec}")


def _roots_to_ascending_coefficients(roots: Sequence[float]) -> tuple[float, ...]:
    if not roots:
        return (1.0,)
    descending = np.poly(np.asarray(roots, dtype=np.float64))
    return tuple(float(value) for value in descending[::-1])


def _notch_roots(eigenvalues: np.ndarray, spec: str) -> tuple[str, np.ndarray]:
    label = _safe_label(spec)
    if ":" not in spec:
        raise ValueError(f"Spectral notch spec must look like low:8, mid:8, or high:8, got {spec}")
    region, count_text = spec.strip().lower().split(":", 1)
    count = int(count_text)
    if count <= 0:
        raise ValueError(f"Spectral notch count must be positive, got {spec}")
    count = min(count, int(eigenvalues.size))
    if region == "low":
        return label, eigenvalues[:count]
    if region == "high":
        return label, eigenvalues[-count:]
    if region in {"mid", "middle"}:
        center = int(eigenvalues.size // 2)
        start = max(0, center - count // 2)
        end = min(int(eigenvalues.size), start + count)
        start = max(0, end - count)
        return label, eigenvalues[start:end]
    if region.startswith("p"):
        center_fraction = float(region[1:]) / 100.0
        if not 0.0 <= center_fraction <= 1.0:
            raise ValueError(f"Spectral notch percentile must be in [0, 100], got {spec}")
        center = int(round(center_fraction * float(eigenvalues.size - 1)))
        start = max(0, center - count // 2)
        end = min(int(eigenvalues.size), start + count)
        start = max(0, end - count)
        return label, eigenvalues[start:end]
    raise ValueError(f"Unsupported spectral notch region: {spec}")


def spectral_shift_pairs(operator: np.ndarray, specs: Sequence[str]) -> list[PolynomialPair]:
    """Build ``x`` versus ``x-lambda`` pairs from operator eigenvalues."""

    matrix = np.asarray(operator, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Spectral shifts require a square operator")
    if np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-6):
        eigenvalues = np.linalg.eigvalsh(matrix)
    else:
        eigenvalues = np.sort(np.real(np.linalg.eigvals(matrix)))
    pairs = []
    for spec in specs:
        label, shift = _spectral_value(eigenvalues, spec)
        gcd: tuple[float, ...]
        lcm: tuple[float, ...]
        if abs(shift) <= 1e-12:
            gcd = (0.0, 1.0)
            lcm = (0.0, 1.0)
        else:
            gcd = (1.0,)
            lcm = (0.0, -shift, 1.0)
        pairs.append(
            PolynomialPair(
                name=f"x_vs_x_minus_lambda_{label}",
                f=(0.0, 1.0),
                g=(-shift, 1.0),
                gcd=gcd,
                lcm=lcm,
                note=f"rank M + rank(M-lambda I), lambda={shift:.8g} from spectral spec {spec}",
                spectral_roots=(shift,),
            )
        )
    return pairs


def spectral_notch_pairs(operator: np.ndarray, specs: Sequence[str]) -> list[PolynomialPair]:
    """Build ``x`` versus multi-root spectral-notch polynomial pairs."""

    matrix = np.asarray(operator, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Spectral notches require a square operator")
    if np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-6):
        eigenvalues = np.linalg.eigvalsh(matrix)
    else:
        eigenvalues = np.sort(np.real(np.linalg.eigvals(matrix)))

    pairs = []
    for spec in specs:
        label, roots = _notch_roots(eigenvalues, spec)
        zero_roots = [root for root in roots if abs(float(root)) <= 1e-12]
        nonzero_roots = [float(root) for root in roots if abs(float(root)) > 1e-12]
        base = _roots_to_ascending_coefficients(nonzero_roots)
        g: tuple[float, ...]
        gcd: tuple[float, ...]
        lcm: tuple[float, ...]
        if zero_roots:
            g = (0.0, *base)
            gcd = (0.0, 1.0)
            lcm = g
        else:
            g = base
            gcd = (1.0,)
            lcm = (0.0, *g)
        pairs.append(
            PolynomialPair(
                name=f"x_vs_spectral_notch_{label}",
                f=(0.0, 1.0),
                g=g,
                gcd=gcd,
                lcm=lcm,
                note=(
                    f"rank M + rank spectral notch for {spec}; "
                    f"roots={len(roots)}, range=[{float(roots[0]):.8g}, {float(roots[-1]):.8g}]"
                ),
                spectral_roots=tuple(float(root) for root in roots),
            )
        )
    return pairs


def toy_weight() -> np.ndarray:
    """Return a deterministic weight with useful rank defects in its Gram."""

    return np.diag(np.array([3.0, 2.0, 1.0, 0.0], dtype=np.float32))


def build_operator(weight: np.ndarray, mode: str) -> np.ndarray:
    """Build the square matrix that the polynomial identity will inspect."""

    matrix = np.asarray(weight, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Weight must be a 2D array, got shape {matrix.shape}")
    if mode == "row-gram":
        return matrix @ matrix.T
    if mode == "col-gram":
        return matrix.T @ matrix
    if mode == "direct":
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("direct operator mode requires a square matrix")
        return matrix
    raise ValueError(f"Unsupported operator mode: {mode}")


def operator_shape(weight: np.ndarray, mode: str) -> tuple[int, int]:
    """Return the square operator shape without materializing the operator."""

    matrix = np.asarray(weight)
    if matrix.ndim != 2:
        raise ValueError(f"Weight must be a 2D array, got shape {matrix.shape}")
    if mode == "row-gram":
        return int(matrix.shape[0]), int(matrix.shape[0])
    if mode == "col-gram":
        return int(matrix.shape[1]), int(matrix.shape[1])
    if mode == "direct":
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("direct operator mode requires a square matrix")
        return int(matrix.shape[0]), int(matrix.shape[1])
    raise ValueError(f"Unsupported operator mode: {mode}")


def projection_basis(size: int, dim: int, seed: int = 0) -> np.ndarray:
    """Build a deterministic orthonormal projection basis."""

    if dim <= 0:
        raise ValueError("Projection dimension must be positive")
    if dim > size:
        raise ValueError(f"Projection dimension {dim} exceeds operator size {size}")
    rng = np.random.default_rng(int(seed))
    samples = rng.normal(size=(size, dim)).astype(np.float32)
    basis, _ = np.linalg.qr(samples, mode="reduced")
    return basis.astype(np.float32, copy=False)


def project_weight_operator(weight: np.ndarray, mode: str, basis: np.ndarray) -> np.ndarray:
    """Project an operator from ``weight`` without building the full Gram first."""

    matrix = np.asarray(weight, dtype=np.float32)
    projection = np.asarray(basis, dtype=np.float32)
    if mode == "row-gram":
        reduced = projection.T @ matrix
        projected = reduced @ reduced.T
    elif mode == "col-gram":
        reduced = matrix @ projection
        projected = reduced.T @ reduced
    elif mode == "direct":
        projected = projection.T @ matrix @ projection
    else:
        raise ValueError(f"Unsupported operator mode: {mode}")
    if np.allclose(projected, projected.T, rtol=1e-4, atol=1e-5):
        projected = 0.5 * (projected + projected.T)
    return projected.astype(np.float32, copy=False)


def project_operator(operator: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project ``operator`` into ``basis`` coordinates."""

    projected = (basis.T @ operator @ basis).astype(np.float32, copy=False)
    if np.allclose(operator, operator.T, rtol=1e-4, atol=1e-5):
        projected = 0.5 * (projected + projected.T)
    return projected.astype(np.float32, copy=False)


def polynomial_to_string(coefficients: Sequence[float]) -> str:
    terms: list[str] = []
    for power, coeff in enumerate(coefficients):
        value = float(coeff)
        if abs(value) < 1e-12:
            continue
        if power == 0:
            term = f"{value:g}"
        elif power == 1:
            term = f"{value:g}*x"
        else:
            term = f"{value:g}*x^{power}"
        terms.append(term)
    return " + ".join(terms) if terms else "0"


def evaluate_polynomial(matrix: np.ndarray, coefficients: Sequence[float]) -> np.ndarray:
    """Evaluate a polynomial in ascending-coefficient form at ``matrix``."""

    operator = np.asarray(matrix, dtype=np.float32)
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("Polynomial evaluation requires a square matrix")

    n = operator.shape[0]
    result = np.zeros_like(operator, dtype=np.float32)
    power = np.eye(n, dtype=np.float32)
    for coeff in coefficients:
        if coeff:
            result = result + float(coeff) * power
        power = power @ operator
    return result


def _rank_from_singulars(singular_values: np.ndarray, rank_tol: float) -> int:
    if singular_values.size == 0:
        return 0
    top = float(singular_values[0])
    if top <= 0.0:
        return 0
    return int(np.sum(singular_values > max(float(rank_tol), 0.0) * top))


def numerical_rank(matrix: np.ndarray, rank_tol: float = 1e-5) -> int:
    """Return relative numerical rank using the largest singular value."""

    values = np.linalg.svd(np.asarray(matrix, dtype=np.float32), compute_uv=False)
    return _rank_from_singulars(values, rank_tol)


def _basis(matrix: np.ndarray, rank_tol: float, rank_cap: int | None = None) -> np.ndarray:
    data = np.asarray(matrix, dtype=np.float32)
    if data.size == 0:
        return np.zeros((data.shape[0], 0), dtype=np.float32)
    u, singulars, _ = np.linalg.svd(data, full_matrices=False)
    rank = _rank_from_singulars(singulars, rank_tol)
    if rank_cap is not None:
        rank = min(rank, max(0, int(rank_cap)))
    return u[:, :rank].astype(np.float32, copy=False)


def _basis_rank(matrix: np.ndarray) -> int:
    return int(np.asarray(matrix).shape[1])


def _spectral_root_mask(eigenvalues: np.ndarray, roots: Sequence[float]) -> np.ndarray:
    mask = np.zeros(eigenvalues.shape, dtype=bool)
    used: set[int] = set()
    for root in roots:
        distances = np.abs(eigenvalues - float(root))
        for index in np.argsort(distances):
            selected = int(index)
            if selected not in used:
                used.add(selected)
                mask[selected] = True
                break
    return mask


def _union_rank(left_basis: np.ndarray, right_basis: np.ndarray, rank_tol: float) -> int:
    if left_basis.shape[1] == 0:
        return int(right_basis.shape[1])
    if right_basis.shape[1] == 0:
        return int(left_basis.shape[1])
    combined = np.concatenate([left_basis, right_basis], axis=1)
    return numerical_rank(combined, rank_tol)


def _overlap_rank(left_basis: np.ndarray, right_basis: np.ndarray, rank_tol: float) -> int:
    union = _union_rank(left_basis, right_basis, rank_tol)
    return max(0, int(left_basis.shape[1] + right_basis.shape[1] - union))


def _subspace_energy_fraction(source_basis: np.ndarray, target_basis: np.ndarray) -> float:
    if source_basis.shape[1] == 0 or target_basis.shape[1] == 0:
        return 0.0
    projection = target_basis.T @ source_basis
    return float(np.sum(np.square(projection)) / float(source_basis.shape[1]))


def evaluate_pair(
    operator: np.ndarray,
    pair: PolynomialPair,
    *,
    rank_tol: float = 1e-5,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if pair.spectral_roots:
        if np.allclose(operator, operator.T, rtol=1e-4, atol=1e-5):
            operator = 0.5 * (operator + operator.T)
        else:
            raise ValueError("Spectral-root pair evaluation requires a symmetric operator")
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(operator, dtype=np.float32))
        root_mask = _spectral_root_mask(eigenvalues, pair.spectral_roots)
        top = max(float(np.max(np.abs(eigenvalues))), 1.0)
        nonzero = np.abs(eigenvalues) > max(float(rank_tol), 0.0) * top
        g_keep = ~root_mask
        zero_root = any(abs(float(root)) <= 1e-12 for root in pair.spectral_roots)
        gcd_keep = nonzero if zero_root else np.ones_like(g_keep, dtype=bool)
        lcm_keep = g_keep & nonzero
        matrices = {
            "f": eigenvectors[:, nonzero].astype(np.float32, copy=False),
            "g": eigenvectors[:, g_keep].astype(np.float32, copy=False),
            "gcd": eigenvectors[:, gcd_keep].astype(np.float32, copy=False),
            "lcm": eigenvectors[:, lcm_keep].astype(np.float32, copy=False),
            "notch_null": eigenvectors[:, root_mask].astype(np.float32, copy=False),
        }
        rank_matrices = {name: matrices[name] for name in ("f", "g", "gcd", "lcm")}
        ranks = {name: _basis_rank(value) for name, value in rank_matrices.items()}
        lhs = int(ranks["f"] + ranks["g"])
        rhs = int(ranks["gcd"] + ranks["lcm"])
        return (
            {
                "pair": pair.name,
                "note": pair.note,
                "polynomials": {
                    "f": polynomial_to_string(pair.f),
                    "g": polynomial_to_string(pair.g),
                    "gcd": polynomial_to_string(pair.gcd),
                    "lcm": polynomial_to_string(pair.lcm),
                },
                "ranks": ranks,
                "lhs": lhs,
                "rhs": rhs,
                "rank_gap": lhs - rhs,
                "identity_holds": lhs == rhs,
                "spectral_roots": [float(root) for root in pair.spectral_roots],
            },
            matrices,
        )

    matrices = {
        "f": evaluate_polynomial(operator, pair.f),
        "g": evaluate_polynomial(operator, pair.g),
        "gcd": evaluate_polynomial(operator, pair.gcd),
        "lcm": evaluate_polynomial(operator, pair.lcm),
    }
    ranks = {name: numerical_rank(value, rank_tol) for name, value in matrices.items()}
    lhs = int(ranks["f"] + ranks["g"])
    rhs = int(ranks["gcd"] + ranks["lcm"])
    return (
        {
            "pair": pair.name,
            "note": pair.note,
            "polynomials": {
                "f": polynomial_to_string(pair.f),
                "g": polynomial_to_string(pair.g),
                "gcd": polynomial_to_string(pair.gcd),
                "lcm": polynomial_to_string(pair.lcm),
            },
            "ranks": ranks,
            "lhs": lhs,
            "rhs": rhs,
            "rank_gap": lhs - rhs,
            "identity_holds": lhs == rhs,
        },
        matrices,
    )


def _group_lora_tensors(tensors: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for key, value in tensors.items():
        if ".lora." not in key:
            continue
        prefix, suffix = key.split(".lora.", 1)
        grouped.setdefault(prefix, {})[suffix] = value
    return grouped


def _lowrank_product_rank(left: np.ndarray, right: np.ndarray, rank_tol: float) -> int:
    if left.size == 0 or right.size == 0:
        return 0
    _, left_r = np.linalg.qr(left, mode="reduced")
    _, right_t_r = np.linalg.qr(right.T, mode="reduced")
    core = left_r @ right_t_r.T
    return numerical_rank(core, rank_tol)


def _load_adapter_factors(pack_dir: Path) -> list[tuple[str, np.ndarray, np.ndarray, int]]:
    metadata = load_pack_metadata(pack_dir / "meta.json")
    tensors = load_pack(pack_dir / "pack.safetensors")
    adapters: list[tuple[str, np.ndarray, np.ndarray, int]] = []

    for key, group in sorted(_group_lora_tensors(tensors).items()):
        if not {"A", "B", "alpha"} <= set(group):
            continue
        left = np.asarray(group["A"], dtype=np.float32)
        right = np.asarray(group["B"], dtype=np.float32)
        alpha = float(metadata.alpha_map.get(key, np.asarray(group["alpha"]).reshape(()).item()))
        if left.ndim != 2 or right.ndim != 2:
            continue
        if left.shape[1] != right.shape[0]:
            continue
        declared_value = metadata.rank_map.get(key)
        declared_rank = int(declared_value if declared_value is not None else int(left.shape[1]))
        if declared_rank <= 0:
            continue
        left = (left * (alpha / float(declared_rank))).astype(np.float32, copy=False)
        adapters.append((key, left, right.astype(np.float32, copy=False), declared_rank))
    return adapters


def _adapter_space_basis(
    left: np.ndarray,
    right: np.ndarray,
    operator_size: int,
    rank_tol: float,
    rank_cap: int,
    projection: np.ndarray | None = None,
) -> tuple[str, np.ndarray] | None:
    if projection is not None:
        original_size = projection.shape[0]
        if left.shape[0] == original_size:
            return "projected_output_column_space", _basis(projection.T @ left, rank_tol, rank_cap)
        if right.shape[1] == original_size:
            return "projected_input_row_space", _basis(projection.T @ right.T, rank_tol, rank_cap)
        return None
    if left.shape[0] == operator_size:
        return "output_column_space", _basis(left, rank_tol, rank_cap)
    if right.shape[1] == operator_size:
        return "input_row_space", _basis(right.T, rank_tol, rank_cap)
    return None


def adapter_overlap_reports(
    operator: np.ndarray,
    image_matrices: dict[str, dict[str, np.ndarray]],
    *,
    pack_dir: Path,
    rank_tol: float = 1e-5,
    max_adapters: int = 8,
    projection: np.ndarray | None = None,
    adapter_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Compare compatible LoRA adapter subspaces to polynomial image spaces."""

    operator_size = int(operator.shape[0])
    reports: list[dict[str, Any]] = []
    for key, left, right, declared_rank in _load_adapter_factors(pack_dir):
        if adapter_keys is not None and key not in adapter_keys:
            continue
        effective_rank = min(_lowrank_product_rank(left, right, rank_tol), declared_rank)
        space = _adapter_space_basis(left, right, operator_size, rank_tol, effective_rank, projection)
        if space is None:
            continue
        space_name, adapter_basis = space
        pair_rows: list[dict[str, Any]] = []
        for pair_name, matrices in image_matrices.items():
            target_bases = {
                label: _basis(matrix, rank_tol)
                for label, matrix in matrices.items()
            }
            pair_rows.append(
                {
                    "pair": pair_name,
                    "overlap_rank": {
                        label: _overlap_rank(adapter_basis, target_basis, rank_tol)
                        for label, target_basis in target_bases.items()
                    },
                    "energy_fraction": {
                        label: _subspace_energy_fraction(adapter_basis, target_basis)
                        for label, target_basis in target_bases.items()
                    },
                }
            )
        reports.append(
            {
                "adapter": key,
                "shape": [int(left.shape[0]), int(right.shape[1])],
                "declared_rank": declared_rank,
                "effective_rank": effective_rank,
                "space_rank": int(adapter_basis.shape[1]),
                "matched_space": space_name,
                "pairs": pair_rows,
            }
        )
        if len(reports) >= max_adapters:
            break
    return reports


def run_probe(
    *,
    weight: np.ndarray | None = None,
    operator_mode: str = "row-gram",
    source: str = "toy",
    pair_names: set[str] | None = None,
    shift: float = 1.0,
    rank_tol: float = 1e-5,
    pack_dir: Path | None = None,
    max_adapters: int = 8,
    projection_dim: int | None = None,
    projection_seed: int = 0,
    spectral_shift_specs: Sequence[str] | None = None,
    spectral_notch_specs: Sequence[str] | None = None,
    adapter_keys: set[str] | None = None,
) -> dict[str, Any]:
    selected_weight = toy_weight() if weight is None else np.asarray(weight, dtype=np.float32)
    raw_operator_shape = operator_shape(selected_weight, operator_mode)
    projection = None
    if projection_dim is not None:
        projection = projection_basis(raw_operator_shape[0], projection_dim, projection_seed)
        operator = project_weight_operator(selected_weight, operator_mode, projection)
    else:
        raw_operator = build_operator(selected_weight, operator_mode)
        operator = raw_operator
    if spectral_shift_specs or spectral_notch_specs:
        pairs = []
        if spectral_shift_specs:
            pairs.extend(spectral_shift_pairs(operator, spectral_shift_specs))
        if spectral_notch_specs:
            pairs.extend(spectral_notch_pairs(operator, spectral_notch_specs))
    else:
        pairs = default_pairs(shift)
    pairs = [pair for pair in pairs if pair_names is None or pair.name in pair_names]
    if not pairs:
        raise ValueError("No polynomial pairs selected")

    pair_reports: list[dict[str, Any]] = []
    image_matrices: dict[str, dict[str, np.ndarray]] = {}
    for pair in pairs:
        pair_report, matrices = evaluate_pair(operator, pair, rank_tol=rank_tol)
        pair_reports.append(pair_report)
        image_matrices[pair.name] = matrices

    final_report: dict[str, Any] = {
        "kind": "pop_polynomial_probe",
        "source": source,
        "rank_tol": rank_tol,
        "operator": {
            "mode": operator_mode,
            "shape": [int(operator.shape[0]), int(operator.shape[1])],
            "rank": numerical_rank(operator, rank_tol),
            "unprojected_shape": [int(raw_operator_shape[0]), int(raw_operator_shape[1])],
            "source_weight_shape": [int(selected_weight.shape[0]), int(selected_weight.shape[1])],
            "projection_dim": projection_dim,
            "projection_seed": projection_seed if projection_dim is not None else None,
        },
        "spectral_shift_specs": list(spectral_shift_specs or []),
        "spectral_notch_specs": list(spectral_notch_specs or []),
        "pairs": pair_reports,
        "all_identities_hold": all(row["identity_holds"] for row in pair_reports),
        "adapter_overlaps": [],
        "claim_boundary": (
            "This verifies matrix-polynomial rank accounting on the chosen operator. "
            "Adapter overlaps are diagnostics and do not establish quality benefit."
        ),
    }
    if pack_dir is not None:
        final_report["adapter_overlaps"] = adapter_overlap_reports(
            operator,
            image_matrices,
            pack_dir=pack_dir,
            rank_tol=rank_tol,
            max_adapters=max_adapters,
            projection=projection,
            adapter_keys=adapter_keys,
        )
    return final_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify Pop's matrix-polynomial rank identity and probe LoRA subspace overlap."
    )
    parser.add_argument("--weight-npy", type=Path, help="Optional .npy weight/operator source.")
    parser.add_argument(
        "--operator",
        choices=("row-gram", "col-gram", "direct"),
        default="row-gram",
        help="Operator to build from --weight-npy. Default uses W @ W.T.",
    )
    parser.add_argument(
        "--pair",
        action="append",
        help="Polynomial pair to run. Repeat to select multiple. Defaults to all built-in pairs.",
    )
    parser.add_argument("--shift", type=float, default=1.0, help="Shift a for x_vs_x_minus_a.")
    parser.add_argument("--rank-tol", type=float, default=1e-5, help="Relative numerical rank tolerance.")
    parser.add_argument(
        "--spectral-shift",
        action="append",
        help=(
            "Use an eigenvalue-targeted shift for x_vs_x_minus_lambda. "
            "Examples: min, p10, p50, p90, max, index:0, value:1.0. "
            "Repeat for multiple shifts."
        ),
    )
    parser.add_argument(
        "--spectral-notch",
        action="append",
        help=(
            "Use a multi-root spectral notch polynomial. Examples: low:8, mid:8, high:8, p25:8. "
            "Repeat for multiple notches."
        ),
    )
    parser.add_argument("--pack-dir", type=Path, help="Optional LoRA pack directory for adapter overlaps.")
    parser.add_argument(
        "--adapter-key",
        action="append",
        help="Exact adapter key to include from --pack-dir. Repeat for multiple keys.",
    )
    parser.add_argument("--max-adapters", type=int, default=8, help="Maximum compatible adapters to report.")
    parser.add_argument(
        "--projection-dim",
        type=int,
        help="Project the operator and compatible adapter spaces to this dimension before probing.",
    )
    parser.add_argument("--projection-seed", type=int, default=0, help="Seed for deterministic projection.")
    parser.add_argument("--out", type=Path, help="Optional JSON output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    weight = np.load(args.weight_npy) if args.weight_npy else None
    report = run_probe(
        weight=weight,
        operator_mode=args.operator,
        source=str(args.weight_npy) if args.weight_npy else "toy",
        pair_names=set(args.pair) if args.pair else None,
        shift=args.shift,
        rank_tol=args.rank_tol,
        pack_dir=args.pack_dir,
        max_adapters=max(1, int(args.max_adapters)),
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        spectral_shift_specs=args.spectral_shift,
        spectral_notch_specs=args.spectral_notch,
        adapter_keys=set(args.adapter_key) if args.adapter_key else None,
    )
    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
