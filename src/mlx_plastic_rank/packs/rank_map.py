"""Rank-map candidate generators for LoRA packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_pack, load_pack_metadata


@dataclass(frozen=True)
class AdapterShape:
    adapter: str
    target: str
    out_dim: int
    in_dim: int

    @property
    def unit_params(self) -> int:
        return self.out_dim + self.in_dim


@dataclass(frozen=True)
class SpectralRankMapConfig:
    allowed_ranks: tuple[int, ...]
    budget_params: int | None = None
    budget_bytes: int | None = None
    policy: str = "balanced"
    promote_target: str = "attn.k_proj"
    demote_target: str = "attn.q_proj"
    promote_low_lift: float = 1.4
    promote_rank: int = 32
    demote_min_rank: int = 4


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


def load_adapter_shapes(pack_dir: Path) -> dict[str, AdapterShape]:
    """Read adapter in/out dimensions from pack SafeTensors."""

    tensors = load_pack(pack_dir / "pack.safetensors")
    grouped = _group_lora_tensors(tensors)
    shapes: dict[str, AdapterShape] = {}
    for adapter, parts in grouped.items():
        if "A" not in parts or "B" not in parts:
            missing = sorted({"A", "B"} - set(parts))
            raise ValueError(f"Missing LoRA tensor(s) for {adapter}: {missing}")
        left = np.asarray(parts["A"])
        right = np.asarray(parts["B"])
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError(f"LoRA tensors for {adapter} must be matrices")
        if left.shape[1] != right.shape[0]:
            raise ValueError(
                f"LoRA rank mismatch for {adapter}: A rank {left.shape[1]} != B rank {right.shape[0]}"
            )
        shapes[adapter] = AdapterShape(
            adapter=adapter,
            target=_target_from_key(adapter),
            out_dim=int(left.shape[0]),
            in_dim=int(right.shape[1]),
        )
    return shapes


def load_spectral_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Load spectral probe rows keyed by adapter name."""

    rows_by_adapter: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Spectral probe JSON not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Spectral probe JSON is invalid: {path}: {exc}") from exc
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError(f"Spectral probe JSON must contain a 'rows' list: {path}")
        for raw_row in rows:
            if not isinstance(raw_row, dict):
                continue
            adapter = raw_row.get("adapter")
            if not adapter:
                continue
            row = dict(raw_row)
            row["spectral_source"] = str(path)
            rows_by_adapter[str(adapter)] = row
    if not rows_by_adapter:
        raise ValueError("No spectral probe rows were loaded.")
    return rows_by_adapter


def estimate_lora_params(rank_map: dict[str, int], shapes: dict[str, AdapterShape]) -> int:
    total = 0
    for adapter, rank in rank_map.items():
        if adapter not in shapes:
            raise ValueError(f"No LoRA tensor shape found for rank-map adapter {adapter!r}.")
        total += shapes[adapter].unit_params * int(rank)
    return total


def estimate_lora_bytes(rank_map: dict[str, int], shapes: dict[str, AdapterShape]) -> int:
    params = estimate_lora_params(rank_map, shapes)
    return params * 2 + len(rank_map) * 4


def _normalise_allowed_ranks(allowed_ranks: tuple[int, ...]) -> tuple[int, ...]:
    allowed = tuple(sorted({int(rank) for rank in allowed_ranks if int(rank) > 0}))
    if not allowed:
        raise ValueError("At least one allowed rank is required.")
    return allowed


def _floor_allowed_rank(candidate: int, allowed_ranks: tuple[int, ...]) -> int:
    floor = [rank for rank in allowed_ranks if rank <= candidate]
    if not floor:
        raise ValueError(f"No allowed rank is <= {candidate}.")
    return floor[-1]


def _ceil_allowed_rank(candidate: int, allowed_ranks: tuple[int, ...]) -> int:
    ceiling = [rank for rank in allowed_ranks if rank >= candidate]
    if not ceiling:
        raise ValueError(f"No allowed rank is >= {candidate}.")
    return ceiling[0]


def _previous_allowed_rank(
    current: int,
    allowed_ranks: tuple[int, ...],
    minimum: int,
) -> int | None:
    lower = [rank for rank in allowed_ranks if minimum <= rank < current]
    return lower[-1] if lower else None


def _metric_value(
    rank_map: dict[str, int],
    shapes: dict[str, AdapterShape],
    budget_kind: str,
) -> int:
    if budget_kind == "bytes":
        return estimate_lora_bytes(rank_map, shapes)
    return estimate_lora_params(rank_map, shapes)


def _row_float(row: dict[str, Any] | None, key: str, default: float) -> float:
    if row is None:
        return default
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _change_row(
    adapter: str,
    old_rank: int,
    new_rank: int,
    shapes: dict[str, AdapterShape],
    spectral_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    spectral_row = spectral_rows.get(adapter, {})
    unit_params = shapes[adapter].unit_params
    param_delta = unit_params * (new_rank - old_rank)
    return {
        "adapter": adapter,
        "target": shapes[adapter].target,
        "layer": spectral_row.get("layer"),
        "layer_type": spectral_row.get("layer_type"),
        "old_rank": old_rank,
        "new_rank": new_rank,
        "low_lift": _row_float(spectral_row, "low_8_lift", 0.0),
        "estimated_param_delta": param_delta,
        "estimated_byte_delta": param_delta * 2,
    }


def _should_promote(row: dict[str, Any], config: SpectralRankMapConfig) -> bool:
    if config.policy == "balanced" and row.get("layer_type") != "full_attention":
        return False
    if config.policy not in {"balanced", "all-key"}:
        raise ValueError(f"Unsupported spectral rank-map policy: {config.policy}")
    return _row_float(row, "low_8_lift", 0.0) >= config.promote_low_lift


def build_spectral_rank_map_candidate(
    source_pack_dir: Path,
    spectral_paths: list[Path],
    config: SpectralRankMapConfig,
) -> dict[str, Any]:
    """Build a spectral-key-biased rank-map candidate from probe rows."""

    if config.budget_params is not None and config.budget_bytes is not None:
        raise ValueError("Use only one budget: budget_params or budget_bytes.")
    allowed_ranks = _normalise_allowed_ranks(config.allowed_ranks)
    promote_rank = _floor_allowed_rank(config.promote_rank, allowed_ranks)
    minimum_rank = _ceil_allowed_rank(config.demote_min_rank, allowed_ranks)

    metadata = load_pack_metadata(source_pack_dir / "meta.json")
    source_rank_map = dict(metadata.rank_map)
    if not source_rank_map:
        raise ValueError(f"Source pack has no rank_map metadata: {source_pack_dir}")
    unsupported = sorted({rank for rank in source_rank_map.values() if rank not in allowed_ranks})
    if unsupported:
        raise ValueError(
            f"Source pack uses ranks {unsupported}; allowed ranks for this run are {list(allowed_ranks)}."
        )

    shapes = load_adapter_shapes(source_pack_dir)
    spectral_rows = load_spectral_rows(spectral_paths)
    candidate_rank_map = dict(source_rank_map)

    source_params = estimate_lora_params(source_rank_map, shapes)
    source_bytes = estimate_lora_bytes(source_rank_map, shapes)
    if config.budget_bytes is not None:
        budget_kind = "bytes"
        budget_value = int(config.budget_bytes)
    else:
        budget_kind = "params"
        budget_value = int(config.budget_params) if config.budget_params is not None else source_params

    promotions: list[dict[str, Any]] = []
    for adapter, old_rank in source_rank_map.items():
        shape = shapes.get(adapter)
        row = spectral_rows.get(adapter)
        if shape is None or row is None:
            continue
        if shape.target != config.promote_target:
            continue
        if old_rank >= promote_rank:
            continue
        if not _should_promote(row, config):
            continue
        candidate_rank_map[adapter] = promote_rank
        promotions.append(_change_row(adapter, old_rank, promote_rank, shapes, spectral_rows))

    reductions: list[dict[str, Any]] = []
    while _metric_value(candidate_rank_map, shapes, budget_kind) > budget_value:
        reduced_this_pass = False
        candidates: list[tuple[float, str]] = []
        for adapter, rank in candidate_rank_map.items():
            shape = shapes.get(adapter)
            if shape is None or shape.target != config.demote_target:
                continue
            if _previous_allowed_rank(rank, allowed_ranks, minimum_rank) is None:
                continue
            candidates.append((_row_float(spectral_rows.get(adapter), "low_8_lift", float("inf")), adapter))
        for _, adapter in sorted(candidates):
            current_rank = candidate_rank_map[adapter]
            next_rank = _previous_allowed_rank(current_rank, allowed_ranks, minimum_rank)
            if next_rank is None:
                continue
            candidate_rank_map[adapter] = next_rank
            reductions.append(_change_row(adapter, current_rank, next_rank, shapes, spectral_rows))
            reduced_this_pass = True
            if _metric_value(candidate_rank_map, shapes, budget_kind) <= budget_value:
                break
        if not reduced_this_pass:
            current_value = _metric_value(candidate_rank_map, shapes, budget_kind)
            raise ValueError(
                f"Could not fit spectral rank map under {budget_kind} budget "
                f"{budget_value}; best candidate is {current_value}."
            )

    candidate_params = estimate_lora_params(candidate_rank_map, shapes)
    candidate_bytes = estimate_lora_bytes(candidate_rank_map, shapes)
    alpha_map = {adapter: 2.0 * rank for adapter, rank in candidate_rank_map.items()}
    missing_spectral = sorted(adapter for adapter in source_rank_map if adapter not in spectral_rows)

    return {
        "kind": "spectral_rank_map_candidate",
        "name": f"spectral-{config.policy}-from-{metadata.pack_name}",
        "source_pack": str(source_pack_dir),
        "source_pack_name": metadata.pack_name,
        "policy": config.policy,
        "rule": (
            f"Promote {config.promote_target} adapters with low_8_lift >= "
            f"{config.promote_low_lift:g}"
            + (" in full_attention layers" if config.policy == "balanced" else "")
            + f" to rank {promote_rank}, then reduce lowest-low-lift "
            f"{config.demote_target} ranks down to rank {minimum_rank} until "
            f"estimated {budget_kind} <= budget."
        ),
        "allowed_ranks": list(allowed_ranks),
        "budget_kind": budget_kind,
        "budget_value": budget_value,
        "original_declared_rank_sum": sum(source_rank_map.values()),
        "candidate_declared_rank_sum": sum(candidate_rank_map.values()),
        "original_estimated_lora_params": source_params,
        "candidate_estimated_lora_params": candidate_params,
        "estimated_param_delta": candidate_params - source_params,
        "original_estimated_lora_bytes": source_bytes,
        "candidate_estimated_lora_bytes": candidate_bytes,
        "estimated_byte_delta": candidate_bytes - source_bytes,
        "spectral_files": [str(path) for path in spectral_paths],
        "missing_spectral_adapters": missing_spectral,
        "promotions": promotions,
        "reductions": reductions,
        "rank_map": candidate_rank_map,
        "alpha_map": alpha_map,
    }
