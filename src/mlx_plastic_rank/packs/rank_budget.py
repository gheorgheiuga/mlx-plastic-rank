"""Shape-aware rank-map budget utilities for PopRank experiments."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .inspection import allowed_ranks_for
from .io import load_pack

DTYPE_BYTES = {
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
}

TARGET_FIXED_R16 = "fixed_r16"
TARGET_FIXED_R32 = "fixed_r32"
TARGET_RANK_MAP = "rank_map"
TARGET_BYTES = "bytes"
TARGET_FIXED_R32_PERCENT = "fixed_r32_percent"


class RankBudgetError(ValueError):
    """Raised when a rank map, shape table, or budget target is invalid."""


@dataclass(frozen=True)
class AdapterShape:
    """LoRA adapter matrix geometry."""

    adapter: str
    target: str
    layer: int
    in_dim: int
    out_dim: int

    @property
    def unit_params(self) -> int:
        return self.in_dim + self.out_dim

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "target": self.target,
            "layer": self.layer,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "unit_params": self.unit_params,
        }


def dtype_num_bytes(dtype: str) -> int:
    """Return bytes per scalar for supported PopRank accounting dtypes."""

    key = str(dtype).strip().lower()
    if key not in DTYPE_BYTES:
        raise RankBudgetError(
            f"Unsupported dtype {dtype!r}; expected one of {sorted(DTYPE_BYTES)}."
        )
    return DTYPE_BYTES[key]


def _target_from_adapter(adapter: str) -> str:
    parts = adapter.split(".", 2)
    if len(parts) != 3 or parts[0] != "blocks":
        raise RankBudgetError(f"Invalid adapter key {adapter!r}; expected blocks.<idx>.attn.*")
    target = parts[2]
    if not target.startswith("attn."):
        raise RankBudgetError(f"Invalid adapter target in {adapter!r}; expected attn.*")
    return target


def _layer_from_adapter(adapter: str) -> int:
    parts = adapter.split(".", 2)
    if len(parts) < 2:
        raise RankBudgetError(f"Invalid adapter key {adapter!r}; expected blocks.<idx>.attn.*")
    try:
        return int(parts[1])
    except ValueError as exc:
        raise RankBudgetError(f"Invalid adapter layer in {adapter!r}.") from exc


def _matrix_shape(value: Any, *, key: str) -> tuple[int, int]:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 2:
        raise RankBudgetError(f"LoRA tensor {key} must be a rank-2 array.")
    return int(shape[0]), int(shape[1])


def _group_lora_tensors(tensors: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for key, value in tensors.items():
        if ".lora." not in key:
            continue
        prefix, suffix = key.split(".lora.", 1)
        grouped.setdefault(prefix, {})[suffix] = value
    return grouped


def adapter_shapes_from_tensors(tensors: Mapping[str, Any]) -> dict[str, AdapterShape]:
    """Discover adapter shapes from LoRA SafeTensors arrays."""

    grouped = _group_lora_tensors(tensors)
    if not grouped:
        raise RankBudgetError("No LoRA tensors found.")
    shapes: dict[str, AdapterShape] = {}
    for adapter in sorted(grouped):
        parts = grouped[adapter]
        if "A" not in parts or "B" not in parts:
            missing = sorted({"A", "B"} - set(parts))
            raise RankBudgetError(f"Missing LoRA tensor(s) for {adapter}: {missing}")
        left_rows, left_cols = _matrix_shape(parts["A"], key=f"{adapter}.lora.A")
        right_rows, right_cols = _matrix_shape(parts["B"], key=f"{adapter}.lora.B")
        if left_cols != right_rows:
            raise RankBudgetError(
                f"LoRA rank mismatch for {adapter}: A rank {left_cols} != B rank {right_rows}"
            )
        shapes[adapter] = AdapterShape(
            adapter=adapter,
            target=_target_from_adapter(adapter),
            layer=_layer_from_adapter(adapter),
            in_dim=right_cols,
            out_dim=left_rows,
        )
    return shapes


def adapter_shapes_from_pack(pack_dir: Path) -> dict[str, AdapterShape]:
    """Discover adapter shapes from ``pack.safetensors`` under a pack directory."""

    tensor_path = pack_dir / "pack.safetensors"
    if not tensor_path.exists():
        raise RankBudgetError(f"Pack tensor file not found: {tensor_path}")
    return adapter_shapes_from_tensors(load_pack(tensor_path))


def _canonical_target(target: str) -> str:
    aliases = {
        "q": "attn.q_proj",
        "k": "attn.k_proj",
        "v": "attn.v_proj",
        "o": "attn.o_proj",
    }
    value = target.strip()
    return aliases.get(value, value)


def adapter_shapes_from_model(model: Any, targets: Iterable[str]) -> dict[str, AdapterShape]:
    """Discover supported adapter shapes from an already loaded model and target list."""

    from .manager import LoRAManager, _try_get_nested_attr

    manager = LoRAManager(model)
    canonical_targets = [_canonical_target(target) for target in targets]
    blocks = manager._blocks()
    shapes: dict[str, AdapterShape] = {}
    for block_idx, block in enumerate(blocks):
        for target in canonical_targets:
            spec = manager._target_specs.get(target)
            if spec is None:
                raise RankBudgetError(f"Target {target!r} is not supported by this model.")
            if _try_get_nested_attr(block, spec.wrapper_attr) is None:
                continue
            adapter = f"blocks.{block_idx}.{target}"
            shapes[adapter] = AdapterShape(
                adapter=adapter,
                target=target,
                layer=block_idx,
                in_dim=int(spec.input_dim),
                out_dim=int(spec.output_dim),
            )
    if not shapes:
        raise RankBudgetError("No matching adapter shapes were discovered from the model.")
    return shapes


def load_rank_map_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a consumable rank-map JSON or a richer report containing rank_map."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RankBudgetError(f"Rank-map JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RankBudgetError(f"Rank-map JSON is invalid: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RankBudgetError("Rank-map JSON must be an object.")
    rank_payload = payload.get("rank_map", payload)
    alpha_payload = payload.get("alpha_map", {})
    if not isinstance(rank_payload, Mapping) or not rank_payload:
        raise RankBudgetError("Rank-map JSON must contain a non-empty rank_map object.")
    if not isinstance(alpha_payload, Mapping):
        raise RankBudgetError("Rank-map JSON alpha_map must be an object when provided.")
    return dict(rank_payload), dict(alpha_payload)


def alpha_map_for_rank_map(rank_map: Mapping[str, int]) -> dict[str, float]:
    """Return the repo-standard alpha map for a rank map."""

    return {str(adapter): 2.0 * int(rank) for adapter, rank in sorted(rank_map.items())}


def fixed_rank_map(
    shapes: Mapping[str, AdapterShape],
    rank: int,
    *,
    profile: str,
    adapter_keys: Iterable[str] | None = None,
) -> dict[str, int]:
    """Build a fixed-rank map over a shape table or adapter subset."""

    allowed = set(allowed_ranks_for(profile))
    if rank not in allowed:
        raise RankBudgetError(
            f"Rank {rank} is not allowed for profile={profile}; allowed ranks are {sorted(allowed)}."
        )
    keys = sorted(adapter_keys if adapter_keys is not None else shapes.keys())
    missing = [key for key in keys if key not in shapes]
    if missing:
        raise RankBudgetError(f"Cannot build fixed-rank map; missing shapes for {missing}.")
    return {key: int(rank) for key in keys}


def validate_rank_map(
    rank_map: Mapping[str, Any],
    shapes: Mapping[str, AdapterShape],
    *,
    profile: str,
    alpha_map: Mapping[str, Any] | None = None,
) -> tuple[dict[str, int], dict[str, float]]:
    """Validate rank/alpha values and return normalized maps."""

    if not isinstance(rank_map, Mapping) or not rank_map:
        raise RankBudgetError("Rank map must be a non-empty object.")
    allowed = set(int(rank) for rank in allowed_ranks_for(profile))
    normalized_rank_map: dict[str, int] = {}
    for raw_adapter, raw_rank in rank_map.items():
        adapter = str(raw_adapter)
        if adapter not in shapes:
            raise RankBudgetError(f"Unsupported adapter key {adapter!r}; no shape is known for it.")
        if isinstance(raw_rank, bool) or not isinstance(raw_rank, int):
            raise RankBudgetError(f"Rank map entry {adapter!r} must be an integer rank.")
        rank = int(raw_rank)
        if rank not in allowed:
            raise RankBudgetError(
                f"Rank map entry {adapter!r} uses unsupported rank {rank}; "
                f"allowed ranks are {sorted(allowed)}."
            )
        normalized_rank_map[adapter] = rank

    provided_alpha = dict(alpha_map or {})
    extra_alpha = sorted(str(key) for key in provided_alpha if str(key) not in normalized_rank_map)
    if extra_alpha:
        raise RankBudgetError(f"Alpha map contains entries not present in rank_map: {extra_alpha}.")

    normalized_alpha_map: dict[str, float] = {}
    for adapter, rank in normalized_rank_map.items():
        raw_alpha = provided_alpha.get(adapter, 2.0 * rank)
        try:
            alpha = float(raw_alpha)
        except (TypeError, ValueError) as exc:
            raise RankBudgetError(f"Alpha map entry {adapter!r} must be numeric.") from exc
        expected = 2.0 * rank
        if alpha != 0.0 and not math.isclose(alpha, expected, rel_tol=1e-6, abs_tol=1e-6):
            raise RankBudgetError(
                f"Alpha map entry {adapter!r} must be 0.0 or 2*rank ({expected})."
            )
        normalized_alpha_map[adapter] = alpha
    return dict(sorted(normalized_rank_map.items())), dict(sorted(normalized_alpha_map.items()))


def estimate_adapter_params(shape: AdapterShape, rank: int) -> int:
    """Estimate LoRA parameters for one adapter as rank * (in_dim + out_dim)."""

    return int(rank) * shape.unit_params


def estimate_adapter_bytes(
    shape: AdapterShape,
    rank: int,
    *,
    tensor_dtype: str = "fp16",
    alpha_dtype: str = "fp32",
) -> dict[str, int]:
    """Estimate tensor bytes for one LoRA adapter."""

    tensor_nbytes = dtype_num_bytes(tensor_dtype)
    alpha_nbytes = dtype_num_bytes(alpha_dtype)
    rank_int = int(rank)
    a_bytes = shape.out_dim * rank_int * tensor_nbytes
    b_bytes = rank_int * shape.in_dim * tensor_nbytes
    return {
        "a_bytes": int(a_bytes),
        "b_bytes": int(b_bytes),
        "alpha_bytes": int(alpha_nbytes),
        "tensor_bytes": int(a_bytes + b_bytes + alpha_nbytes),
    }


def _summary_bucket(bucket: dict[str, Any], row: Mapping[str, Any]) -> None:
    bucket["adapter_count"] += 1
    bucket["rank"] += int(row["rank"])
    bucket["params"] += int(row["params"])
    bucket["a_bytes"] += int(row["a_bytes"])
    bucket["b_bytes"] += int(row["b_bytes"])
    bucket["alpha_bytes"] += int(row["alpha_bytes"])
    bucket["tensor_bytes"] += int(row["tensor_bytes"])


def _empty_bucket(**extra: Any) -> dict[str, Any]:
    return {
        **extra,
        "adapter_count": 0,
        "rank": 0,
        "params": 0,
        "a_bytes": 0,
        "b_bytes": 0,
        "alpha_bytes": 0,
        "tensor_bytes": 0,
    }


def rank_map_budget_report(
    rank_map: Mapping[str, Any],
    shapes: Mapping[str, AdapterShape],
    *,
    profile: str,
    alpha_map: Mapping[str, Any] | None = None,
    tensor_dtype: str = "fp16",
    alpha_dtype: str = "fp32",
    file_overhead_bytes: int = 0,
    target_budget_bytes: int | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Validate and budget a rank map with shape-aware byte accounting."""

    normalized_rank_map, normalized_alpha_map = validate_rank_map(
        rank_map,
        shapes,
        profile=profile,
        alpha_map=alpha_map,
    )
    if file_overhead_bytes < 0:
        raise RankBudgetError("file_overhead_bytes must be >= 0.")
    if target_budget_bytes is not None and target_budget_bytes < 0:
        raise RankBudgetError("target_budget_bytes must be >= 0 when provided.")

    adapter_rows: list[dict[str, Any]] = []
    by_layer: dict[int, dict[str, Any]] = {}
    by_target: dict[str, dict[str, Any]] = {}
    for adapter, rank in normalized_rank_map.items():
        shape = shapes[adapter]
        params = estimate_adapter_params(shape, rank)
        byte_parts = estimate_adapter_bytes(
            shape,
            rank,
            tensor_dtype=tensor_dtype,
            alpha_dtype=alpha_dtype,
        )
        row = {
            "adapter": adapter,
            "target": shape.target,
            "layer": shape.layer,
            "rank": rank,
            "alpha": normalized_alpha_map[adapter],
            "in_dim": shape.in_dim,
            "out_dim": shape.out_dim,
            "unit_params": shape.unit_params,
            "params": params,
            **byte_parts,
        }
        adapter_rows.append(row)
        _summary_bucket(by_layer.setdefault(shape.layer, _empty_bucket(layer=shape.layer)), row)
        _summary_bucket(
            by_target.setdefault(shape.target, _empty_bucket(target=shape.target)),
            row,
        )

    total_params = sum(int(row["params"]) for row in adapter_rows)
    tensor_bytes = sum(int(row["tensor_bytes"]) for row in adapter_rows)
    total_bytes = tensor_bytes + int(file_overhead_bytes)
    budget_slack = None
    if target_budget_bytes is not None:
        budget_slack = int(target_budget_bytes) - total_bytes

    return {
        "kind": "rank_map_budget_report",
        "status": "passed",
        "name": name,
        "profile": profile,
        "tensor_dtype": tensor_dtype,
        "alpha_dtype": alpha_dtype,
        "rank_map": normalized_rank_map,
        "alpha_map": normalized_alpha_map,
        "summary": {
            "adapter_count": len(adapter_rows),
            "rank": sum(int(row["rank"]) for row in adapter_rows),
            "params": total_params,
            "tensor_bytes": tensor_bytes,
            "file_overhead_bytes": int(file_overhead_bytes),
            "total_bytes": total_bytes,
            "target_budget_bytes": target_budget_bytes,
            "budget_slack_bytes": budget_slack,
        },
        "by_layer": [by_layer[key] for key in sorted(by_layer)],
        "by_target": [by_target[key] for key in sorted(by_target)],
        "adapters": adapter_rows,
    }


def _report_total_bytes(report: Mapping[str, Any]) -> int:
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise RankBudgetError("Budget report is missing summary.")
    return int(summary["total_bytes"])


def resolve_target_budget_bytes(
    target: str,
    shapes: Mapping[str, AdapterShape],
    *,
    profile: str,
    source_rank_map: Mapping[str, int],
    target_rank_map: Mapping[str, Any] | None = None,
    target_alpha_map: Mapping[str, Any] | None = None,
    explicit_budget_bytes: int | None = None,
    fixed_r32_percent: float | None = None,
    tensor_dtype: str = "fp16",
    alpha_dtype: str = "fp32",
    file_overhead_bytes: int = 0,
) -> dict[str, Any]:
    """Resolve a named target budget into bytes."""

    adapter_keys = sorted(source_rank_map)
    target_key = target.replace("-", "_")
    if target_key == TARGET_FIXED_R16:
        target_map = fixed_rank_map(shapes, 16, profile=profile, adapter_keys=adapter_keys)
        report = rank_map_budget_report(
            target_map,
            shapes,
            profile=profile,
            tensor_dtype=tensor_dtype,
            alpha_dtype=alpha_dtype,
            file_overhead_bytes=file_overhead_bytes,
            name="fixed_r16_budget",
        )
        return {
            "kind": TARGET_FIXED_R16,
            "budget_bytes": _report_total_bytes(report),
            "reference_summary": report["summary"],
        }
    if target_key == TARGET_FIXED_R32:
        target_map = fixed_rank_map(shapes, 32, profile=profile, adapter_keys=adapter_keys)
        report = rank_map_budget_report(
            target_map,
            shapes,
            profile=profile,
            tensor_dtype=tensor_dtype,
            alpha_dtype=alpha_dtype,
            file_overhead_bytes=file_overhead_bytes,
            name="fixed_r32_budget",
        )
        return {
            "kind": TARGET_FIXED_R32,
            "budget_bytes": _report_total_bytes(report),
            "reference_summary": report["summary"],
        }
    if target_key == TARGET_RANK_MAP:
        if target_rank_map is None:
            raise RankBudgetError("target_rank_map is required for rank-map budget targets.")
        report = rank_map_budget_report(
            target_rank_map,
            shapes,
            profile=profile,
            alpha_map=target_alpha_map,
            tensor_dtype=tensor_dtype,
            alpha_dtype=alpha_dtype,
            file_overhead_bytes=file_overhead_bytes,
            name="rank_map_budget",
        )
        return {
            "kind": TARGET_RANK_MAP,
            "budget_bytes": _report_total_bytes(report),
            "reference_summary": report["summary"],
        }
    if target_key == TARGET_BYTES:
        if explicit_budget_bytes is None:
            raise RankBudgetError("explicit_budget_bytes is required for bytes budget targets.")
        if explicit_budget_bytes < 0:
            raise RankBudgetError("explicit_budget_bytes must be >= 0.")
        return {"kind": TARGET_BYTES, "budget_bytes": int(explicit_budget_bytes)}
    if target_key in {TARGET_FIXED_R32_PERCENT, "fixed_r32_pct"}:
        if fixed_r32_percent is None:
            raise RankBudgetError("fixed_r32_percent is required for fixed-r32-percent targets.")
        fraction = float(fixed_r32_percent)
        if fraction > 1.0:
            fraction = fraction / 100.0
        if fraction < 0.0:
            raise RankBudgetError("fixed_r32_percent must be >= 0.")
        fixed = resolve_target_budget_bytes(
            TARGET_FIXED_R32,
            shapes,
            profile=profile,
            source_rank_map=source_rank_map,
            tensor_dtype=tensor_dtype,
            alpha_dtype=alpha_dtype,
            file_overhead_bytes=file_overhead_bytes,
        )
        budget_bytes = int(math.floor(int(fixed["budget_bytes"]) * fraction))
        return {
            "kind": TARGET_FIXED_R32_PERCENT,
            "budget_bytes": budget_bytes,
            "percent_of_fixed_r32": fraction,
            "fixed_r32_budget_bytes": fixed["budget_bytes"],
        }
    raise RankBudgetError(
        f"Unsupported target budget {target!r}; expected fixed-r16, fixed-r32, "
        "rank-map, bytes, or fixed-r32-percent."
    )


def _rank_bytes(
    adapter: str,
    rank: int,
    shapes: Mapping[str, AdapterShape],
    *,
    tensor_dtype: str,
    alpha_dtype: str,
) -> int:
    return estimate_adapter_bytes(
        shapes[adapter],
        rank,
        tensor_dtype=tensor_dtype,
        alpha_dtype=alpha_dtype,
    )["tensor_bytes"]


def _change_row(
    adapter: str,
    old_rank: int,
    new_rank: int,
    shapes: Mapping[str, AdapterShape],
    *,
    tensor_dtype: str,
    alpha_dtype: str,
) -> dict[str, Any]:
    old_bytes = _rank_bytes(
        adapter,
        old_rank,
        shapes,
        tensor_dtype=tensor_dtype,
        alpha_dtype=alpha_dtype,
    )
    new_bytes = _rank_bytes(
        adapter,
        new_rank,
        shapes,
        tensor_dtype=tensor_dtype,
        alpha_dtype=alpha_dtype,
    )
    shape = shapes[adapter]
    return {
        "adapter": adapter,
        "target": shape.target,
        "layer": shape.layer,
        "action": "promote" if new_rank > old_rank else "demote",
        "old_rank": old_rank,
        "new_rank": new_rank,
        "param_delta": estimate_adapter_params(shape, new_rank)
        - estimate_adapter_params(shape, old_rank),
        "byte_delta": new_bytes - old_bytes,
    }


def normalize_rank_map_to_budget(
    source_rank_map: Mapping[str, Any],
    shapes: Mapping[str, AdapterShape],
    *,
    profile: str,
    target_budget_bytes: int,
    source_alpha_map: Mapping[str, Any] | None = None,
    allow_over_budget: bool = False,
    tensor_dtype: str = "fp16",
    alpha_dtype: str = "fp32",
    file_overhead_bytes: int = 0,
    target: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a source map to fit a target byte budget."""

    normalized_source, source_alpha = validate_rank_map(
        source_rank_map,
        shapes,
        profile=profile,
        alpha_map=source_alpha_map,
    )
    if target_budget_bytes < 0:
        raise RankBudgetError("target_budget_bytes must be >= 0.")
    allowed = sorted(allowed_ranks_for(profile))
    if not allowed:
        raise RankBudgetError(f"No ranks are allowed for profile={profile}.")
    current = dict(normalized_source)
    changes: list[dict[str, Any]] = []

    def current_report() -> dict[str, Any]:
        return rank_map_budget_report(
            current,
            shapes,
            profile=profile,
            alpha_map=alpha_map_for_rank_map(current),
            tensor_dtype=tensor_dtype,
            alpha_dtype=alpha_dtype,
            file_overhead_bytes=file_overhead_bytes,
            target_budget_bytes=target_budget_bytes,
            name="normalized_rank_map",
        )

    source_report = rank_map_budget_report(
        normalized_source,
        shapes,
        profile=profile,
        alpha_map=source_alpha,
        tensor_dtype=tensor_dtype,
        alpha_dtype=alpha_dtype,
        file_overhead_bytes=file_overhead_bytes,
        target_budget_bytes=target_budget_bytes,
        name="source_rank_map",
    )
    report = current_report()
    total = _report_total_bytes(report)

    while total > target_budget_bytes:
        options: list[dict[str, Any]] = []
        for adapter in sorted(current):
            rank = current[adapter]
            lower = [candidate for candidate in allowed if candidate < rank]
            if not lower:
                continue
            next_rank = lower[-1]
            old_bytes = _rank_bytes(
                adapter,
                rank,
                shapes,
                tensor_dtype=tensor_dtype,
                alpha_dtype=alpha_dtype,
            )
            new_bytes = _rank_bytes(
                adapter,
                next_rank,
                shapes,
                tensor_dtype=tensor_dtype,
                alpha_dtype=alpha_dtype,
            )
            next_total = total - (old_bytes - new_bytes)
            options.append(
                {
                    "adapter": adapter,
                    "next_rank": next_rank,
                    "saving": old_bytes - new_bytes,
                    "next_total": next_total,
                    "slack": target_budget_bytes - next_total,
                }
            )
        if not options:
            if allow_over_budget:
                break
            raise RankBudgetError(
                "Cannot fit rank map under target budget with the selected allowed ranks."
            )
        finishers = [option for option in options if option["next_total"] <= target_budget_bytes]
        if finishers:
            chosen = sorted(finishers, key=lambda option: (option["slack"], option["adapter"]))[0]
        else:
            chosen = sorted(options, key=lambda option: (-option["saving"], option["adapter"]))[0]
        adapter = str(chosen["adapter"])
        old_rank = current[adapter]
        new_rank = int(chosen["next_rank"])
        current[adapter] = new_rank
        changes.append(
            _change_row(
                adapter,
                old_rank,
                new_rank,
                shapes,
                tensor_dtype=tensor_dtype,
                alpha_dtype=alpha_dtype,
            )
        )
        report = current_report()
        total = _report_total_bytes(report)

    while total <= target_budget_bytes:
        options = []
        for adapter in sorted(current):
            rank = current[adapter]
            higher = [candidate for candidate in allowed if candidate > rank]
            if not higher:
                continue
            next_rank = higher[0]
            old_bytes = _rank_bytes(
                adapter,
                rank,
                shapes,
                tensor_dtype=tensor_dtype,
                alpha_dtype=alpha_dtype,
            )
            new_bytes = _rank_bytes(
                adapter,
                next_rank,
                shapes,
                tensor_dtype=tensor_dtype,
                alpha_dtype=alpha_dtype,
            )
            delta = new_bytes - old_bytes
            next_total = total + delta
            if next_total <= target_budget_bytes:
                options.append(
                    {
                        "adapter": adapter,
                        "next_rank": next_rank,
                        "delta": delta,
                        "next_total": next_total,
                        "slack": target_budget_bytes - next_total,
                    }
                )
        if not options:
            break
        chosen = sorted(options, key=lambda option: (option["slack"], option["adapter"]))[0]
        adapter = str(chosen["adapter"])
        old_rank = current[adapter]
        new_rank = int(chosen["next_rank"])
        current[adapter] = new_rank
        changes.append(
            _change_row(
                adapter,
                old_rank,
                new_rank,
                shapes,
                tensor_dtype=tensor_dtype,
                alpha_dtype=alpha_dtype,
            )
        )
        report = current_report()
        total = _report_total_bytes(report)

    final_report = current_report()
    final_total = _report_total_bytes(final_report)
    if final_total > target_budget_bytes and not allow_over_budget:
        raise RankBudgetError("Normalized rank map still exceeds target budget.")

    return {
        "kind": "rank_map_budget_normalization",
        "status": "passed" if final_total <= target_budget_bytes else "over_budget_allowed",
        "profile": profile,
        "tensor_dtype": tensor_dtype,
        "alpha_dtype": alpha_dtype,
        "target": dict(target or {"kind": TARGET_BYTES, "budget_bytes": target_budget_bytes}),
        "source_summary": source_report["summary"],
        "normalized_summary": final_report["summary"],
        "changes": changes,
        "rank_map": dict(sorted(current.items())),
        "alpha_map": alpha_map_for_rank_map(current),
    }


def normalize_rank_map_to_target(
    source_rank_map: Mapping[str, Any],
    shapes: Mapping[str, AdapterShape],
    *,
    profile: str,
    target: str,
    source_alpha_map: Mapping[str, Any] | None = None,
    target_rank_map: Mapping[str, Any] | None = None,
    target_alpha_map: Mapping[str, Any] | None = None,
    explicit_budget_bytes: int | None = None,
    fixed_r32_percent: float | None = None,
    allow_over_budget: bool = False,
    tensor_dtype: str = "fp16",
    alpha_dtype: str = "fp32",
    file_overhead_bytes: int = 0,
) -> dict[str, Any]:
    """Normalize a rank map against a named target budget."""

    normalized_source, _ = validate_rank_map(
        source_rank_map,
        shapes,
        profile=profile,
        alpha_map=source_alpha_map,
    )
    target_info = resolve_target_budget_bytes(
        target,
        shapes,
        profile=profile,
        source_rank_map=normalized_source,
        target_rank_map=target_rank_map,
        target_alpha_map=target_alpha_map,
        explicit_budget_bytes=explicit_budget_bytes,
        fixed_r32_percent=fixed_r32_percent,
        tensor_dtype=tensor_dtype,
        alpha_dtype=alpha_dtype,
        file_overhead_bytes=file_overhead_bytes,
    )
    return normalize_rank_map_to_budget(
        normalized_source,
        shapes,
        profile=profile,
        target_budget_bytes=int(target_info["budget_bytes"]),
        source_alpha_map=source_alpha_map,
        allow_over_budget=allow_over_budget,
        tensor_dtype=tensor_dtype,
        alpha_dtype=alpha_dtype,
        file_overhead_bytes=file_overhead_bytes,
        target=target_info,
    )


def _same_budget_reference(
    source_rank_map: Mapping[str, Any],
    shapes: Mapping[str, AdapterShape],
    *,
    profile: str,
    source_alpha_map: Mapping[str, Any] | None,
    tensor_dtype: str,
    alpha_dtype: str,
    file_overhead_bytes: int,
) -> dict[str, Any]:
    normalized_source, normalized_alpha = validate_rank_map(
        source_rank_map,
        shapes,
        profile=profile,
        alpha_map=source_alpha_map,
    )
    reference_report = rank_map_budget_report(
        normalized_source,
        shapes,
        profile=profile,
        alpha_map=normalized_alpha,
        tensor_dtype=tensor_dtype,
        alpha_dtype=alpha_dtype,
        file_overhead_bytes=file_overhead_bytes,
        name="same_budget_reference",
    )
    budget_bytes = _report_total_bytes(reference_report)
    return {
        "rank_map": normalized_source,
        "alpha_map": normalized_alpha,
        "report": reference_report,
        "budget_bytes": budget_bytes,
    }


def _control_report(
    *,
    control: str,
    seed: int,
    reference: Mapping[str, Any],
    initial_rank_map: Mapping[str, int],
    normalized: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "rank_map_control",
        "control": control,
        "status": normalized["status"],
        "seed": int(seed),
        "profile": normalized["profile"],
        "tensor_dtype": normalized["tensor_dtype"],
        "alpha_dtype": normalized["alpha_dtype"],
        "target": {
            "kind": "source_rank_map",
            "budget_bytes": reference["budget_bytes"],
        },
        "reference_summary": reference["report"]["summary"],
        "candidate_summary": normalized["source_summary"],
        "normalized_summary": normalized["normalized_summary"],
        "changes": normalized["changes"],
        "reference_rank_map": dict(reference["rank_map"]),
        "reference_alpha_map": dict(reference["alpha_map"]),
        "initial_rank_map": dict(sorted(initial_rank_map.items())),
        "initial_alpha_map": alpha_map_for_rank_map(initial_rank_map),
        "rank_map": normalized["rank_map"],
        "alpha_map": normalized["alpha_map"],
    }
    if extra:
        payload.update(dict(extra))
    return payload


def random_same_budget_rank_map(
    source_rank_map: Mapping[str, Any],
    shapes: Mapping[str, AdapterShape],
    *,
    profile: str,
    seed: int = 0,
    source_alpha_map: Mapping[str, Any] | None = None,
    allow_over_budget: bool = False,
    tensor_dtype: str = "fp16",
    alpha_dtype: str = "fp32",
    file_overhead_bytes: int = 0,
) -> dict[str, Any]:
    """Generate a seeded random rank-map control under the source byte budget."""

    reference = _same_budget_reference(
        source_rank_map,
        shapes,
        profile=profile,
        source_alpha_map=source_alpha_map,
        tensor_dtype=tensor_dtype,
        alpha_dtype=alpha_dtype,
        file_overhead_bytes=file_overhead_bytes,
    )
    rng = random.Random(int(seed))
    allowed = sorted(int(rank) for rank in allowed_ranks_for(profile))
    initial_rank_map = {
        adapter: int(rng.choice(allowed))
        for adapter in sorted(reference["rank_map"])
    }
    normalized = normalize_rank_map_to_budget(
        initial_rank_map,
        shapes,
        profile=profile,
        target_budget_bytes=int(reference["budget_bytes"]),
        allow_over_budget=allow_over_budget,
        tensor_dtype=tensor_dtype,
        alpha_dtype=alpha_dtype,
        file_overhead_bytes=file_overhead_bytes,
        target={
            "kind": "source_rank_map",
            "budget_bytes": reference["budget_bytes"],
            "control": "random_same_budget",
        },
    )
    return _control_report(
        control="random_same_budget",
        seed=int(seed),
        reference=reference,
        initial_rank_map=initial_rank_map,
        normalized=normalized,
    )


def _shuffled_values(values: list[int], *, seed: int) -> list[int]:
    shuffled = list(values)
    if len(shuffled) <= 1 or len(set(shuffled)) <= 1:
        return shuffled
    rng = random.Random(int(seed))
    for _ in range(16):
        rng.shuffle(shuffled)
        if shuffled != values:
            return shuffled
    for offset in range(1, len(values)):
        rotated = values[offset:] + values[:offset]
        if rotated != values:
            return rotated
    return shuffled


def shuffled_discovered_rank_map(
    source_rank_map: Mapping[str, Any],
    shapes: Mapping[str, AdapterShape],
    *,
    profile: str,
    seed: int = 0,
    source_alpha_map: Mapping[str, Any] | None = None,
    allow_over_budget: bool = False,
    tensor_dtype: str = "fp16",
    alpha_dtype: str = "fp32",
    file_overhead_bytes: int = 0,
) -> dict[str, Any]:
    """Shuffle discovered ranks across adapters, then normalize to the same byte budget."""

    reference = _same_budget_reference(
        source_rank_map,
        shapes,
        profile=profile,
        source_alpha_map=source_alpha_map,
        tensor_dtype=tensor_dtype,
        alpha_dtype=alpha_dtype,
        file_overhead_bytes=file_overhead_bytes,
    )
    adapters = sorted(reference["rank_map"])
    source_ranks = [int(reference["rank_map"][adapter]) for adapter in adapters]
    shuffled_ranks = _shuffled_values(source_ranks, seed=int(seed))
    initial_rank_map = {
        adapter: int(rank)
        for adapter, rank in zip(adapters, shuffled_ranks)
    }
    normalized = normalize_rank_map_to_budget(
        initial_rank_map,
        shapes,
        profile=profile,
        target_budget_bytes=int(reference["budget_bytes"]),
        allow_over_budget=allow_over_budget,
        tensor_dtype=tensor_dtype,
        alpha_dtype=alpha_dtype,
        file_overhead_bytes=file_overhead_bytes,
        target={
            "kind": "source_rank_map",
            "budget_bytes": reference["budget_bytes"],
            "control": "shuffled_discovered",
        },
    )
    final_rank_map = normalized["rank_map"]
    shuffle_rows = [
        {
            "adapter": adapter,
            "reference_rank": int(reference["rank_map"][adapter]),
            "shuffled_rank": int(initial_rank_map[adapter]),
            "normalized_rank": int(final_rank_map[adapter]),
        }
        for adapter in adapters
    ]
    changed = sum(1 for row in shuffle_rows if row["reference_rank"] != row["shuffled_rank"])
    normalized_changed = sum(1 for row in shuffle_rows if row["shuffled_rank"] != row["normalized_rank"])
    return _control_report(
        control="shuffled_discovered",
        seed=int(seed),
        reference=reference,
        initial_rank_map=initial_rank_map,
        normalized=normalized,
        extra={
            "shuffle": shuffle_rows,
            "shuffle_changed_adapters": changed,
            "normalization_changed_adapters": normalized_changed,
            "initial_rank_multiset_matches_reference": sorted(source_ranks) == sorted(shuffled_ranks),
        },
    )


def budget_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown budget report."""

    summary = report["summary"]
    lines = [
        f"# {report.get('name') or 'Rank Map Budget Report'}",
        "",
        f"- Profile: `{report.get('profile')}`",
        f"- Tensor dtype: `{report.get('tensor_dtype')}`",
        f"- Alpha dtype: `{report.get('alpha_dtype')}`",
        f"- Adapters: {summary['adapter_count']}",
        f"- Total rank: {summary['rank']}",
        f"- LoRA params: {summary['params']}",
        f"- Tensor bytes: {summary['tensor_bytes']}",
        f"- File overhead bytes: {summary['file_overhead_bytes']}",
        f"- Total bytes: {summary['total_bytes']}",
    ]
    if summary.get("target_budget_bytes") is not None:
        lines.append(f"- Target budget bytes: {summary['target_budget_bytes']}")
        lines.append(f"- Budget slack bytes: {summary['budget_slack_bytes']}")
    lines.extend(["", "## By Target", "", "| Target | Adapters | Rank | Params | Tensor Bytes |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in report.get("by_target", []):
        lines.append(
            f"| `{row['target']}` | {row['adapter_count']} | {row['rank']} | "
            f"{row['params']} | {row['tensor_bytes']} |"
        )
    lines.extend(["", "## By Layer", "", "| Layer | Adapters | Rank | Params | Tensor Bytes |", "| ---: | ---: | ---: | ---: | ---: |"])
    for row in report.get("by_layer", []):
        lines.append(
            f"| {row['layer']} | {row['adapter_count']} | {row['rank']} | "
            f"{row['params']} | {row['tensor_bytes']} |"
        )
    return "\n".join(lines) + "\n"


def rank_control_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown report for rank-map controls."""

    reference = report["reference_summary"]
    candidate = report["candidate_summary"]
    normalized = report["normalized_summary"]
    lines = [
        f"# {str(report.get('control', 'Rank Map Control')).replace('_', ' ').title()}",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Seed: {report.get('seed')}",
        f"- Profile: `{report.get('profile')}`",
        f"- Tensor dtype: `{report.get('tensor_dtype')}`",
        f"- Alpha dtype: `{report.get('alpha_dtype')}`",
        f"- Reference total bytes: {reference['total_bytes']}",
        f"- Initial control total bytes: {candidate['total_bytes']}",
        f"- Normalized total bytes: {normalized['total_bytes']}",
        f"- Budget slack bytes: {normalized['budget_slack_bytes']}",
    ]
    if "shuffle_changed_adapters" in report:
        lines.append(f"- Shuffled adapters changed: {report['shuffle_changed_adapters']}")
        lines.append(f"- Normalization-adjusted adapters: {report['normalization_changed_adapters']}")
    lines.extend(
        [
            "",
            "## Changes",
            "",
            "| Adapter | Action | Old Rank | New Rank | Byte Delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    changes = report.get("changes", [])
    if not changes:
        lines.append("| _none_ |  |  |  |  |")
    for row in changes:
        lines.append(
            f"| `{row['adapter']}` | {row['action']} | {row['old_rank']} | "
            f"{row['new_rank']} | {row['byte_delta']} |"
        )
    if "shuffle" in report:
        lines.extend(
            [
                "",
                "## Shuffle",
                "",
                "| Adapter | Reference Rank | Shuffled Rank | Normalized Rank |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in report["shuffle"]:
            lines.append(
                f"| `{row['adapter']}` | {row['reference_rank']} | "
                f"{row['shuffled_rank']} | {row['normalized_rank']} |"
            )
    return "\n".join(lines) + "\n"


def normalization_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact Markdown normalization report."""

    source = report["source_summary"]
    normalized = report["normalized_summary"]
    target = report["target"]
    lines = [
        "# Rank Map Budget Normalization",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Target kind: `{target.get('kind')}`",
        f"- Target budget bytes: {target.get('budget_bytes')}",
        f"- Source total bytes: {source['total_bytes']}",
        f"- Normalized total bytes: {normalized['total_bytes']}",
        f"- Budget slack bytes: {normalized['budget_slack_bytes']}",
        "",
        "## Changes",
        "",
        "| Adapter | Action | Old Rank | New Rank | Byte Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    changes = report.get("changes", [])
    if not changes:
        lines.append("| _none_ |  |  |  |  |")
    for row in changes:
        lines.append(
            f"| `{row['adapter']}` | {row['action']} | {row['old_rank']} | "
            f"{row['new_rank']} | {row['byte_delta']} |"
        )
    return "\n".join(lines) + "\n"


def consumable_rank_map_payload(
    rank_map: Mapping[str, int],
    alpha_map: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return JSON payload accepted by ``packs create --rank-map-json``."""

    normalized_rank_map = {str(key): int(value) for key, value in sorted(rank_map.items())}
    normalized_alpha_map = (
        {str(key): float(value) for key, value in sorted(alpha_map.items())}
        if alpha_map is not None
        else alpha_map_for_rank_map(normalized_rank_map)
    )
    return {
        "rank_map": normalized_rank_map,
        "alpha_map": normalized_alpha_map,
    }
