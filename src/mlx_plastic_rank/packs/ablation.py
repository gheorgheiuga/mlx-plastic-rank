"""Rank-channel ablation reports for LoRA packs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mlx.core as mx

from .io import PackMetadata, load_pack, load_pack_metadata, save_pack, save_pack_metadata


class AblationError(ValueError):
    """Raised when an ablation report or generated ablation pack is invalid."""


def _target_from_adapter(adapter: str) -> str:
    parts = adapter.split(".", 2)
    return parts[2] if len(parts) == 3 else adapter


def _layer_from_adapter(adapter: str) -> int | None:
    parts = adapter.split(".", 2)
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _matrix_shape(value: Any, *, key: str) -> tuple[int, int]:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 2:
        raise AblationError(f"LoRA tensor {key} must be a rank-2 array.")
    return int(shape[0]), int(shape[1])


def _numel(value: Any) -> int:
    total = 1
    for dim in getattr(value, "shape", ()):
        total *= int(dim)
    return int(total)


def _group_lora_tensors(tensors: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for key, value in tensors.items():
        if ".lora." not in key:
            continue
        prefix, suffix = key.split(".lora.", 1)
        grouped.setdefault(prefix, {})[suffix] = value
    return grouped


def _scalar_alpha(value: Any) -> float:
    alpha = mx.array(value, dtype=mx.float32)
    return float(mx.reshape(alpha, ()).item())


def _lowrank_fro_norm(left: Any, right: Any, scale: float) -> float:
    if _numel(left) == 0 or _numel(right) == 0:
        return 0.0
    left32 = mx.array(left, dtype=mx.float32)
    right32 = mx.array(right, dtype=mx.float32)
    left_gram = mx.matmul(mx.transpose(left32), left32)
    right_gram = mx.matmul(right32, mx.transpose(right32))
    fro2 = float(mx.sum(left_gram * mx.transpose(right_gram)).item())
    return abs(float(scale)) * math.sqrt(max(0.0, fro2))


def _require_adapter_parts(adapter: str, parts: Mapping[str, Any]) -> tuple[Any, Any, float]:
    missing = sorted({"A", "B", "alpha"} - set(parts))
    if missing:
        raise AblationError(f"Missing LoRA tensor(s) for {adapter}: {missing}")
    left = parts["A"]
    right = parts["B"]
    left_rows, left_cols = _matrix_shape(left, key=f"{adapter}.lora.A")
    right_rows, _ = _matrix_shape(right, key=f"{adapter}.lora.B")
    if left_cols != right_rows:
        raise AblationError(
            f"LoRA rank mismatch for {adapter}: A rank {left_cols} != B rank {right_rows}"
        )
    return left, right, _scalar_alpha(parts["alpha"])


def _adapter_rank(metadata: PackMetadata, adapter: str, actual_rank: int) -> int:
    declared = int(metadata.rank_map.get(adapter, actual_rank))
    if declared != actual_rank:
        raise AblationError(
            f"Adapter {adapter} metadata rank {declared} does not match tensor rank {actual_rank}."
        )
    return declared


def _channel_rows(
    metadata: PackMetadata,
    grouped: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    channels: list[dict[str, Any]] = []
    adapters: dict[str, dict[str, Any]] = {}
    for adapter in sorted(grouped):
        left, right, alpha = _require_adapter_parts(adapter, grouped[adapter])
        rank = _adapter_rank(metadata, adapter, int(left.shape[1]))
        scale = 0.0 if alpha == 0.0 else alpha / float(max(rank, 1))
        left32 = mx.array(left, dtype=mx.float32)
        right32 = mx.array(right, dtype=mx.float32)
        adapter_channels = []
        for channel in range(rank):
            left_channel = left32[:, channel]
            right_channel = right32[channel, :]
            a_norm = float(mx.sqrt(mx.sum(left_channel * left_channel)).item())
            b_norm = float(mx.sqrt(mx.sum(right_channel * right_channel)).item())
            proxy = abs(scale) * a_norm * b_norm
            row = {
                "adapter": adapter,
                "target": _target_from_adapter(adapter),
                "layer": _layer_from_adapter(adapter),
                "channel": channel,
                "rank": rank,
                "alpha": alpha,
                "scale": scale,
                "a_norm": a_norm,
                "b_norm": b_norm,
                "proxy_update_fro": proxy,
            }
            channels.append(row)
            adapter_channels.append(row)
        adapter_fro = _lowrank_fro_norm(left, right, scale)
        adapters[adapter] = {
            "adapter": adapter,
            "target": _target_from_adapter(adapter),
            "layer": _layer_from_adapter(adapter),
            "rank": rank,
            "alpha": alpha,
            "scale": scale,
            "proxy_update_fro": adapter_fro,
            "channels": adapter_channels,
        }
    return channels, adapters


def _selected_adapters(
    adapters: Mapping[str, Mapping[str, Any]],
    *,
    targets: Iterable[str] | None,
    layers: Iterable[int] | None,
) -> list[str]:
    target_set = {str(target) for target in targets or []}
    layer_set = {int(layer) for layer in layers or []}
    selected = []
    for adapter, row in sorted(adapters.items()):
        if target_set and row.get("target") not in target_set:
            continue
        if layer_set and row.get("layer") not in layer_set:
            continue
        selected.append(adapter)
    return selected


def _unit_fro(adapters: Mapping[str, Mapping[str, Any]], channels_by_adapter: Mapping[str, Sequence[int]]) -> float:
    total2 = 0.0
    for adapter, channels in channels_by_adapter.items():
        if not channels:
            continue
        row = adapters[adapter]
        channel_rows = {int(item["channel"]): item for item in row["channels"]}
        for channel in channels:
            proxy = float(channel_rows[int(channel)]["proxy_update_fro"])
            total2 += proxy * proxy
    return math.sqrt(total2)


def _make_unit(
    *,
    unit: str,
    operation: str,
    ablation_id: str,
    channels_by_adapter: Mapping[str, Sequence[int]],
    adapters: Mapping[str, Mapping[str, Any]],
    target: str | None = None,
    layer: int | None = None,
    channel: int | None = None,
    prefix_rank: int | None = None,
) -> dict[str, Any]:
    adapter_names = sorted(channels_by_adapter)
    channel_count = sum(len(channels) for channels in channels_by_adapter.values())
    proxy = _unit_fro(adapters, channels_by_adapter)
    return {
        "ablation_id": ablation_id,
        "unit": unit,
        "operation": operation,
        "adapters": adapter_names,
        "adapter_count": len(adapter_names),
        "target": target,
        "layer": layer,
        "channel": channel,
        "prefix_rank": prefix_rank,
        "channel_count": channel_count,
        "channels_by_adapter": {
            adapter: [int(value) for value in channels_by_adapter[adapter]]
            for adapter in adapter_names
        },
        "proxy_removed_fro": proxy,
    }


def _sort_units(units: list[dict[str, Any]], top_k: int | None) -> list[dict[str, Any]]:
    sorted_units = sorted(
        units,
        key=lambda row: (
            -float(row["proxy_removed_fro"]),
            str(row["ablation_id"]),
        ),
    )
    if top_k is None or top_k <= 0:
        return sorted_units
    return sorted_units[:top_k]


def _build_units(
    unit: str,
    adapters: Mapping[str, Mapping[str, Any]],
    *,
    targets: Iterable[str] | None,
    layers: Iterable[int] | None,
    top_k: int | None,
    prefix_rank: int | None,
) -> list[dict[str, Any]]:
    selected = _selected_adapters(adapters, targets=targets, layers=layers)
    units: list[dict[str, Any]] = []
    if unit == "channel":
        for adapter in selected:
            for row in adapters[adapter]["channels"]:
                channel = int(row["channel"])
                units.append(
                    _make_unit(
                        unit="channel",
                        operation="zero_channel",
                        ablation_id=f"channel-{_slug(adapter)}-c{channel:04d}",
                        channels_by_adapter={adapter: [channel]},
                        adapters=adapters,
                        target=str(row["target"]),
                        layer=row["layer"],
                        channel=channel,
                    )
                )
    elif unit == "adapter":
        for adapter in selected:
            channels = [int(row["channel"]) for row in adapters[adapter]["channels"]]
            units.append(
                _make_unit(
                    unit="adapter",
                    operation="zero_adapter_channels",
                    ablation_id=f"adapter-{_slug(adapter)}",
                    channels_by_adapter={adapter: channels},
                    adapters=adapters,
                    target=str(adapters[adapter]["target"]),
                    layer=adapters[adapter]["layer"],
                )
            )
    elif unit == "target":
        by_target: dict[str, dict[str, list[int]]] = {}
        for adapter in selected:
            target = str(adapters[adapter]["target"])
            by_target.setdefault(target, {})[adapter] = [
                int(row["channel"]) for row in adapters[adapter]["channels"]
            ]
        for target, channels_by_adapter in sorted(by_target.items()):
            units.append(
                _make_unit(
                    unit="target",
                    operation="zero_target_channels",
                    ablation_id=f"target-{_slug(target)}",
                    channels_by_adapter=channels_by_adapter,
                    adapters=adapters,
                    target=target,
                )
            )
    elif unit == "layer":
        by_layer: dict[int, dict[str, list[int]]] = {}
        for adapter in selected:
            layer = adapters[adapter]["layer"]
            if layer is None:
                continue
            by_layer.setdefault(int(layer), {})[adapter] = [
                int(row["channel"]) for row in adapters[adapter]["channels"]
            ]
        for layer, channels_by_adapter in sorted(by_layer.items()):
            units.append(
                _make_unit(
                    unit="layer",
                    operation="zero_layer_channels",
                    ablation_id=f"layer-{layer:04d}",
                    channels_by_adapter=channels_by_adapter,
                    adapters=adapters,
                    layer=layer,
                )
            )
    elif unit == "prefix":
        if prefix_rank is None or prefix_rank < 0:
            raise AblationError("--prefix-rank must be >= 0 for prefix ablations.")
        for adapter in selected:
            rank = int(adapters[adapter]["rank"])
            if prefix_rank >= rank:
                continue
            channels = list(range(int(prefix_rank), rank))
            units.append(
                _make_unit(
                    unit="prefix",
                    operation="zero_suffix_after_prefix",
                    ablation_id=f"prefix-{_slug(adapter)}-keep{int(prefix_rank):04d}",
                    channels_by_adapter={adapter: channels},
                    adapters=adapters,
                    target=str(adapters[adapter]["target"]),
                    layer=adapters[adapter]["layer"],
                    prefix_rank=int(prefix_rank),
                )
            )
    else:
        raise AblationError(
            f"Unsupported ablation unit {unit!r}; expected channel, adapter, target, layer, or prefix."
        )
    return _sort_units(units, top_k)


def _copy_tensors(tensors: Mapping[str, Any]) -> dict[str, Any]:
    # SafeTensors pack IO currently enters through safetensors.numpy. Keep these
    # arrays in serialized form for copying/writing only; tensor scoring uses MLX.
    return {
        key: value.copy() if hasattr(value, "copy") else value
        for key, value in tensors.items()
    }


def write_ablated_pack(
    source_pack_dir: Path,
    unit_row: Mapping[str, Any],
    out_root: Path,
) -> Path:
    """Write an ablated pack copy without mutating the source pack."""

    metadata = load_pack_metadata(source_pack_dir / "meta.json")
    tensors = _copy_tensors(load_pack(source_pack_dir / "pack.safetensors"))
    channels_by_adapter = unit_row.get("channels_by_adapter")
    if not isinstance(channels_by_adapter, Mapping):
        raise AblationError("Ablation unit is missing channels_by_adapter.")
    for adapter, raw_channels in channels_by_adapter.items():
        channels = [int(value) for value in raw_channels]
        if not channels:
            continue
        a_key = f"{adapter}.lora.A"
        b_key = f"{adapter}.lora.B"
        if a_key not in tensors or b_key not in tensors:
            raise AblationError(f"Ablation adapter {adapter!r} is not present in source tensors.")
        left = tensors[a_key]
        right = tensors[b_key]
        for channel in channels:
            if channel < 0 or channel >= left.shape[1] or channel >= right.shape[0]:
                raise AblationError(f"Invalid channel {channel} for adapter {adapter}.")
        left[:, channels] = 0
        right[channels, :] = 0
        tensors[a_key] = left
        tensors[b_key] = right

    ablation_id = str(unit_row["ablation_id"])
    out_dir = out_root / ablation_id
    metadata_payload = metadata.to_dict()
    metadata_payload["pack_name"] = ablation_id
    metadata_payload["notes"] = (
        f"{metadata.notes}\nAblation copy of {metadata.pack_name}: {ablation_id}"
        if metadata.notes
        else f"Ablation copy of {metadata.pack_name}: {ablation_id}"
    )
    training_config = dict(metadata.training_config)
    training_config["ablation"] = {
        "source_pack": metadata.pack_name,
        "ablation_id": ablation_id,
        "unit": unit_row.get("unit"),
        "operation": unit_row.get("operation"),
        "channels_by_adapter": unit_row.get("channels_by_adapter"),
    }
    metadata_payload["training_config"] = training_config
    ablated_metadata = PackMetadata.from_dict(metadata_payload)
    save_pack(tensors, out_dir / "pack.safetensors")
    save_pack_metadata(ablated_metadata, out_dir / "meta.json")
    return out_dir


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AblationError(f"Eval report not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AblationError(f"Eval report is invalid JSON: {path}: {exc}") from exc


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _eval_metrics_from_report(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    rows: list[Mapping[str, Any]]
    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, Mapping)]
    elif isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        rows = [row for row in payload["rows"] if isinstance(row, Mapping)]
    elif isinstance(payload, Mapping):
        rows = [payload]
    else:
        rows = []
    if not rows:
        raise AblationError(f"Eval report has no metric rows: {path}")
    pack_rows = [row for row in rows if row.get("pack") not in (None, "")]
    row = pack_rows[-1] if pack_rows else rows[-1]
    return {
        "source": str(path),
        "pack": row.get("pack"),
        "perplexity": _as_float(row.get("perplexity")),
        "token_accuracy": _as_float(row.get("token_accuracy")),
        "domain_metric": _as_float(row.get("domain_metric")),
        "peak_memory_gb": _as_float(row.get("peak_memory_gb")),
        "tokens_per_sec": _as_float(row.get("tokens_per_sec")),
    }


def _metric_deltas(baseline: Mapping[str, Any], ablated: Mapping[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {"ablated_metrics": dict(ablated)}
    for metric in ("perplexity", "token_accuracy", "domain_metric", "tokens_per_sec"):
        base_value = _as_float(baseline.get(metric))
        ablated_value = _as_float(ablated.get(metric))
        delta = None
        delta_pct = None
        if base_value is not None and ablated_value is not None:
            delta = ablated_value - base_value
            delta_pct = (delta / base_value) * 100.0 if base_value else None
        deltas[f"{metric}_delta"] = delta
        deltas[f"{metric}_delta_pct"] = delta_pct
    return deltas


def _attach_eval_metrics(
    report: dict[str, Any],
    *,
    baseline_eval: Path | None,
    ablation_eval_reports: Mapping[str, Path] | None,
) -> None:
    if baseline_eval is None and not ablation_eval_reports:
        report["evidence_status"] = "mechanistic_proxy_only"
        report["causal_note"] = (
            "This report ranks proposed interventions by removed LoRA update norm. "
            "Run paired evals for generated ablation packs before making causal quality claims."
        )
        return
    if baseline_eval is None:
        raise AblationError("baseline_eval is required when ablation eval reports are provided.")
    baseline_metrics = _eval_metrics_from_report(baseline_eval)
    report["baseline_eval"] = baseline_metrics
    matched = 0
    for unit in report["ablations"]:
        ablation_id = str(unit["ablation_id"])
        path = (ablation_eval_reports or {}).get(ablation_id)
        if path is None:
            continue
        unit["eval"] = _metric_deltas(baseline_metrics, _eval_metrics_from_report(path))
        matched += 1
    report["evidence_status"] = "paired_eval" if matched else "baseline_eval_only"
    report["causal_note"] = (
        "Causal evidence is local to the paired eval artifacts and selected examples. "
        "Use matched prompts/seeds for baseline and ablated packs."
    )


def build_rank_channel_ablation_report(
    pack_dir: Path,
    *,
    unit: str = "channel",
    top_k: int | None = 20,
    prefix_rank: int | None = None,
    targets: Iterable[str] | None = None,
    layers: Iterable[int] | None = None,
    ablation_pack_root: Path | None = None,
    baseline_eval: Path | None = None,
    ablation_eval_reports: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Build a rank-channel ablation report and optional ablated pack copies."""

    metadata = load_pack_metadata(pack_dir / "meta.json")
    tensors = load_pack(pack_dir / "pack.safetensors")
    grouped = _group_lora_tensors(tensors)
    if not grouped:
        raise AblationError(f"Pack has no LoRA tensors: {pack_dir}")
    channels, adapters = _channel_rows(metadata, grouped)
    units = _build_units(
        unit,
        adapters,
        targets=targets,
        layers=layers,
        top_k=top_k,
        prefix_rank=prefix_rank,
    )
    total_proxy2 = sum(float(adapter["proxy_update_fro"]) ** 2 for adapter in adapters.values())
    total_proxy = math.sqrt(total_proxy2)
    for row in units:
        row["proxy_removed_fro_share"] = (
            float(row["proxy_removed_fro"]) / total_proxy if total_proxy else 0.0
        )
        if ablation_pack_root is not None:
            row["generated_pack_dir"] = str(write_ablated_pack(pack_dir, row, ablation_pack_root))

    report: dict[str, Any] = {
        "kind": "causal_rank_channel_ablation_report",
        "source_pack_dir": str(pack_dir),
        "metadata": metadata.to_dict(),
        "policy": {
            "unit": unit,
            "top_k": top_k,
            "prefix_rank": prefix_rank,
            "targets": sorted(str(target) for target in targets or []),
            "layers": sorted(int(layer) for layer in layers or []),
        },
        "summary": {
            "adapter_count": len(adapters),
            "channel_count": len(channels),
            "ablation_count": len(units),
            "total_proxy_update_fro": total_proxy,
            "generated_pack_count": sum(1 for row in units if row.get("generated_pack_dir")),
        },
        "adapters": [
            {
                "adapter": row["adapter"],
                "target": row["target"],
                "layer": row["layer"],
                "rank": row["rank"],
                "alpha": row["alpha"],
                "proxy_update_fro": row["proxy_update_fro"],
            }
            for row in sorted(adapters.values(), key=lambda item: item["adapter"])
        ],
        "channels": sorted(
            channels,
            key=lambda row: (-float(row["proxy_update_fro"]), str(row["adapter"]), int(row["channel"])),
        ),
        "ablations": units,
    }
    _attach_eval_metrics(
        report,
        baseline_eval=baseline_eval,
        ablation_eval_reports=ablation_eval_reports,
    )
    return report


def ablation_rows_for_csv(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in report.get("ablations", []):
        eval_row = row.get("eval") if isinstance(row.get("eval"), Mapping) else {}
        rows.append(
            {
                "ablation_id": row.get("ablation_id"),
                "unit": row.get("unit"),
                "operation": row.get("operation"),
                "adapter_count": row.get("adapter_count"),
                "adapters": ",".join(row.get("adapters", [])),
                "target": row.get("target"),
                "layer": row.get("layer"),
                "channel": row.get("channel"),
                "prefix_rank": row.get("prefix_rank"),
                "channel_count": row.get("channel_count"),
                "proxy_removed_fro": row.get("proxy_removed_fro"),
                "proxy_removed_fro_share": row.get("proxy_removed_fro_share"),
                "generated_pack_dir": row.get("generated_pack_dir"),
                "perplexity_delta": eval_row.get("perplexity_delta"),
                "token_accuracy_delta": eval_row.get("token_accuracy_delta"),
                "domain_metric_delta": eval_row.get("domain_metric_delta"),
            }
        )
    return rows


def ablation_report_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Causal Rank-Channel Ablation Report",
        "",
        f"- Evidence status: `{report.get('evidence_status')}`",
        f"- Source pack: `{report.get('source_pack_dir')}`",
        f"- Unit: `{report.get('policy', {}).get('unit')}`",
        f"- Adapters: {summary['adapter_count']}",
        f"- Channels: {summary['channel_count']}",
        f"- Ablations: {summary['ablation_count']}",
        f"- Generated packs: {summary['generated_pack_count']}",
        "",
        str(report.get("causal_note", "")),
        "",
        "## Ablations",
        "",
        "| ID | Unit | Target | Layer | Channel | Channels | Proxy Removed Fro | Share |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("ablations", []):
        lines.append(
            f"| `{row['ablation_id']}` | {row['unit']} | {row.get('target')} | "
            f"{row.get('layer')} | {row.get('channel')} | {row['channel_count']} | "
            f"{row['proxy_removed_fro']:.6g} | {row['proxy_removed_fro_share']:.6g} |"
        )
    return "\n".join(lines) + "\n"
