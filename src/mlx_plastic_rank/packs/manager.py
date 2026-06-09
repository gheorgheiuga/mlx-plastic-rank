"""Management helpers to attach/detach LoRA packs to MLX transformer models."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import mlx.core as mx
import numpy as np

from ..rank_select import choose_rank
from .inspection import ALLOWED_RANKS, allowed_ranks_for, size_limit_for
from .io import PackMetadata, compute_sha256, load_pack, load_pack_metadata
from .lora import SLICE_MAP, LoRAFusedLinear, SliceLoRA


class PackApplicationError(RuntimeError):
    """Raised when a pack cannot be applied to a base model."""


@dataclass
class AdapterKey:
    block: int
    target: str  # e.g. attn.q_proj

    @property
    def name(self) -> str:
        return f"blocks.{self.block}.{self.target}"


@dataclass(frozen=True)
class TargetSpec:
    """Describe how a canonical target maps to an attention module slice."""

    wrapper_attr: str
    start: int
    end: int
    input_dim: int
    output_dim: int
    kind: str
    hidden_size: int
    kv_hidden: int

    def bounds(self) -> Tuple[int, int]:
        return self.start, self.end

    def default_rank(self, base_rank: int) -> int:
        candidate: float
        if self.kind in {"q", "o"}:
            candidate = float(base_rank)
        else:
            if self.hidden_size <= 0:
                candidate = float(base_rank)
            else:
                ratio = base_rank * self.kv_hidden / float(self.hidden_size)
                candidate = max(2.0, ratio)
        allowed = sorted(ALLOWED_RANKS)
        candidate_int = int(round(candidate))
        for rank in allowed:
            if rank >= candidate_int:
                return rank
        return allowed[-1]


def _get_nested_attr(obj, attr_path: str):
    target = obj
    for part in attr_path.split("."):
        target = getattr(target, part)
    return target


def _try_get_nested_attr(obj, attr_path: str):
    try:
        return _get_nested_attr(obj, attr_path)
    except AttributeError:
        return None


def _set_nested_attr(obj, attr_path: str, value) -> None:
    parts = attr_path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _candidate_configs(*objects: Any) -> List[Any]:
    configs: List[Any] = []
    seen: set[int] = set()
    for obj in objects:
        if obj is None:
            continue
        config = getattr(obj, "config", None)
        for candidate in (config, getattr(config, "text_config", None) if config is not None else None):
            if candidate is None:
                continue
            ident = id(candidate)
            if ident not in seen:
                configs.append(candidate)
                seen.add(ident)
    return configs


def _first_value(objects: Iterable[Any], *names: str) -> Any | None:
    for obj in objects:
        if obj is None:
            continue
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_quantized_linear(linear) -> bool:
    return hasattr(linear, "bits") and getattr(linear, "bits", None) is not None and hasattr(linear, "weight")


def _linear_output_dim(linear) -> int:
    weight = linear.weight
    return int(weight.shape[0])


def _linear_input_dim(linear) -> int:
    weight = linear.weight
    if _is_quantized_linear(linear) and weight.dtype == mx.uint32:
        bits = int(getattr(linear, "bits", 32))
        if bits <= 0 or 32 % bits != 0:
            raise PackApplicationError(f"Unsupported quantized bit width {bits}")
        factor = 32 // bits
        return int(weight.shape[1]) * factor
    return int(weight.shape[1])


def _linear_weight_array(linear) -> mx.array:
    if _is_quantized_linear(linear):
        biases = linear.get("biases") if isinstance(linear, dict) else getattr(linear, "biases", None)
        weight = mx.dequantize(
            linear.weight,
            linear.scales,
            biases,
            group_size=getattr(linear, "group_size", 64),
            bits=getattr(linear, "bits", 4),
            mode=getattr(linear, "mode", "affine"),
        )
        return weight.astype(mx.float32)
    return linear.weight.astype(mx.float32)


def _local_adapter_geometry(spec: TargetSpec, wrapper: LoRAFusedLinear) -> Tuple[int, int, int, int]:
    start, end = spec.bounds()
    input_dim = spec.input_dim
    output_dim = spec.output_dim
    if spec.start == 0 and spec.end == spec.output_dim and wrapper.output_dim <= spec.output_dim:
        end = wrapper.output_dim
        input_dim = wrapper.input_dim
        output_dim = wrapper.output_dim
    return start, end, input_dim, output_dim


class LoRAManager:
    """Injection manager for MLX transformer LoRA skill packs."""

    def __init__(
        self,
        model,
        base_checkpoint: Path | None = None,
        base_model: str | None = None,
    ):
        self.model = model
        self.base_checkpoint = base_checkpoint
        self.base_model = base_model or (str(base_checkpoint) if base_checkpoint else None)
        self.base_hash: str | None = None
        self._wrapped = False
        self._wrappers: Dict[Tuple[int, str], LoRAFusedLinear] = {}
        self._active_pack: str | None = None
        self._adapter_registry: Dict[str, SliceLoRA] = {}
        self.model_type = getattr(model, "model_type", None) or getattr(getattr(model, "config", None), "model_type", None)
        if base_checkpoint and base_checkpoint.exists():
            self.base_hash = compute_sha256(base_checkpoint)
        self._blocks_cache: list | None = None
        self._target_specs: Dict[str, TargetSpec] = {}
        self._wrapper_attrs: List[str] = []
        self._slice_map: Dict[str, Tuple[int, int]] = {}
        self.hidden_size = 0
        self.current_dropout = 0.0
        self._initialise_schema()

    # ------------------------------------------------------------------
    # Wrapping helpers
    # ------------------------------------------------------------------
    def _blocks(self):
        if self._blocks_cache is not None:
            return self._blocks_cache
        blocks = None
        model = self.model
        language_model = getattr(model, "language_model", None)
        if language_model is not None:
            inner = getattr(language_model, "model", None)
            if hasattr(inner, "layers"):
                blocks = inner.layers
            elif hasattr(language_model, "layers"):
                blocks = language_model.layers
        if blocks is None and hasattr(model, "model"):
            inner = model.model
            if isinstance(inner, dict) and "h" in inner:
                blocks = inner["h"]
            elif hasattr(inner, "layers"):
                blocks = inner.layers
        if blocks is None and hasattr(model, "layers"):
            blocks = model.layers
        if blocks is None:
            raise PackApplicationError("Unsupported model structure: unable to locate transformer blocks")
        self._blocks_cache = list(blocks)
        return self._blocks_cache

    def _initialise_schema(self) -> None:
        blocks = self._blocks()
        if not blocks:
            raise PackApplicationError("Model has no transformer blocks to adapt")
        block0 = blocks[0]
        attn_attr = None
        for candidate in ("self_attn", "attn"):
            if hasattr(block0, candidate):
                attn_attr = candidate
                break
        if attn_attr is None:
            raise PackApplicationError("Unsupported transformer block structure: missing attention module")

        attn_module = getattr(block0, attn_attr)

        language_model = getattr(self.model, "language_model", None)
        language_inner = getattr(language_model, "model", None) if language_model is not None else None
        configs = _candidate_configs(
            attn_module,
            block0,
            language_inner,
            language_model,
            getattr(self.model, "model", None),
            self.model,
        )
        if self.model_type is None:
            maybe_model_type = _first_value(configs, "model_type")
            if maybe_model_type is not None:
                self.model_type = str(maybe_model_type)

        lookup_objects: List[Any] = [attn_module, block0, *configs]
        hidden_size = _int_or_none(_first_value(lookup_objects, "hidden_size", "n_embd", "d_model")) or 0
        n_heads = _int_or_none(_first_value(lookup_objects, "n_heads", "num_attention_heads"))
        n_kv_heads = _int_or_none(
            _first_value(lookup_objects, "n_kv_heads", "num_key_value_heads", "num_global_key_value_heads")
        )
        head_dim = _int_or_none(_first_value(lookup_objects, "head_dim"))

        target_specs: Dict[str, TargetSpec] = {}
        wrapper_attrs: List[str] = []
        slice_map: Dict[str, Tuple[int, int]] = {}

        def register_slice(name: str, wrapper_attr: str, start: int, end: int, kind: str, input_dim: int, output_dim: int, kv_hidden: int) -> None:
            target_specs[name] = TargetSpec(
                wrapper_attr=wrapper_attr,
                start=start,
                end=end,
                input_dim=input_dim,
                output_dim=output_dim,
                kind=kind,
                hidden_size=hidden_size,
                kv_hidden=kv_hidden,
            )
            slice_map[name] = (start, end)

        if hasattr(attn_module, "c_attn"):
            qkv_linear = attn_module.c_attn
            wrapper_attr = f"{attn_attr}.c_attn"
            if wrapper_attr not in wrapper_attrs:
                wrapper_attrs.append(wrapper_attr)
            out_dim = _linear_output_dim(qkv_linear)
            input_dim = _linear_input_dim(qkv_linear)

            if hidden_size <= 0:
                hidden_size = out_dim // 3

            if n_heads is None:
                if head_dim:
                    n_heads = max(1, hidden_size // max(1, head_dim))
                else:
                    n_heads = 1
            if n_kv_heads is None:
                n_kv_heads = n_heads

            d_head = head_dim or hidden_size // max(n_heads, 1)
            kv_hidden = d_head * max(n_kv_heads, 1)

            expected = hidden_size + 2 * kv_hidden
            if expected != out_dim:
                if out_dim % 3 != 0:
                    raise PackApplicationError(
                        "Expected fused qkv projection to have q + 2*kv structure"
                    )
                slice_size = out_dim // 3
                kv_hidden = slice_size
                hidden_size = slice_size
                expected = hidden_size + 2 * kv_hidden
            if expected != out_dim:
                raise PackApplicationError(
                    f"Fused projection dimensions mismatch: expected {expected}, found {out_dim}"
                )
            starts = [0, hidden_size, hidden_size + kv_hidden]
            lengths = [hidden_size, kv_hidden, kv_hidden]
            names = ["attn.q_proj", "attn.k_proj", "attn.v_proj"]
            kinds = ["q", "k", "v"]
            for name, start, length, kind in zip(names, starts, lengths, kinds):
                register_slice(
                    name,
                    wrapper_attr,
                    start,
                    start + length,
                    kind,
                    input_dim,
                    length,
                    kv_hidden,
                )
            if hasattr(attn_module, "c_proj"):
                out_linear = attn_module.c_proj
                wrapper_attr = f"{attn_attr}.c_proj"
                if wrapper_attr not in wrapper_attrs:
                    wrapper_attrs.append(wrapper_attr)
                register_slice(
                    "attn.o_proj",
                    wrapper_attr,
                    0,
                    _linear_output_dim(out_linear),
                    "o",
                    _linear_input_dim(out_linear),
                    _linear_output_dim(out_linear),
                    kv_hidden,
                )
        else:
            module_map = (
                ("attn.q_proj", "q_proj", "q", True),
                ("attn.k_proj", "k_proj", "k", True),
                ("attn.v_proj", "v_proj", "v", True),
                ("attn.o_proj", "o_proj", "o", False),
            )
            for name, attr_name, kind, required in module_map:
                if not hasattr(attn_module, attr_name):
                    if required:
                        raise PackApplicationError(
                            f"Attention module missing expected linear '{attr_name}'"
                        )
                    continue
                linear = getattr(attn_module, attr_name)
                out_dim = _linear_output_dim(linear)
                input_dim = _linear_input_dim(linear)
                wrapper_attr = f"{attn_attr}.{attr_name}"
                if wrapper_attr not in wrapper_attrs:
                    wrapper_attrs.append(wrapper_attr)
                register_slice(
                    name,
                    wrapper_attr,
                    0,
                    out_dim,
                    kind,
                    input_dim,
                    out_dim,
                    out_dim if kind != "q" else out_dim,
                )
            q_spec = target_specs.get("attn.q_proj")
            k_spec = target_specs.get("attn.k_proj")
            o_spec = target_specs.get("attn.o_proj")

            if hidden_size <= 0:
                if o_spec is not None:
                    hidden_size = o_spec.output_dim
                elif q_spec is not None:
                    hidden_size = q_spec.input_dim

            if head_dim is None and n_heads and q_spec is not None:
                head_dim = max(1, q_spec.output_dim // max(n_heads, 1))
            if n_heads is None and q_spec is not None:
                if head_dim:
                    n_heads = max(1, q_spec.output_dim // max(head_dim, 1))
                elif hidden_size:
                    head_dim_guess = math.gcd(hidden_size, q_spec.output_dim) or hidden_size
                    n_heads = max(1, q_spec.output_dim // max(head_dim_guess, 1))
                    head_dim = head_dim_guess
            if head_dim is None and n_heads and q_spec is not None:
                head_dim = max(1, q_spec.output_dim // max(n_heads, 1))
            if n_kv_heads is None:
                if head_dim and k_spec is not None:
                    n_kv_heads = max(1, k_spec.output_dim // max(head_dim, 1))
                else:
                    n_kv_heads = n_heads

            kv_hidden = 0
            if head_dim and n_kv_heads:
                kv_hidden = head_dim * max(n_kv_heads, 1)
            elif k_spec is not None:
                kv_hidden = k_spec.output_dim
            for name, spec in target_specs.items():
                target_specs[name] = TargetSpec(
                    wrapper_attr=spec.wrapper_attr,
                    start=spec.start,
                    end=spec.end,
                    input_dim=spec.input_dim,
                    output_dim=spec.output_dim,
                    kind=spec.kind,
                    hidden_size=hidden_size,
                    kv_hidden=kv_hidden if kv_hidden else spec.output_dim,
                )

        self.hidden_size = hidden_size

        self._target_specs = target_specs
        self._wrapper_attrs = wrapper_attrs
        self._slice_map = slice_map
        # keep lora slice map in sync for utilities/tests expecting canonical slices
        SLICE_MAP.clear()
        SLICE_MAP.update(slice_map)

    def _ensure_wrapped(self) -> None:
        if self._wrapped:
            return
        for idx, block in enumerate(self._blocks()):
            for attr_path in self._wrapper_attrs:
                linear = _try_get_nested_attr(block, attr_path)
                if linear is None:
                    continue
                output_dim = _linear_output_dim(linear)
                input_dim = _linear_input_dim(linear)
                wrapper = LoRAFusedLinear(
                    linear,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    dropout=self.current_dropout,
                )
                _set_nested_attr(block, attr_path, wrapper)
                self._wrappers[(idx, attr_path)] = wrapper
        self._wrapped = True

    # ------------------------------------------------------------------
    # Adapter lifecycle
    # ------------------------------------------------------------------
    def clear(self) -> None:
        if not self._wrapped:
            return
        blocks = self._blocks()
        for (block_idx, attr_path), wrapper in self._wrappers.items():
            block = blocks[int(block_idx)]
            _set_nested_attr(block, attr_path, wrapper.base)
        self._wrapped = False
        self._wrappers.clear()
        self._adapter_registry.clear()
        self._active_pack = None
        self.current_dropout = 0.0

    def _adapter_key(self, block_idx: int, target: str) -> str:
        return f"blocks.{block_idx}.{target}"

    def iter_adapters(self) -> Iterable[Tuple[str, SliceLoRA]]:
        return self._adapter_registry.items()

    def detach_pack(self) -> None:
        for wrapper in self._wrappers.values():
            wrapper.clear_adapters()
        self.set_dropout(0.0)
        self._adapter_registry.clear()
        self._active_pack = None

    # ------------------------------------------------------------------
    # Rank selection helpers
    # ------------------------------------------------------------------
    def compute_auto_ranks(
        self,
        targets: List[str],
        *,
        strategy: str,
        target_compression: float,
        eps: float = 1e-6,
        allowed_ranks: tuple[int, ...] | None = None,
    ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        if strategy not in {"stable", "theorem"}:
            raise PackApplicationError(f"Unsupported rank strategy '{strategy}'")
        if not targets:
            raise PackApplicationError("No targets specified for auto rank selection")

        block0 = self._blocks()[0]
        rank_map: Dict[str, int] = {}
        alpha_map: Dict[str, float] = {}
        residuals: Dict[str, float] = {}

        allowed = sorted(allowed_ranks or ALLOWED_RANKS)

        for target in targets:
            spec = self._target_specs.get(target)
            if spec is None:
                raise PackApplicationError(f"Target '{target}' not supported for model type '{self.model_type}'")
            linear = _get_nested_attr(block0, spec.wrapper_attr)
            weight = _linear_weight_array(linear)
            mx.eval(weight)
            slice_weight = weight[spec.start : spec.end, :]
            matrix = np.array(slice_weight, dtype=np.float32)
            if matrix.size == 0:
                raise PackApplicationError(f"Unable to compute rank for empty slice '{target}'")
            mat_for_rank = matrix
            if strategy == "theorem":
                mat_for_rank = matrix @ matrix.T
            rank, residual = choose_rank(mat_for_rank, target_compression, strategy=strategy, eps=eps)
            if rank <= 0:
                rank = allowed[0]
            selected = allowed[-1]
            for candidate in allowed:
                if candidate >= rank:
                    selected = candidate
                    break
            rank_map[target] = selected
            alpha_map[target] = 2.0 * selected
            residuals[target] = float(residual)
        return rank_map, alpha_map, residuals

    # ------------------------------------------------------------------
    # Training preparation
    # ------------------------------------------------------------------
    def initialize_adapters(
        self,
        targets: List[str],
        rank: int,
        alpha: float,
        seed: int = 42,
        *,
        rank_map: Dict[str, int] | None = None,
        alpha_map: Dict[str, float] | None = None,
        dropout: float = 0.0,
        allowed_ranks: tuple[int, ...] | None = None,
        initial_active_rank: int | None = None,
    ) -> Dict[str, SliceLoRA]:
        self._ensure_wrapped()
        mx.random.seed(seed)
        adapters: Dict[str, SliceLoRA] = {}
        self.set_dropout(dropout)
        rank_map = dict(rank_map or {})
        alpha_map = dict(alpha_map or {})
        for target in targets:
            if target not in self._target_specs:
                raise PackApplicationError(f"Target '{target}' not supported for model type '{self.model_type}'")

        allowed = tuple(sorted(allowed_ranks or ALLOWED_RANKS))

        def _default_rank_for_target(spec: TargetSpec, base_rank: int) -> int:
            candidate: float
            if spec.kind in {"q", "o"}:
                candidate = float(base_rank)
            elif spec.hidden_size <= 0:
                candidate = float(base_rank)
            else:
                ratio = base_rank * spec.kv_hidden / float(spec.hidden_size)
                candidate = max(2.0, ratio)
            candidate_int = int(round(candidate))
            for chosen in allowed:
                if chosen >= candidate_int:
                    return chosen
            return allowed[-1]

        def _snap_active_rank(requested: int, max_rank: int) -> int:
            candidates = [candidate for candidate in allowed if candidate <= max_rank]
            if not candidates:
                return max_rank
            for candidate in candidates:
                if candidate >= requested:
                    return candidate
            return candidates[-1]

        resolved_ranks: Dict[str, int] = {}
        resolved_alphas: Dict[str, float] = {}

        def _validate_rank_alpha(key: str, local_rank: int, local_alpha: float) -> None:
            if local_rank not in allowed:
                raise PackApplicationError(
                    f"Unsupported LoRA rank {local_rank} for target {key}; allowed ranks {list(allowed)}"
                )
            expected_alpha = 2.0 * local_rank
            if local_alpha != 0.0 and not math.isclose(
                local_alpha,
                expected_alpha,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise PackApplicationError(
                    f"LoRA alpha must equal 2*rank ({expected_alpha}) for {key}; received {local_alpha}"
                )

        for target in targets:
            spec = self._target_specs[target]
            local_rank = rank_map.get(target)
            if local_rank is None:
                local_rank = _default_rank_for_target(spec, rank)
            default_alpha = alpha if (spec.kind in {"q", "o"} and target not in alpha_map) else 2.0 * local_rank
            local_alpha = alpha_map.get(target, default_alpha)
            _validate_rank_alpha(target, local_rank, local_alpha)
            resolved_ranks[target] = local_rank
            resolved_alphas[target] = float(local_alpha)
            rank_map[target] = local_rank
            alpha_map[target] = float(local_alpha)

        block_wrappers: Dict[int, Dict[str, LoRAFusedLinear]] = {}
        for (block_idx, attr_path), wrapper in self._wrappers.items():
            block_wrappers.setdefault(block_idx, {})[attr_path] = wrapper

        for idx, attr_wrappers in block_wrappers.items():
            for target in targets:
                spec = self._target_specs[target]
                if spec.wrapper_attr not in attr_wrappers:
                    continue
                wrapper = attr_wrappers[spec.wrapper_attr]
                start, end, input_dim, output_dim = _local_adapter_geometry(spec, wrapper)
                adapter_key = self._adapter_key(idx, target)
                local_rank = rank_map.get(adapter_key, resolved_ranks[target])
                local_alpha = alpha_map.get(adapter_key, 2.0 * local_rank)
                _validate_rank_alpha(adapter_key, local_rank, local_alpha)
                if local_rank <= 0:
                    continue
                A = mx.zeros((output_dim, local_rank), dtype=mx.float16)
                B = mx.random.normal((local_rank, input_dim), dtype=mx.float32) * (
                    1.0 / math.sqrt(max(input_dim, 1))
                )
                B = B.astype(mx.float16)
                adapter = SliceLoRA(
                    name=adapter_key,
                    start=start,
                    end=end,
                    rank=local_rank,
                    alpha=local_alpha,
                    A=A,
                    B=B,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
                if initial_active_rank is not None:
                    adapter.set_active_rank(_snap_active_rank(initial_active_rank, local_rank))
                wrapper.add_adapter(adapter)
                adapters[adapter.name] = adapter
                self._adapter_registry[adapter.name] = adapter
                active_note = (
                    f" active_rank={adapter.active_rank}"
                    if adapter.gates is not None
                    else ""
                )
                print(
                    f"Initialised LoRA slice {adapter.name}: kind={spec.kind} rank={adapter.rank}{active_note} alpha={adapter.alpha} slice=({adapter.start},{adapter.end})"
                )
        self._active_pack = None
        for target in targets:
            if not any(key.endswith(f".{target}") for key in adapters):
                raise PackApplicationError(
                    f"Target '{target}' was not found on any transformer block"
                )
        return adapters

    @staticmethod
    def _adapter_rank_signal(adapter: SliceLoRA) -> float:
        active_rank = adapter.active_rank
        if active_rank <= 0:
            return 0.0
        A = np.array(adapter.A[:, :active_rank], dtype=np.float32)
        B = np.array(adapter.B[:active_rank, :], dtype=np.float32)
        if A.size == 0 or B.size == 0:
            return 0.0
        column_norms = np.linalg.norm(A, axis=0)
        row_norms = np.linalg.norm(B, axis=1)
        utilities = column_norms * row_norms
        return float(np.sum(utilities))

    @staticmethod
    def _active_rank_choices(
        adapter: SliceLoRA,
        allowed_ranks: tuple[int, ...],
        min_rank: int,
    ) -> list[int]:
        allowed = sorted(rank for rank in allowed_ranks if rank <= adapter.rank)
        if not allowed:
            return [adapter.rank]
        floor = min_rank
        candidates = [rank for rank in allowed if rank >= floor]
        if candidates:
            return candidates
        return [allowed[0]]

    def adjust_dynamic_ranks(
        self,
        *,
        allowed_ranks: tuple[int, ...],
        min_rank: int,
        grow_threshold: float,
        prune_threshold: float,
    ) -> list[dict[str, Any]]:
        """Grow or shrink gated active ranks based on per-adapter signal."""

        gated_adapters = [
            adapter
            for adapter in self._adapter_registry.values()
            if adapter.gates is not None
        ]
        if not gated_adapters:
            return []

        signals = {adapter.name: self._adapter_rank_signal(adapter) for adapter in gated_adapters}
        global_signal = max(signals.values(), default=0.0)
        if global_signal <= 0.0:
            return []

        events: list[dict[str, Any]] = []
        grow_bar = max(0.0, grow_threshold) * global_signal
        prune_bar = max(0.0, prune_threshold) * global_signal
        for adapter in gated_adapters:
            choices = self._active_rank_choices(adapter, allowed_ranks, min_rank)
            active = adapter.active_rank
            if active not in choices:
                choices = sorted(set(choices + [active]))
            idx = choices.index(active)
            signal = signals[adapter.name]
            next_rank = active
            action = "keep"
            if signal <= prune_bar and idx > 0:
                next_rank = choices[idx - 1]
                action = "shrink"
            elif signal >= grow_bar and idx < len(choices) - 1:
                next_rank = choices[idx + 1]
                action = "grow"

            if next_rank != active:
                adapter.set_active_rank(next_rank)
                event = {
                    "adapter": adapter.name,
                    "action": action,
                    "from_rank": active,
                    "to_rank": next_rank,
                    "max_rank": adapter.rank,
                    "signal": signal,
                    "global_signal": global_signal,
                }
                events.append(event)
        return events

    def trainable_parameters(self) -> List[mx.array]:
        params: List[mx.array] = []
        for adapter in self._adapter_registry.values():
            params.append(adapter.A.astype(mx.float32))
            params.append(adapter.B.astype(mx.float32))
        return params

    def set_trainable_parameters(self, arrays: Iterable[mx.array]) -> None:
        iterator = iter(arrays)
        for adapter in self._adapter_registry.values():
            adapter.A = next(iterator).astype(mx.float16)
            adapter.B = next(iterator).astype(mx.float16)

    # ------------------------------------------------------------------
    # Pack application / serialization
    # ------------------------------------------------------------------
    def apply_pack(self, pack_dir: Path) -> PackMetadata:
        tensor_path = pack_dir / "pack.safetensors"
        meta_path = pack_dir / "meta.json"
        if not tensor_path.exists():
            raise PackApplicationError(f"Missing tensor file: {tensor_path}")
        if not meta_path.exists():
            raise PackApplicationError(f"Missing metadata file: {meta_path}")

        metadata = load_pack_metadata(meta_path)
        if self.base_hash and metadata.base_hash and metadata.base_hash != self.base_hash:
            raise PackApplicationError(
                f"Base hash mismatch: expected {self.base_hash}, pack built for {metadata.base_hash}"
            )
        self._validate_rank_alpha(metadata.rank_map, metadata.alpha_map, profile=metadata.profile)
        tensors = load_pack(tensor_path)
        expected_keys: set[str] = set()
        for key in metadata.rank_map:
            expected_keys.add(f"{key}.lora.A")
            expected_keys.add(f"{key}.lora.B")
            expected_keys.add(f"{key}.lora.alpha")
        unexpected = sorted(k for k in tensors.keys() if k not in expected_keys)
        if unexpected:
            raise PackApplicationError(f"Pack contains unexpected tensors: {unexpected}")
        total_bytes = sum(arr.nbytes for arr in tensors.values())
        limit = size_limit_for(metadata)
        if total_bytes > limit:
            raise PackApplicationError(
                f"Pack size {total_bytes / (1024**2):.2f} MB exceeds limit {(limit / (1024**2)):.1f} MB"
            )
        self._ensure_wrapped()
        self.detach_pack()
        for key, adapter in self._load_adapters_from_tensors(tensors, metadata).items():
            self._adapter_registry[key] = adapter
            block_idx, target = self._parse_adapter_name(key)
            spec = self._target_specs.get(target)
            if spec is None:
                raise PackApplicationError(f"Pack targets unsupported for this base: '{target}'")
            wrapper = self._wrappers.get((block_idx, spec.wrapper_attr))
            if wrapper is None:
                raise PackApplicationError(
                    f"Missing wrapper for block {block_idx} attribute '{spec.wrapper_attr}'"
                )
            wrapper.add_adapter(adapter)
            wrapper.set_dropout(0.0)
            print(
                f"Attached LoRA slice {key}: rank={adapter.rank} alpha={adapter.alpha} slice=({adapter.start},{adapter.end})"
            )
        self._active_pack = metadata.pack_name
        self.set_dropout(0.0)
        return metadata

    def _parse_adapter_name(self, name: str) -> Tuple[int, str]:
        try:
            _, block_idx, target = name.split(".", 2)
            if not target.startswith("attn"):
                raise ValueError
            return int(block_idx), target
        except ValueError as exc:
            raise PackApplicationError(f"Invalid adapter key '{name}'") from exc

    def _load_adapters_from_tensors(
        self,
        tensors: Dict[str, np.ndarray],
        metadata: PackMetadata,
    ) -> Dict[str, SliceLoRA]:

        adapters: Dict[str, SliceLoRA] = {}
        for key, rank in metadata.rank_map.items():
            A_key = f"{key}.lora.A"
            B_key = f"{key}.lora.B"
            alpha_arr = tensors.get(f"{key}.lora.alpha")
            if A_key not in tensors or B_key not in tensors or alpha_arr is None:
                raise PackApplicationError(f"Missing tensors for {key}")
            A = tensors[A_key]
            B = tensors[B_key]
            target = key.split(".", 2)[2]
            block_idx, _ = self._parse_adapter_name(key)
            spec = self._target_specs.get(target)
            if spec is None:
                raise PackApplicationError(f"Pack target '{target}' unsupported for this model")
            wrapper = self._wrappers.get((block_idx, spec.wrapper_attr))
            if wrapper is not None:
                expected_start, expected_end, expected_input, expected_output = _local_adapter_geometry(spec, wrapper)
            else:
                expected_start, expected_end = spec.bounds()
                expected_input = spec.input_dim
                expected_output = spec.output_dim
            if rank <= 0:
                raise PackApplicationError(f"Invalid LoRA rank {rank} on {key}")
            if A.ndim != 2 or B.ndim != 2:
                raise PackApplicationError(f"LoRA tensors for {key} must be rank-2 matrices")
            if A.shape[0] != expected_output:
                raise PackApplicationError(
                    f"LoRA A shape mismatch on {key}: expected first dim {expected_output}, found {A.shape[0]}"
                )
            if B.shape[1] != expected_input:
                raise PackApplicationError(
                    f"LoRA B shape mismatch on {key}: expected second dim {expected_input}, found {B.shape[1]}"
                )
            if A.shape[1] != B.shape[0]:
                raise PackApplicationError(
                    f"LoRA rank mismatch on {key}: A rank {A.shape[1]} != B rank {B.shape[0]}"
                )
            if A.shape[1] != rank:
                raise PackApplicationError(
                    f"LoRA metadata rank mismatch on {key}: meta rank {rank}, tensor rank {A.shape[1]}"
                )
            adapter = SliceLoRA(
                name=key,
                start=expected_start,
                end=expected_end,
                rank=rank,
                alpha=float(np.asarray(alpha_arr).reshape(()).item()),
                A=mx.array(A.astype(np.float16, copy=False)),
                B=mx.array(B.astype(np.float16, copy=False)),
                input_dim=expected_input,
                output_dim=expected_output,
            )
            adapters[key] = adapter
        return adapters

    def _validate_rank_alpha(
        self,
        rank_map: Dict[str, int],
        alpha_map: Dict[str, float],
        *,
        profile: str = "lite",
    ) -> None:
        try:
            allowed_ranks = tuple(sorted(allowed_ranks_for(profile)))
        except ValueError as exc:
            raise PackApplicationError(str(exc)) from exc
        for key, rank in rank_map.items():
            if rank not in allowed_ranks:
                raise PackApplicationError(
                    f"Unsupported LoRA rank {rank} on {key}; allowed ranks {list(allowed_ranks)}"
                )
            expected_alpha = 2.0 * rank
            actual_alpha = float(alpha_map.get(key, expected_alpha))
            if actual_alpha != 0.0 and not math.isclose(actual_alpha, expected_alpha, rel_tol=1e-6, abs_tol=1e-6):
                raise PackApplicationError(
                    f"LoRA alpha mismatch on {key}: expected {expected_alpha}, found {actual_alpha}"
                )

    def export_active_pack(
        self,
        name: str,
        base_dir: Path,
        notes: str = "",
        *,
        profile: str = "lite",
    ) -> Tuple[Dict[str, np.ndarray], PackMetadata]:
        if not self._adapter_registry:
            raise PackApplicationError("No active adapters to export")
        tensors: Dict[str, np.ndarray] = {}
        rank_map: Dict[str, int] = {}
        alpha_map: Dict[str, float] = {}
        target_layers: List[str] = []
        for key, adapter in self._adapter_registry.items():
            export_A, export_B, export_alpha, export_rank = adapter.export_arrays()
            rank_map[key] = export_rank
            alpha_map[key] = export_alpha
            target_layers.append(key)
            tensors[f"{key}.lora.A"] = np.array(export_A, dtype=np.float16)
            tensors[f"{key}.lora.B"] = np.array(export_B, dtype=np.float16)
            tensors[f"{key}.lora.alpha"] = np.array(export_alpha, dtype=np.float32)
        self._validate_rank_alpha(rank_map, alpha_map, profile=profile)
        metadata = PackMetadata(
            pack_name=name,
            base_hash=self.base_hash or "",
            base_model=self.base_model,
            profile=(profile or "lite").lower(),
            rank_map=rank_map,
            alpha_map=alpha_map,
            target_layers=target_layers,
            created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            notes=notes,
        )
        total_bytes = sum(arr.nbytes for arr in tensors.values())
        limit = size_limit_for(metadata)
        if total_bytes > limit:
            raise PackApplicationError(
                f"Pack size {total_bytes / (1024**2):.2f} MB exceeds limit {(limit / (1024**2)):.1f} MB"
            )
        out_dir = base_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        return tensors, metadata

    def set_dropout(self, rate: float) -> None:
        value = float(rate)
        if not math.isfinite(value) or value < 0.0 or value >= 1.0:
            raise PackApplicationError("LoRA dropout must be in the range [0.0, 1.0).")
        self.current_dropout = value
        if self._wrapped:
            for wrapper in self._wrappers.values():
                wrapper.set_dropout(self.current_dropout)

    def _slice_bounds(self, target: str) -> Tuple[int, int]:
        if target not in self._slice_map:
            raise KeyError(f"Unknown target slice '{target}'")
        return self._slice_map[target]
