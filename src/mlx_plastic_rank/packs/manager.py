"""Management helpers to attach/detach LoRA packs to MLX transformer models."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mlx.core as mx
import numpy as np

from ..rank_select import choose_rank
from .inspection import ALLOWED_RANKS, size_limit_for
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
        if self.kind == "q":
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


def _set_nested_attr(obj, attr_path: str, value) -> None:
    parts = attr_path.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


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


class LoRAManager:
    """Injection manager for GPT-2 LoRA skill packs."""

    def __init__(self, model, base_checkpoint: Path | None = None):
        self.model = model
        self.base_checkpoint = base_checkpoint
        self.base_hash: str | None = None
        self._wrapped = False
        self._wrappers: Dict[Tuple[int, str], LoRAFusedLinear] = {}
        self._active_pack: str | None = None
        self._adapter_registry: Dict[str, SliceLoRA] = {}
        self.model_type = getattr(model, "model_type", None)
        if base_checkpoint:
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
        if hasattr(model, "model"):
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

        hidden_size = int(getattr(block0, "hidden_size", 0) or getattr(attn_module, "hidden_size", 0) or 0)
        n_heads = getattr(attn_module, "n_heads", None) or getattr(block0, "num_attention_heads", None)
        n_kv_heads = (
            getattr(attn_module, "n_kv_heads", None)
            or getattr(attn_module, "num_key_value_heads", None)
            or getattr(block0, "num_key_value_heads", None)
        )

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
                # Fall back to assuming equal heads when not provided
                n_heads = max(1, hidden_size // max(1, math.gcd(hidden_size, hidden_size)))
            if n_kv_heads is None:
                n_kv_heads = n_heads

            d_head = hidden_size // max(n_heads, 1)
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
        else:
            module_map = {
                "attn.q_proj": ("q_proj", "q"),
                "attn.k_proj": ("k_proj", "k"),
                "attn.v_proj": ("v_proj", "v"),
            }
            for name, (attr_name, kind) in module_map.items():
                if not hasattr(attn_module, attr_name):
                    raise PackApplicationError(
                        f"Attention module missing expected linear '{attr_name}'"
                    )
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
                if kind == "q" and hidden_size <= 0:
                    hidden_size = out_dim
            if hasattr(attn_module, "o_proj"):
                hidden_size = _linear_output_dim(attn_module.o_proj)

            if n_heads is None and hidden_size:
                q_spec = target_specs.get("attn.q_proj")
                if q_spec:
                    head_dim_guess = math.gcd(hidden_size, q_spec.output_dim)
                    head_dim_guess = head_dim_guess or hidden_size
                    n_heads = max(1, q_spec.output_dim // head_dim_guess)
            if n_kv_heads is None:
                n_kv_heads = n_heads

            kv_hidden = 0
            if hidden_size and n_heads and n_kv_heads:
                d_head = hidden_size // max(n_heads, 1)
                kv_hidden = d_head * max(n_kv_heads, 1)
            for name, spec in target_specs.items():
                if spec.kind != "q":
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
                linear = _get_nested_attr(block, attr_path)
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
    ) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
        if strategy not in {"stable", "theorem"}:
            raise PackApplicationError(f"Unsupported rank strategy '{strategy}'")
        if not targets:
            raise PackApplicationError("No targets specified for auto rank selection")

        block0 = self._blocks()[0]
        rank_map: Dict[str, int] = {}
        alpha_map: Dict[str, float] = {}
        residuals: Dict[str, float] = {}

        allowed = sorted(ALLOWED_RANKS)

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

        resolved_ranks: Dict[str, int] = {}
        resolved_alphas: Dict[str, float] = {}
        for target in targets:
            spec = self._target_specs[target]
            local_rank = rank_map.get(target)
            if local_rank is None:
                local_rank = spec.default_rank(rank)
            if local_rank not in ALLOWED_RANKS:
                raise PackApplicationError(
                    f"Unsupported LoRA rank {local_rank} for target {target}; allowed ranks {sorted(ALLOWED_RANKS)}"
                )
            default_alpha = alpha if (spec.kind == "q" and target not in alpha_map) else 2.0 * local_rank
            local_alpha = alpha_map.get(target, default_alpha)
            expected_alpha = 2.0 * local_rank
            if local_alpha != 0.0 and not math.isclose(
                local_alpha,
                expected_alpha,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise PackApplicationError(
                    f"LoRA alpha must equal 2*rank ({expected_alpha}) for {target}; received {local_alpha}"
                )
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
                    raise PackApplicationError(
                        f"Wrapper '{spec.wrapper_attr}' not initialised for block {idx}"
                    )
                wrapper = attr_wrappers[spec.wrapper_attr]
                start, end = spec.bounds()
                local_rank = resolved_ranks[target]
                local_alpha = resolved_alphas[target]
                if local_rank <= 0:
                    continue
                input_dim = spec.input_dim
                output_dim = spec.output_dim
                A = mx.zeros((output_dim, local_rank), dtype=mx.float16)
                B = mx.random.normal((local_rank, input_dim), dtype=mx.float32) * (1.0 / max(input_dim, 1))
                B = B.astype(mx.float16)
                adapter = SliceLoRA(
                    name=self._adapter_key(idx, target),
                    start=start,
                    end=end,
                    rank=local_rank,
                    alpha=local_alpha,
                    A=A,
                    B=B,
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
                wrapper.add_adapter(adapter)
                adapters[adapter.name] = adapter
                self._adapter_registry[adapter.name] = adapter
                print(
                    f"Initialised LoRA slice {adapter.name}: kind={spec.kind} rank={adapter.rank} alpha={adapter.alpha} slice=({adapter.start},{adapter.end})"
                )
        self._active_pack = None
        return adapters

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
        self._validate_rank_alpha(metadata.rank_map, metadata.alpha_map)
        tensors = load_pack(tensor_path)
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
            spec = self._target_specs.get(target)
            if spec is None:
                raise PackApplicationError(f"Pack target '{target}' unsupported for this model")
            start, end = spec.bounds()
            adapter = SliceLoRA(
                name=key,
                start=start,
                end=end,
                rank=rank,
                alpha=float(np.asarray(alpha_arr).reshape(()).item()),
                A=mx.array(A.astype(np.float16, copy=False)),
                B=mx.array(B.astype(np.float16, copy=False)),
                input_dim=spec.input_dim,
                output_dim=spec.output_dim,
            )
            adapters[key] = adapter
        return adapters

    def _validate_rank_alpha(self, rank_map: Dict[str, int], alpha_map: Dict[str, float]) -> None:
        for key, rank in rank_map.items():
            if rank not in ALLOWED_RANKS:
                raise PackApplicationError(
                    f"Unsupported LoRA rank {rank} on {key}; allowed ranks {sorted(ALLOWED_RANKS)}"
                )
            expected_alpha = 2.0 * rank
            actual_alpha = float(alpha_map.get(key, expected_alpha))
            if actual_alpha != 0.0 and not math.isclose(actual_alpha, expected_alpha, rel_tol=1e-6, abs_tol=1e-6):
                raise PackApplicationError(
                    f"LoRA alpha mismatch on {key}: expected {expected_alpha}, found {actual_alpha}"
                )

    def export_active_pack(self, name: str, base_dir: Path, notes: str = "") -> Tuple[Dict[str, np.ndarray], PackMetadata]:
        if not self._adapter_registry:
            raise PackApplicationError("No active adapters to export")
        tensors: Dict[str, np.ndarray] = {}
        rank_map: Dict[str, int] = {}
        alpha_map: Dict[str, float] = {}
        target_layers: List[str] = []
        for key, adapter in self._adapter_registry.items():
            rank_map[key] = adapter.rank
            alpha_map[key] = adapter.alpha
            target_layers.append(key)
            tensors[f"{key}.lora.A"] = np.array(adapter.A, dtype=np.float16)
            tensors[f"{key}.lora.B"] = np.array(adapter.B, dtype=np.float16)
            tensors[f"{key}.lora.alpha"] = np.array(adapter.alpha, dtype=np.float32)
        self._validate_rank_alpha(rank_map, alpha_map)
        metadata = PackMetadata(
            pack_name=name,
            base_hash=self.base_hash or "",
            base_model=str(self.base_checkpoint) if self.base_checkpoint else None,
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
        self.current_dropout = max(0.0, float(rate))
        if self._wrapped:
            for wrapper in self._wrappers.values():
                wrapper.set_dropout(self.current_dropout)

    def _slice_bounds(self, target: str) -> Tuple[int, int]:
        if target not in self._slice_map:
            raise KeyError(f"Unknown target slice '{target}'")
        return self._slice_map[target]
