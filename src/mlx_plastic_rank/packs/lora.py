"""LoRA adapter primitives for MLX transformers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, cast

import mlx.core as mx
import mlx.nn as nn


@dataclass
class SliceLoRA:
    """Low-rank adapter tied to a slice of a fused linear output."""

    name: str
    start: int
    end: int
    rank: int
    alpha: float
    A: mx.array
    B: mx.array
    input_dim: int
    output_dim: int
    gates: mx.array | None = None

    def as_arrays(self) -> Tuple[mx.array, mx.array]:
        return self.A, self.B

    def set_arrays(self, A: mx.array, B: mx.array) -> None:
        self.A = A
        self.B = B

    @property
    def scale(self) -> float:
        return self.alpha / max(self.rank, 1)

    @property
    def active_rank(self) -> int:
        if self.gates is None:
            return self.rank
        active = mx.sum((self.gates > 0.5).astype(mx.int32))
        return int(active.item())

    @property
    def active_component_indices(self) -> Tuple[int, ...]:
        """Return active factor-pair indices in physical tensor order."""

        if self.gates is None:
            return tuple(range(self.rank))
        mask = self.gates > 0.5
        mx.eval(mask)
        return tuple(
            index
            for index in range(self.rank)
            if bool(mask[index].item())
        )

    def component_utilities(self) -> Tuple[float, ...]:
        """Return the raw pair-norm heuristic for each factor pair.

        ``||A[:, j]|| * ||B[j, :]||`` is stable under reciprocal rescaling of
        one pair, but not under general rotations or mixing of the factors. Use
        loss ablation, gradients, or a canonicalized update for evidentiary
        claims about a component's causal value.
        """

        A = self.A.astype(mx.float32)
        B = self.B.astype(mx.float32)
        column_norms = mx.sqrt(mx.sum(A * A, axis=0))
        row_norms = mx.sqrt(mx.sum(B * B, axis=1))
        utilities = column_norms * row_norms
        mx.eval(utilities)
        return tuple(float(utilities[index].item()) for index in range(self.rank))

    def set_active_components(self, indices: Iterable[int]) -> None:
        """Activate arbitrary factor pairs without moving trainable arrays."""

        selected = tuple(int(index) for index in indices)
        if not selected:
            raise ValueError("At least one component must remain active")
        if len(set(selected)) != len(selected):
            raise ValueError("Active component indices must be unique")
        if any(index < 0 or index >= self.rank for index in selected):
            raise ValueError(
                f"Active component indices must be in [0, {self.rank - 1}], got {selected}"
            )
        active = set(selected)
        self.gates = mx.array(
            [1.0 if index in active else 0.0 for index in range(self.rank)],
            dtype=mx.float32,
        )

    def set_active_rank(self, active_rank: int) -> None:
        if active_rank <= 0 or active_rank > self.rank:
            raise ValueError(f"Active rank must be in [1, {self.rank}], got {active_rank}")
        self.set_active_components(range(active_rank))

    def export_arrays(self) -> Tuple[mx.array, mx.array, float, int]:
        active_indices = self.active_component_indices
        export_rank = len(active_indices)
        indices = mx.array(active_indices, dtype=mx.int32)
        A = mx.take(self.A, indices, axis=1)
        B = mx.take(self.B, indices, axis=0)
        export_alpha = 0.0 if self.alpha == 0.0 else self.scale * export_rank
        return A, B, float(export_alpha), export_rank

    def delta(self, x: mx.array) -> mx.array:
        x_fp32 = x.astype(mx.float32)
        if self.gates is None:
            B_fp32 = self.B.astype(mx.float32)
            A_fp32 = self.A.astype(mx.float32)
        else:
            # Select before multiplication. Dormant arrays remain stored for
            # restoration; they no longer consume projection matrix products.
            # Include every nonzero gate to preserve fractional-gate semantics.
            active = [i for i, gate in enumerate(cast(list[float], self.gates.tolist())) if gate != 0]
            if not active:
                return mx.zeros((*x.shape[:-1], self.output_dim), dtype=x.dtype)
            indices = mx.array(active, dtype=mx.int32)
            B_fp32 = mx.take(self.B, indices, axis=0).astype(mx.float32)
            A_fp32 = mx.take(self.A, indices, axis=1).astype(mx.float32)
        projected = mx.matmul(x_fp32, B_fp32.T)
        if self.gates is not None:
            projected = projected * mx.take(self.gates, indices).astype(projected.dtype)
        delta = mx.matmul(projected, A_fp32.T)
        scaled = self.scale * delta
        return scaled.astype(x.dtype)

    def to_numpy(self) -> Tuple[mx.array, mx.array, float]:
        return self.A, self.B, self.alpha


class LoRAFusedLinear(nn.Module):
    """Wrap an MLX linear layer with optional LoRA slices for q/k/v."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base = base
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adapters: Dict[str, SliceLoRA] = {}
        self.dropout = float(dropout)

    def validate_adapter(self, adapter: SliceLoRA) -> None:
        """Validate attachment geometry without mutating the wrapper."""

        if adapter.end > self.output_dim:
            raise ValueError(
                f"LoRA slice {adapter.name} end {adapter.end} exceeds output dim {self.output_dim}"
            )
        if adapter.input_dim != self.input_dim:
            raise ValueError(
                f"LoRA slice {adapter.name} input dim {adapter.input_dim} does not match base {self.input_dim}"
            )
        if adapter.output_dim != adapter.end - adapter.start:
            raise ValueError(
                f"LoRA slice {adapter.name} output dim mismatch"
            )

    def add_adapter(self, adapter: SliceLoRA) -> None:
        self.validate_adapter(adapter)
        self.adapters[adapter.name] = adapter

    def remove_adapter(self, name: str) -> None:
        self.adapters.pop(name, None)

    def clear_adapters(self) -> None:
        self.adapters.clear()

    def active_adapters(self) -> Iterable[SliceLoRA]:
        return self.adapters.values()

    def set_dropout(self, rate: float) -> None:
        value = float(rate)
        if not math.isfinite(value) or value < 0.0 or value >= 1.0:
            raise ValueError("LoRA dropout must be in the range [0.0, 1.0).")
        self.dropout = value

    def _apply_dropout(self, x: mx.array) -> mx.array:
        if self.dropout <= 0.0:
            return x
        keep_prob = 1.0 - self.dropout
        mask = (mx.random.uniform(low=0.0, high=1.0, shape=x.shape) < keep_prob).astype(x.dtype)
        return x * mask / keep_prob

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        if not self.adapters:
            return base_out
        lora_input = self._apply_dropout(x)
        segments = []
        last = 0
        for adapter in sorted(self.adapters.values(), key=lambda a: a.start):
            if adapter.start < last:
                raise ValueError("Overlapping LoRA adapters are not supported")
            if adapter.start > last:
                segments.append(base_out[..., last : adapter.start])
            delta_slice = adapter.delta(lora_input)
            base_slice = base_out[..., adapter.start : adapter.end]
            segments.append(base_slice + delta_slice)
            last = adapter.end
        if last < base_out.shape[-1]:
            segments.append(base_out[..., last:])
        return mx.concatenate(segments, axis=-1)

    def parameters(self) -> List[mx.array]:
        params: List[mx.array] = []
        for adapter in self.adapters.values():
            params.extend([adapter.A, adapter.B])
        return params

    def set_parameter_arrays(self, arrays: Iterable[mx.array]) -> None:
        arr_iter = iter(arrays)
        for adapter in self.adapters.values():
            adapter.A = next(arr_iter)
            adapter.B = next(arr_iter)


SLICE_MAP: Dict[str, Tuple[int, int]] = {
    "attn.q_proj": (0, 768),
    "attn.k_proj": (768, 768 * 2),
    "attn.v_proj": (768 * 2, 768 * 3),
}


def slice_bounds(name: str) -> Tuple[int, int]:
    if name not in SLICE_MAP:
        raise KeyError(f"Unsupported LoRA target slice '{name}'")
    return SLICE_MAP[name]
