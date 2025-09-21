"""LoRA adapter primitives for MLX transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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

    def as_arrays(self) -> Tuple[mx.array, mx.array]:
        return self.A, self.B

    def set_arrays(self, A: mx.array, B: mx.array) -> None:
        self.A = A
        self.B = B

    @property
    def scale(self) -> float:
        return self.alpha / max(self.rank, 1)

    def delta(self, x: mx.array) -> mx.array:
        x_fp32 = x.astype(mx.float32)
        B_fp32 = self.B.astype(mx.float32)
        A_fp32 = self.A.astype(mx.float32)
        projected = mx.matmul(x_fp32, B_fp32.T)
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

    def add_adapter(self, adapter: SliceLoRA) -> None:
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
        self.adapters[adapter.name] = adapter

    def remove_adapter(self, name: str) -> None:
        self.adapters.pop(name, None)

    def clear_adapters(self) -> None:
        self.adapters.clear()

    def active_adapters(self) -> Iterable[SliceLoRA]:
        return self.adapters.values()

    def set_dropout(self, rate: float) -> None:
        self.dropout = max(0.0, float(rate))

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
