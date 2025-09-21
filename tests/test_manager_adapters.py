import types

import mlx.core as mx
import mlx.nn as nn

import pytest

from mlx_plastic_rank.packs.manager import LoRAManager, PackApplicationError
from mlx.nn.layers.quantized import QuantizedLinear


class FusedAttention:
    def __init__(self, hidden: int):
        self.c_attn = nn.Linear(hidden, hidden * 3, bias=False)
        self.c_proj = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # Mirror GPT-style fused attention by splitting projected q/k/v slices.
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        return self.c_proj(q + k + v)


class FusedBlock:
    def __init__(self, hidden: int):
        self.attn = FusedAttention(hidden)


class FusedModel:
    def __init__(self, layers: int = 1, hidden: int = 8):
        self.model = {"h": [FusedBlock(hidden) for _ in range(layers)]}
        self.config = types.SimpleNamespace(n_embd=hidden)
        self.model_type = "gpt2"


class SeparateAttention:
    def __init__(self, hidden: int):
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(q + k + v)


class SeparateBlock:
    def __init__(self, hidden: int):
        self.self_attn = SeparateAttention(hidden)


class SeparateModel:
    def __init__(self, layers: int = 1, hidden: int = 8):
        self.model = types.SimpleNamespace(layers=[SeparateBlock(hidden) for _ in range(layers)])
        self.config = types.SimpleNamespace(hidden_size=hidden)
        self.model_type = "qwen3"


class QuantizedAttention:
    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int):
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.hidden_size = hidden
        self.q_proj = QuantizedLinear(hidden, hidden, bias=False, group_size=32, bits=4)
        kv_hidden = (hidden // n_heads) * n_kv_heads
        self.k_proj = QuantizedLinear(hidden, kv_hidden, bias=False, group_size=32, bits=4)
        self.v_proj = QuantizedLinear(hidden, kv_hidden, bias=False, group_size=32, bits=4)
        self.o_proj = nn.Linear(hidden + 2 * kv_hidden, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        concat = mx.concatenate([q, k, v], axis=-1)
        return self.o_proj(concat)


class QuantizedBlock:
    def __init__(self, hidden: int, n_heads: int, n_kv_heads: int):
        self.self_attn = QuantizedAttention(hidden, n_heads, n_kv_heads)
        self.hidden_size = hidden
        self.num_attention_heads = n_heads
        self.num_key_value_heads = n_kv_heads


class QuantizedModel:
    def __init__(self, layers: int = 1, hidden: int = 256, n_heads: int = 8, n_kv_heads: int = 2):
        self.model = types.SimpleNamespace(
            layers=[QuantizedBlock(hidden, n_heads, n_kv_heads) for _ in range(layers)]
        )
        self.model_type = "qwen3"


def _alpha_zero_noop(model, targets):
    manager = LoRAManager(model)
    hidden = getattr(manager, "hidden_size", 0) or 8
    inputs = mx.random.normal((2, hidden))
    # Capture baseline outputs prior to wrapping
    blocks = manager._blocks()
    baseline = []
    for block in blocks:
        if hasattr(block, "attn"):
            out = block.attn(inputs)
        else:
            out = block.self_attn(inputs)
        mx.eval(out)
        baseline.append(out)

    manager.initialize_adapters(targets=targets, rank=4, alpha=0.0, seed=0)

    wrapped_outputs = []
    for block in blocks:
        if hasattr(block, "attn"):
            wrapped_outputs.append(block.attn(inputs))
        else:
            wrapped_outputs.append(block.self_attn(inputs))

    mx.eval(*wrapped_outputs)

    for base, wrapped in zip(baseline, wrapped_outputs):
        assert mx.allclose(base, wrapped, atol=1e-6)


def test_fused_attention_alpha_zero_is_noop():
    model = FusedModel(hidden=8)
    _alpha_zero_noop(model, ["attn.q_proj", "attn.k_proj", "attn.v_proj"])


def test_separate_attention_alpha_zero_is_noop():
    model = SeparateModel(hidden=8)
    _alpha_zero_noop(model, ["attn.q_proj", "attn.k_proj", "attn.v_proj"])


def test_quantized_attention_alpha_zero_is_noop():
    model = QuantizedModel(hidden=256, n_heads=8, n_kv_heads=2)
    _alpha_zero_noop(model, ["attn.q_proj", "attn.k_proj", "attn.v_proj"])


def test_rejects_invalid_rank():
    model = FusedModel(hidden=8)
    manager = LoRAManager(model)
    with pytest.raises(PackApplicationError):
        manager.initialize_adapters(["attn.q_proj"], rank=6, alpha=12.0)


def test_rejects_alpha_mismatch():
    model = FusedModel(hidden=8)
    manager = LoRAManager(model)
    with pytest.raises(PackApplicationError):
        manager.initialize_adapters(["attn.q_proj"], rank=4, alpha=7.0)


def test_quantized_geometry_defaults():
    model = QuantizedModel(hidden=256, n_heads=8, n_kv_heads=2)
    manager = LoRAManager(model)
    adapters = manager.initialize_adapters(
        targets=["attn.q_proj", "attn.k_proj", "attn.v_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
    )
    q_adapter = adapters["blocks.0.attn.q_proj"]
    k_adapter = adapters["blocks.0.attn.k_proj"]
    v_adapter = adapters["blocks.0.attn.v_proj"]
    assert q_adapter.rank == 4
    assert k_adapter.rank == 2
    assert v_adapter.rank == 2
    assert q_adapter.B.shape == (4, 256)
    assert k_adapter.B.shape == (2, 256)
    assert q_adapter.A.dtype == mx.float16
    specs = manager._target_specs
    assert specs["attn.q_proj"].output_dim == 256
    assert specs["attn.k_proj"].output_dim == 64
