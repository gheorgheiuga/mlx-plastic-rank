import types

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.nn.layers.quantized import QuantizedLinear

from mlx_plastic_rank.packs.manager import LoRAManager, PackApplicationError


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


class GemmaUnifiedAttention:
    def __init__(
        self,
        hidden: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        *,
        use_k_eq_v: bool = False,
    ):
        q_hidden = n_heads * head_dim
        kv_hidden = n_kv_heads * head_dim
        self.use_k_eq_v = use_k_eq_v
        self.q_proj = nn.Linear(hidden, q_hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_hidden, bias=False)
        if not use_k_eq_v:
            self.v_proj = nn.Linear(hidden, kv_hidden, bias=False)
        self.o_proj = nn.Linear(q_hidden, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.o_proj(self.q_proj(x))


class GemmaUnifiedBlock:
    def __init__(
        self,
        hidden: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        *,
        use_k_eq_v: bool = False,
    ):
        self.self_attn = GemmaUnifiedAttention(
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            use_k_eq_v=use_k_eq_v,
        )


class GemmaUnifiedModel:
    def __init__(
        self,
        layers: int = 2,
        hidden: int = 16,
        n_heads: int = 4,
        n_kv_heads: int = 1,
        head_dim: int = 4,
        missing_v_layers: tuple[int, ...] = (),
        global_kv_heads: int = 1,
    ):
        text_config = types.SimpleNamespace(
            model_type="gemma4_text",
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=head_dim,
        )
        self.config = types.SimpleNamespace(model_type="gemma4_unified", text_config=text_config)
        self.language_model = types.SimpleNamespace(
            model=types.SimpleNamespace(
                layers=[
                    GemmaUnifiedBlock(
                        hidden,
                        n_heads,
                        global_kv_heads if index in missing_v_layers else n_kv_heads,
                        head_dim,
                        use_k_eq_v=index in missing_v_layers,
                    )
                    for index in range(layers)
                ]
            ),
            config=text_config,
        )


def _eye_rank(dim: int, rank: int) -> mx.array:
    mat = mx.zeros((dim, dim), dtype=mx.float32)
    for i in range(min(rank, dim)):
        mat[i, i] = 1.0
    return mat


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


def test_gemma_unified_attention_alpha_zero_is_noop():
    model = GemmaUnifiedModel()
    _alpha_zero_noop(model, ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"])


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


def test_gemma_unified_discovers_language_tower_and_projection_geometry():
    model = GemmaUnifiedModel()
    manager = LoRAManager(model)

    assert len(manager._blocks()) == 2
    assert manager.model_type == "gemma4_unified"
    assert manager.hidden_size == 16
    specs = manager._target_specs
    assert set(specs) == {"attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"}
    assert specs["attn.q_proj"].wrapper_attr == "self_attn.q_proj"
    assert specs["attn.q_proj"].input_dim == 16
    assert specs["attn.q_proj"].output_dim == 16
    assert specs["attn.k_proj"].output_dim == 4
    assert specs["attn.v_proj"].output_dim == 4
    assert specs["attn.o_proj"].input_dim == 16
    assert specs["attn.o_proj"].output_dim == 16
    assert specs["attn.k_proj"].kv_hidden == 4


def test_gemma_unified_default_ranks_downrank_kv_and_keep_output_rank():
    model = GemmaUnifiedModel()
    manager = LoRAManager(model)
    adapters = manager.initialize_adapters(
        targets=["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
    )

    assert adapters["blocks.0.attn.q_proj"].rank == 4
    assert adapters["blocks.0.attn.k_proj"].rank == 2
    assert adapters["blocks.0.attn.v_proj"].rank == 2
    assert adapters["blocks.0.attn.o_proj"].rank == 4
    assert adapters["blocks.0.attn.q_proj"].B.shape == (4, 16)
    assert adapters["blocks.0.attn.k_proj"].A.shape == (4, 2)
    assert adapters["blocks.0.attn.o_proj"].A.shape == (16, 4)
    assert adapters["blocks.0.attn.o_proj"].B.shape == (4, 16)


def test_gemma_unified_skips_missing_v_projection_layers():
    model = GemmaUnifiedModel(layers=2, missing_v_layers=(1,))
    manager = LoRAManager(model)
    adapters = manager.initialize_adapters(
        targets=["attn.q_proj", "attn.k_proj", "attn.v_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
    )

    assert "blocks.0.attn.v_proj" in adapters
    assert "blocks.1.attn.v_proj" not in adapters
    assert "blocks.1.attn.q_proj" in adapters
    assert "blocks.1.attn.k_proj" in adapters


def test_gemma_unified_uses_block_local_projection_geometry():
    model = GemmaUnifiedModel(layers=2, n_kv_heads=2, missing_v_layers=(1,), global_kv_heads=1)
    manager = LoRAManager(model)
    adapters = manager.initialize_adapters(
        targets=["attn.k_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
    )

    assert adapters["blocks.0.attn.k_proj"].A.shape == (8, 2)
    assert adapters["blocks.1.attn.k_proj"].A.shape == (4, 2)


def test_dynamic_rank_initialises_gated_active_rank_and_exports_active_columns(tmp_path):
    model = SeparateModel(layers=1, hidden=8)
    manager = LoRAManager(model)
    adapters = manager.initialize_adapters(
        targets=["attn.q_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
        initial_active_rank=2,
    )
    adapter = adapters["blocks.0.attn.q_proj"]

    assert adapter.rank == 4
    assert adapter.active_rank == 2
    tensors, metadata = manager.export_active_pack("dynamic-demo", tmp_path)

    assert metadata.rank_map["blocks.0.attn.q_proj"] == 2
    assert metadata.alpha_map["blocks.0.attn.q_proj"] == 4.0
    assert tensors["blocks.0.attn.q_proj.lora.A"].shape == (8, 2)
    assert tensors["blocks.0.attn.q_proj.lora.B"].shape == (2, 8)


def test_dynamic_rank_adjusts_only_high_signal_adapters():
    model = SeparateModel(layers=2, hidden=8)
    manager = LoRAManager(model)
    adapters = manager.initialize_adapters(
        targets=["attn.q_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
        initial_active_rank=2,
    )
    high = adapters["blocks.0.attn.q_proj"]
    low = adapters["blocks.1.attn.q_proj"]
    high.A = mx.ones_like(high.A)
    low.A = mx.zeros_like(low.A)

    events = manager.adjust_dynamic_ranks(
        allowed_ranks=(2, 4),
        min_rank=2,
        grow_threshold=0.25,
        prune_threshold=0.03,
    )

    assert high.active_rank == 4
    assert low.active_rank == 2
    assert len(events) == 1
    assert events[0]["adapter"] == "blocks.0.attn.q_proj"
    assert events[0]["action"] == "grow"
    assert events[0]["from_rank"] == 2
    assert events[0]["to_rank"] == 4
    assert events[0]["max_rank"] == 4
    assert events[0]["signal"] == pytest.approx(events[0]["global_signal"])


def test_compute_auto_ranks_with_theorem():
    model = SeparateModel(hidden=4)
    block = model.model.layers[0]
    block.self_attn.q_proj.weight = _eye_rank(4, 4)
    block.self_attn.k_proj.weight = _eye_rank(4, 2)
    block.self_attn.v_proj.weight = _eye_rank(4, 2)

    manager = LoRAManager(model)
    targets = ["attn.q_proj", "attn.k_proj", "attn.v_proj"]
    ranks, alphas, residuals = manager.compute_auto_ranks(
        targets,
        strategy="theorem",
        target_compression=0.9,
    )
    assert ranks["attn.q_proj"] == 4
    assert ranks["attn.k_proj"] == 2
    assert ranks["attn.v_proj"] == 2
    assert alphas["attn.q_proj"] == pytest.approx(8.0)
    assert residuals["attn.q_proj"] <= 1e-6
