import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_plastic_rank.packs.lora import LoRAFusedLinear, SliceLoRA, slice_bounds


def test_slice_bounds_known_targets():
    start, end = slice_bounds("attn.q_proj")
    assert start == 0
    assert end > start
    with pytest.raises(KeyError):
        slice_bounds("attn.x_proj")


def test_fused_linear_adds_lora_delta():
    base = nn.Linear(4, 12, bias=False)
    base.weight = mx.ones_like(base.weight)
    wrapper = LoRAFusedLinear(base, input_dim=4, output_dim=12)
    A = mx.ones((4, 2), dtype=mx.float16)
    B = mx.ones((2, 4), dtype=mx.float16)
    adapter = SliceLoRA(
        name="blocks.0.attn.q_proj",
        start=0,
        end=4,
        rank=2,
        alpha=2.0,
        A=A,
        B=B,
        input_dim=4,
        output_dim=4,
    )
    wrapper.add_adapter(adapter)

    x = mx.ones((1, 4))
    out = wrapper(x)
    base_out = base(x)
    # delta should only affect first slice
    delta = out - base_out
    assert float(delta[0, 4]) == pytest.approx(0.0)
    expected_delta = (adapter.alpha / adapter.rank) * 4 * 2
    assert float(delta[0, 0]) == pytest.approx(expected_delta)

    wrapper.clear_adapters()
    out_no_adapter = wrapper(x)
    assert mx.allclose(out_no_adapter, base_out)


def test_zero_lora_no_effect():
    base = nn.Linear(4, 12, bias=False)
    wrapper = LoRAFusedLinear(base, input_dim=4, output_dim=12)
    A = mx.zeros((4, 2), dtype=mx.float16)
    B = mx.zeros((2, 4), dtype=mx.float16)
    adapter = SliceLoRA(
        name="blocks.0.attn.q_proj",
        start=0,
        end=4,
        rank=2,
        alpha=4.0,
        A=A,
        B=B,
        input_dim=4,
        output_dim=4,
    )
    wrapper.add_adapter(adapter)
    x = mx.random.normal((3, 4))
    assert mx.allclose(wrapper(x), base(x))


def test_active_rank_gate_limits_lora_delta():
    base = nn.Linear(4, 12, bias=False)
    base.weight = mx.ones_like(base.weight)
    wrapper = LoRAFusedLinear(base, input_dim=4, output_dim=12)
    A = mx.ones((4, 2), dtype=mx.float16)
    B = mx.ones((2, 4), dtype=mx.float16)
    adapter = SliceLoRA(
        name="blocks.0.attn.q_proj",
        start=0,
        end=4,
        rank=2,
        alpha=2.0,
        A=A,
        B=B,
        input_dim=4,
        output_dim=4,
    )
    adapter.set_active_rank(1)
    wrapper.add_adapter(adapter)

    x = mx.ones((1, 4))
    delta = wrapper(x) - base(x)

    assert adapter.active_rank == 1
    assert float(delta[0, 0]) == pytest.approx(4.0)
    export_A, export_B, export_alpha, export_rank = adapter.export_arrays()
    assert export_A.shape == (4, 1)
    assert export_B.shape == (1, 4)
    assert export_alpha == pytest.approx(1.0)
    assert export_rank == 1


def test_arbitrary_component_gates_apply_and_export_exact_selected_pairs():
    adapter = SliceLoRA(
        name="blocks.0.attn.q_proj",
        start=0,
        end=3,
        rank=3,
        alpha=6.0,
        A=mx.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=mx.float16,
        ),
        B=mx.eye(3, dtype=mx.float16),
        input_dim=3,
        output_dim=3,
    )
    x = mx.array([[2.0, 3.0, 5.0]], dtype=mx.float16)
    adapter.set_active_components((1, 2))
    export_A, export_B, _, export_rank = adapter.export_arrays()

    assert adapter.active_component_indices == (1, 2)
    assert adapter.component_utilities() == pytest.approx((1.0, 4.0, 2.0))
    assert mx.allclose(
        adapter.delta(x),
        mx.array([[0.0, 24.0, 20.0]], dtype=mx.float16),
    )
    assert export_rank == 2
    assert mx.allclose(export_A, adapter.A[:, 1:])
    assert mx.allclose(export_B, adapter.B[1:, :])


def test_alpha_zero_no_effect():
    base = nn.Linear(4, 12, bias=False)
    wrapper = LoRAFusedLinear(base, input_dim=4, output_dim=12)
    A = mx.random.normal((4, 2), dtype=mx.float16)
    B = mx.random.normal((2, 4), dtype=mx.float16)
    adapter = SliceLoRA(
        name="blocks.0.attn.q_proj",
        start=0,
        end=4,
        rank=2,
        alpha=0.0,
        A=A,
        B=B,
        input_dim=4,
        output_dim=4,
    )
    wrapper.add_adapter(adapter)
    x = mx.random.normal((2, 4))
    assert mx.allclose(wrapper(x), base(x))


def test_dropout_bounds_validation():
    base = nn.Linear(4, 12, bias=False)
    wrapper = LoRAFusedLinear(base, input_dim=4, output_dim=12)
    with pytest.raises(ValueError):
        wrapper.set_dropout(-0.1)
    with pytest.raises(ValueError):
        wrapper.set_dropout(1.0)


def test_gated_projection_preserves_outputs_and_inactive_gradients():
    from mlx_plastic_rank.packs.lora import SliceLoRA

    mx.random.seed(81)
    adapter = SliceLoRA("test", 0, 8, 4, 8, mx.random.normal((8, 4)), mx.random.normal((4, 8)), 8, 8)
    adapter.gates = mx.array([0.0, 1.0, 0.0, 0.25])
    inputs = mx.random.normal((2, 3, 8))
    dense = ((inputs @ adapter.B.T) * adapter.gates) @ adapter.A.T * adapter.scale
    assert mx.allclose(adapter.delta(inputs), dense, atol=1e-5).item()

    def loss(arrays):
        adapter.A, adapter.B = arrays
        return mx.sum(adapter.delta(inputs) ** 2)

    _, gradients = mx.value_and_grad(loss)([adapter.A, adapter.B])
    assert mx.array_equal(gradients[0][:, [0, 2]], mx.zeros((8, 2))).item()
    assert mx.array_equal(gradients[1][[0, 2], :], mx.zeros((2, 8))).item()
