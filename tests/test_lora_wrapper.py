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
