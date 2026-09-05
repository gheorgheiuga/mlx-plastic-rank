import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping MLX-dependent tests")
import importlib


def test_legacy_wrapper_exports_canonical_tools():
    from mlx_plastic_rank.lowrank import PlasticBlock, RankLayer
    from mlx_plastic_rank.rank_select import stable_rank

    mod = importlib.import_module("plastic_rank")
    assert mod.RankLayer is RankLayer
    assert mod.PlasticBlock is PlasticBlock
    assert mod.stable_rank is stable_rank


def test_plastic_block_output_is_independent_of_other_batch_members():
    from mlx_plastic_rank.lowrank import PlasticBlock

    mx.random.seed(7)
    block = PlasticBlock(d_model=8, n_heads=2)
    inputs = mx.random.normal((2, 3, 8))
    batched = block(inputs)
    separate = mx.concatenate([block(inputs[i:i + 1]) for i in range(2)])

    assert mx.allclose(batched, separate, atol=1e-5).item()


def test_demo_conserves_components_and_restores_residual_without_rank_growth():
    from plastic_rank import run_demo

    rows = run_demo(steps=10, seed=42, d_model=8)
    assert [row["active_rank"] for row in rows] == [4, 2] + [4] * 8
    assert [row["dormant_components"] for row in rows] == [0, 2] + [0] * 8
    assert all(row["base_unchanged"] for row in rows)
    assert rows[1]["residual_relative_error"] > 0.05
    assert rows[2]["residual_relative_error"] < 0.02
    assert all(row["residual_relative_error"] == rows[2]["residual_relative_error"] for row in rows[3:])


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"d_model": 3}])
def test_demo_rejects_invalid_lifecycle_configuration(kwargs):
    from plastic_rank import run_demo

    with pytest.raises(ValueError):
        run_demo(**kwargs)
