
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")
import mlx.nn as nn

from mlx_plastic_rank.lowrank import RankLayer


@pytest.mark.parametrize("prune_by_tolerance", [False, True])
def test_non_lifo_wake_and_prune_preserves_every_component(prune_by_tolerance):
    mx.random.seed(3)
    layer = RankLayer(mx.eye(4))
    layer.add_rank(4)
    layer.S = mx.array([1., 2., 3., 4.])

    def park():
        if prune_by_tolerance:
            layer.prune_rank(tol=3.5)
        else:
            layer.prune_to_rank(1)

    park()
    layer.wake_rank(min(layer.sleep_dict))
    park()
    assert layer.rank + len(layer.sleep_dict) == 4
    for key in list(layer.sleep_dict):
        layer.wake_rank(key)
    assert sorted(layer.S.tolist()) == [1., 2., 3., 4.]


def test_pruning_preserves_large_negative_coefficients():
    layer = RankLayer(mx.eye(3))
    layer.add_rank(3)
    layer.S = mx.array([-5., 1., -1e-6])
    layer.prune_rank(tol=1e-4)

    assert layer.S.tolist() == [-5., 1.]
    assert len(layer.sleep_dict) == 1


def test_normal_optimizer_updates_only_live_factors_and_bias():
    import mlx.optimizers as optim

    mx.random.seed(17)
    layer = RankLayer(mx.eye(3), mx.zeros(3))
    layer.add_rank(2)
    layer.prune_to_rank(1)
    backbone = mx.array(layer.W0)
    old_s = mx.array(layer.S)
    sleeper = next(iter(layer.sleep_dict.values()))
    optimizer = optim.SGD(learning_rate=0.1)
    loss, grads = nn.value_and_grad(layer, lambda model: mx.sum(model(mx.ones((2, 3))) ** 2))(layer)
    optimizer.update(layer, grads)
    mx.eval(loss, layer.parameters())

    assert mx.array_equal(layer.W0, backbone).item()
    assert not mx.array_equal(layer.S, old_s).item()
    assert mx.array_equal(next(iter(layer.sleep_dict.values()))[0], sleeper[0]).item()
