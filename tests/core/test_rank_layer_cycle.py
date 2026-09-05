import json

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")
import mlx.nn as nn

from mlx_plastic_rank.lowrank import RankLayer
from mlx_plastic_rank.plasticity_manager import PlasticityManager


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


def test_rank_layer_prune_wake_cycle():
    weight = mx.eye(4)
    bias = mx.zeros(4)
    layer = RankLayer(weight, bias)
    model = nn.Sequential(layer)
    mgr = PlasticityManager(model, delta=1e-3, gamma=1e-4, log_path=None)
    layer.add_rank(2)

    # Force one component below tolerance so it is parked in sleep_dict.
    layer.S = mx.array([5e-5, 2e-3])

    layer.prune_rank(tol=1e-4)

    assert layer.rank == 1
    assert len(layer.sleep_dict) == 1

    entry = next(iter(layer.sleep_dict.values()))
    q_u, mn_u, sc_u, s, q_v, mn_v, sc_v = entry
    expected_bytes = (
        int(q_u.size * q_u.dtype.size)
        + int(q_v.size * q_v.dtype.size)
        + 8 * 5
    )
    assert mgr.compress_dict_size() == expected_bytes

    idx = next(iter(layer.sleep_dict.keys()))
    layer.wake_rank(idx)

    assert layer.rank == 2
    assert not layer.sleep_dict
    assert mgr.compress_dict_size() == 0


def test_plasticity_manager_reactivation_wakes_sleepers(monkeypatch):
    weight = mx.eye(4)
    bias = mx.zeros(4)
    model = nn.Sequential(RankLayer(weight, bias))
    mgr = PlasticityManager(model, delta=1e-3, gamma=1e-4, log_path=None)

    layer = mgr.layers[0]
    layer.add_rank(3)
    layer.S = mx.array([1e-3, 1e-3, 1e-3])

    def stub_choose_rank(W, target, strategy):
        return 1, -1.0

    import mlx_plastic_rank.plasticity_manager as pm

    monkeypatch.setattr(pm, "choose_rank", stub_choose_rank)

    mgr.step({"val_loss": 1.0})
    mgr.step({"val_loss": 1.0})

    assert layer.rank == 1
    assert len(layer.sleep_dict) == 2

    mgr.step({"val_loss": 1.5})

    assert layer.rank == 2
    assert len(layer.sleep_dict) == 1


def test_plasticity_manager_does_not_wake_on_same_step_as_shrink(monkeypatch, tmp_path):
    weight = mx.eye(4)
    bias = mx.zeros(4)
    model = nn.Sequential(RankLayer(weight, bias))
    log_path = tmp_path / "plasticity.jsonl"
    mgr = PlasticityManager(model, log_path=str(log_path))

    layer = mgr.layers[0]
    layer.add_rank(3)
    layer.S = mx.array([1e-3, 1e-3, 1e-3])

    choose_calls = 0

    def stub_choose_rank(W, target, strategy):
        nonlocal choose_calls
        choose_calls += 1
        return (1 if choose_calls == 1 else 0), -1.0

    import mlx_plastic_rank.plasticity_manager as pm

    monkeypatch.setattr(pm, "choose_rank", stub_choose_rank)

    mgr.step({"val_loss": 1.0})
    mgr.step({"val_loss": 1.005})

    assert layer.rank == 1
    assert len(layer.sleep_dict) == 2
    assert choose_calls == 1
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["action"] for event in events] == ["shrink"]
    assert events[0]["val_loss_before"] == 1.0
    assert events[0]["val_loss_after"] == 1.005

    mgr.step({"val_loss": 1.01})

    assert layer.rank == 2
    assert len(layer.sleep_dict) == 1
    assert choose_calls == 1
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["action"] for event in events] == ["shrink", "wake"]
    assert events[1]["r0"] == 1
    assert events[1]["r_star"] == 2
    assert events[1]["val_loss_before"] == 1.005
    assert events[1]["val_loss_after"] == 1.01
    assert mgr._last_action[0] == "wake"
