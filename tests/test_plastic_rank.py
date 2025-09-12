import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping MLX-dependent tests")
import importlib


def test_module_imports_and_has_classes():
    mod = importlib.import_module("plastic_rank")
    assert hasattr(mod, "RankLayer")
    assert hasattr(mod, "PlasticBlock")
    assert hasattr(mod, "PlasticityManager")


def test_stable_rank_smoke():
    mod = importlib.import_module("plastic_rank")
    # simple 2x2 matrix
    W = mx.array([[1.0, 0.0], [0.0, 1.0]])
    r = mod.stable_rank(W)
    assert r > 0
