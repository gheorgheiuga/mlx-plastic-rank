import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")

import importlib


def test_stable_rank_random():
    rank_select = importlib.import_module("mlx_plastic_rank.rank_select")
    A = mx.random.normal((16, 8))
    r = rank_select.stable_rank(A)
    assert r > 0
    assert r <= min(A.shape) + 1e-3


def test_stable_rank_near_idempotent():
    rank_select = importlib.import_module("mlx_plastic_rank.rank_select")
    I = mx.eye(10)
    A = I + 1e-6 * mx.random.normal(I.shape)
    r = rank_select.stable_rank(A)
    assert 5 <= r <= 10


def test_theorem_guided_identity_exact():
    rank_select = importlib.import_module("mlx_plastic_rank.rank_select")
    I = mx.eye(12)
    r, res = rank_select.theorem_guided_rank(I, target_compression=1.0)
    assert r == 12
    assert res <= 1e-6


def test_theorem_guided_lowrank_random():
    rank_select = importlib.import_module("mlx_plastic_rank.rank_select")
    # Make a low-rank matrix A = U @ V with rank k
    n = 20
    k = 4
    U = mx.random.normal((n, k))
    V = mx.random.normal((k, n))
    A = U @ V
    r, res = rank_select.theorem_guided_rank(A, target_compression=0.8)
    sr = rank_select.stable_rank(A)
    assert r >= int(sr)
    assert res <= 1.0
