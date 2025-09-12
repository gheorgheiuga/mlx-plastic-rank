import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")

import importlib


def test_svd_lowrank_roundtrip_error():
    lowrank = importlib.import_module("mlx_plastic_rank.lowrank")
    A = mx.random.normal((32, 16))
    r = 8
    A_hat = lowrank.svd_lowrank(A, r)
    err = mx.linalg.norm(A - A_hat) / mx.linalg.norm(A)
    # For random Gaussian matrices, rank-r SVD should reduce error reasonably
    assert float(err) < 0.9
