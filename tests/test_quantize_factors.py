import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")

import importlib


def test_quantize_factors_roundtrip_small_error():
    lowrank = importlib.import_module("mlx_plastic_rank.lowrank")
    A = mx.random.normal((20, 15))
    r = 6
    U, S, Vh = lowrank.factorized_lowrank(A, r)
    exact = (U * S[None, :]) @ Vh
    packed = lowrank.quantize_factors(U, S, Vh, bits=8)
    U2, S2, Vh2 = lowrank.dequantize_factors(packed)
    approx = (U2 * S2[None, :]) @ Vh2
    rel = mx.linalg.norm(exact - approx) / mx.linalg.norm(exact)
    assert float(rel) <= 0.02


def test_quantize_factors_constant_rows():
    lowrank = importlib.import_module("mlx_plastic_rank.lowrank")
    # Construct factors with a constant row in U and Vh
    U = mx.ones((10, 3)) * 0.5
    S = mx.array([1.0, 0.5, 0.25])
    Vh = mx.ones((3, 8)) * -0.3
    packed = lowrank.quantize_factors(U, S, Vh, bits=8)
    U2, S2, Vh2 = lowrank.dequantize_factors(packed)
    # The dequantization should closely match originals
    err = (
        mx.linalg.norm(U - U2) + mx.linalg.norm(S - S2) + mx.linalg.norm(Vh - Vh2)
    )
    assert float(err) < 1e-2

