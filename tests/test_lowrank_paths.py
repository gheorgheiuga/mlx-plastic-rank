import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")

from safetensors.numpy import load_file, save_file

import mlx_plastic_rank.lowrank as lowrank


def test_factorized_lowrank_uses_direct_path(monkeypatch):
    called = {"direct": False}
    original = lowrank._svd_topk_direct

    def spy(A, r):
        called["direct"] = True
        return original(A, r)

    monkeypatch.setattr(lowrank, "_svd_topk_direct", spy)

    A = mx.random.normal((64, 64))
    U, S, Vh = lowrank.factorized_lowrank(A, 4)

    assert called["direct"]
    assert U.shape == (64, 4)
    assert S.shape == (4,)
    assert Vh.shape == (4, 64)


def test_factorized_lowrank_uses_randomized_path(monkeypatch):
    called = {"randomized": False}
    original = lowrank.randomized_svd

    def spy(A, r, *args, **kwargs):
        called["randomized"] = True
        return original(A, r, *args, **kwargs)

    monkeypatch.setattr(lowrank, "randomized_svd", spy)

    A = mx.random.normal((1200, 64))
    U, S, Vh = lowrank.factorized_lowrank(A, 8)

    assert called["randomized"]
    assert U.shape == (1200, 8)
    assert S.shape == (8,)
    assert Vh.shape == (8, 64)


def test_compress_safetensors_file_smoke(tmp_path):
    from scripts import compress_llm_mlx as compress

    arr = np.random.randn(32, 32).astype(np.float32)
    in_path = tmp_path / "weights.safetensors"
    out_path = tmp_path / "compressed.safetensors"

    save_file({"dense": arr}, str(in_path))

    changed, total = compress.compress_safetensors_file(
        in_path,
        out_path,
        target_energy=0.8,
        strategy="stable",
        eps=1e-6,
        min_dim=2,
        svd_kind="full",
        svd_oversamples=2,
        svd_iters=0,
        device="cpu",
        gpu_max_bytes=1_000_000,
        max_rank=16,
        gpu_chunk_k=None,
        gpu_max_dim=4096,
    )

    assert changed == 1
    assert total == 1
    assert out_path.exists()

    tensors = load_file(str(out_path))
    assert "dense" in tensors
    assert tensors["dense"].shape == arr.shape
