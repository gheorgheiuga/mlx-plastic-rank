import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")

from safetensors.numpy import load_file, save_file

import mlx_plastic_rank.lowrank as lowrank


@pytest.mark.parametrize("shape", [(320, 96), (96, 320)])
@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
def test_randomized_svd_keeps_decomposition_workspace_at_sketch_size(monkeypatch, shape, device):
    rank, oversamples = 8, 4
    sketch_size = rank + oversamples
    real_svd = mx.linalg.svd

    def bounded_svd(matrix, *args, **kwargs):
        # Check the allocation request before the backend can create a full
        # square singular-vector matrix along an original input dimension.
        assert max(matrix.shape) <= sketch_size
        return real_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(mx.linalg, "svd", bounded_svd)
    mx.random.seed(42)
    matrix = mx.random.normal(shape)
    u, s, vh = lowrank.randomized_svd(matrix, rank, p=oversamples, device_stream=mx.stream(device))
    mx.eval(u, s, vh)

    assert u.shape == (shape[0], rank)
    assert s.shape == (rank,)
    assert vh.shape == (rank, shape[1])
    np.testing.assert_allclose(np.array(u.T @ u), np.eye(rank), atol=2e-5)
    np.testing.assert_allclose(np.array(vh @ vh.T), np.eye(rank), atol=2e-5)


@pytest.mark.parametrize("shape", [(160, 64), (64, 160)])
@pytest.mark.parametrize("rank", [8, 63, 64])
def test_randomized_svd_accuracy_against_optimal_truncated_svd(shape, rank):
    rng = np.random.default_rng(13)
    width = min(shape)
    left, _ = np.linalg.qr(rng.normal(size=(shape[0], width)))
    right, _ = np.linalg.qr(rng.normal(size=(shape[1], width)))
    values = np.geomspace(100.0, 0.001, width)
    matrix = ((left * values) @ right.T).astype(np.float32)
    mx.random.seed(42)

    u, s, vh = lowrank.randomized_svd(mx.array(matrix), rank, p=8, q=2, chunk_k=5)
    approximation = np.array((u * s) @ vh)
    error = np.linalg.norm(matrix - approximation)
    optimal_error = np.linalg.norm(values[rank:])

    assert error <= 1.03 * optimal_error + 2e-5 * np.linalg.norm(matrix)
    np.testing.assert_allclose(np.array(u.T @ u), np.eye(rank), atol=3e-5)
    np.testing.assert_allclose(np.array(vh @ vh.T), np.eye(rank), atol=3e-5)


def test_power_iterations_remain_finite_on_large_rank_deficient_input():
    rng = np.random.default_rng(71)
    matrix = (rng.normal(size=(128, 6)) @ rng.normal(size=(6, 96)) * 1e9).astype(np.float32)
    mx.random.seed(9)

    u, s, vh = lowrank.randomized_svd(mx.array(matrix), 8, p=4, q=4)
    approximation = np.array((u * s) @ vh)

    assert np.isfinite(approximation).all()
    assert np.linalg.norm(matrix - approximation) / np.linalg.norm(matrix) < 2e-5


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_zero_input_has_finite_zero_reconstruction(dtype):
    matrix = mx.zeros((48, 32), dtype=dtype)
    u, s, vh = lowrank.randomized_svd(matrix, 4, p=4)

    assert np.array_equal(np.array((u * s) @ vh), np.zeros(matrix.shape))


@pytest.mark.parametrize("factorize", [lowrank.randomized_svd, lowrank.factorized_lowrank])
def test_factorization_rejects_complex_input_instead_of_discarding_imaginary_values(factorize):
    matrix = mx.ones((32, 32), dtype=mx.complex64) * 1j

    with pytest.raises(ValueError, match="real"):
        factorize(matrix, 4)


def test_factorized_lowrank_small_matrix_matches_optimal_error():
    mx.random.seed(31)
    A = mx.random.normal((64, 32))
    U, S, Vh = lowrank.factorized_lowrank(A, 4)
    reference_s = np.linalg.svd(np.array(A), compute_uv=False)

    error = float(mx.linalg.norm(A - (U * S) @ Vh))
    assert error == pytest.approx(np.linalg.norm(reference_s[4:]), rel=2e-6)
    assert U.shape == (64, 4)
    assert S.shape == (4,)
    assert Vh.shape == (4, 32)
    # Cover the public reconstruction wrapper against the same optimal result.
    reconstruction = lowrank.svd_lowrank(A, 4)
    assert reconstruction.shape == A.shape
    assert mx.allclose(reconstruction, (U * S) @ Vh, atol=1e-6).item()


def test_factorized_lowrank_large_matrix_bounds_svd_workspace(monkeypatch):
    original = mx.linalg.svd

    def bounded_svd(A, *args, **kwargs):
        assert max(A.shape) <= 16
        return original(A, *args, **kwargs)

    monkeypatch.setattr(mx.linalg, "svd", bounded_svd)

    A = mx.random.normal((1200, 64))
    U, S, Vh = lowrank.factorized_lowrank(A, 8)

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
