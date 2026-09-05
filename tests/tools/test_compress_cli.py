import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_randomized_compression_and_gpu_fallback_preserve_bounded_svd(monkeypatch, device):
    from scripts import compress_llm_mlx as compress

    def reject_full_numpy_svd(*args, **kwargs):
        pytest.fail("A requested randomized factorization must not run a full NumPy SVD")

    original_svd = mx.linalg.svd

    def bounded_svd(matrix, *args, **kwargs):
        assert max(matrix.shape) <= 12
        return original_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", reject_full_numpy_svd)
    monkeypatch.setattr(mx.linalg, "svd", bounded_svd)
    failures = []
    if device == "gpu":
        real_qr = mx.linalg.qr

        def fail_first_backend_call(matrix, *args, **kwargs):
            if not failures:
                failures.append(True)
                raise RuntimeError("Simulated backend allocation failure")
            return real_qr(matrix, *args, **kwargs)

        monkeypatch.setattr(mx.linalg, "qr", fail_first_backend_call)
    matrix = np.random.default_rng(4).normal(size=(96, 64)).astype(np.float32)

    result = compress.mlx_svd_truncate(
        matrix, 8, svd_kind="randomized", oversamples=4, iters=1, device=device,
    )

    assert result.shape == matrix.shape
    assert np.isfinite(result).all()
    assert np.linalg.norm(matrix - result) < np.linalg.norm(matrix)
    assert bool(failures) == (device == "gpu")


@pytest.mark.parametrize(
    "svd, device",
    [
        ("full", "cpu"),
        ("randomized", "gpu"),
    ],
)
def test_compress_cli_main(tmp_path, monkeypatch, svd, device):
    from safetensors.numpy import load_file, save_file

    from scripts import compress_llm_mlx as compress

    repo = tmp_path / "repo"
    repo.mkdir()

    arr = np.random.randn(12, 12).astype(np.float32)
    save_file({"dense": arr}, str(repo / "weights.safetensors"))
    (repo / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(compress, "download_checkpoint", lambda model_id: str(repo))

    out_dir = tmp_path / "out"
    argv = [
        "compress_llm_mlx.py",
        "--hf",
        "dummy/model",
        "--out",
        str(out_dir),
        "--svd",
        svd,
        "--device",
        device,
        "--min-dim",
        "2",
        "--svd-oversamples",
        "2",
        "--svd-iters",
        "0",
        "--gpu-max-bytes",
        "1",
    ]

    monkeypatch.setenv("PYTHONWARNINGS", "ignore::DeprecationWarning")
    import sys

    monkeypatch.setattr(sys, "argv", argv)

    compress.main()

    output_weights = out_dir / "weights.safetensors"
    assert output_weights.exists()

    tensors = load_file(str(output_weights))
    assert "dense" in tensors
    assert tensors["dense"].shape == arr.shape

    meta_path = out_dir / "mlx_plastic_rank_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["svd"] == svd
    assert meta["device"] == device
