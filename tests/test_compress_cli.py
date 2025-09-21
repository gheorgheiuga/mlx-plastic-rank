import json

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")


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

    monkeypatch.setattr(compress, "snapshot_download", lambda *args, **kwargs: str(repo))

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
