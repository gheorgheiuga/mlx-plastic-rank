import os
import tempfile

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")

import importlib


def test_export_pack_save_load_roundtrip_accuracy_and_size():
    lowrank = importlib.import_module("mlx_plastic_rank.lowrank")
    export = importlib.import_module("mlx_plastic_rank.export_safetensors")

    m, n, r = 24, 18, 6
    A = mx.random.normal((m, n))
    U, S, Vh = lowrank.factorized_lowrank(A, r)
    A_r = (U * S[None, :]) @ Vh

    packed = export.pack_lowrank(U, S, Vh, bits=8)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "lr.safetensors")
        export.save_lowrank(path, packed)
        U2, S2, Vh2 = export.load_lowrank(path)
        A_hat = (U2 * S2[None, :]) @ Vh2

        rel = mx.linalg.norm(A_r - A_hat) / mx.linalg.norm(A_r)
        assert float(rel) <= 0.025

        # Rough size expectations
        q_bytes = (m * r) + r + (r * n)  # uint8 counts
        size = os.path.getsize(path)
        # Must be at least q storage and not explode beyond a modest overhead
        assert size >= q_bytes
        overhead = 16 * (m + r + 1 + r) + 8192  # mins/scales + header slack
        assert size <= q_bytes + overhead



@pytest.mark.parametrize("field,value", [("bits", 16), ("version", "future"), ("rank", 99), ("shape", [999, 999])])
def test_load_rejects_inconsistent_quantization_metadata(tmp_path, field, value):
    import json

    import numpy as np
    from safetensors.numpy import save_file

    from mlx_plastic_rank.export_safetensors import load_lowrank, pack_lowrank

    packed = pack_lowrank(mx.eye(2), mx.ones(2), mx.eye(2))
    metadata = json.loads(packed["__meta__.json"].tobytes())
    metadata[field] = value
    packed["__meta__.json"] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    target = tmp_path / "invalid.safetensors"
    save_file(packed, str(target))
    with pytest.raises(ValueError):
        load_lowrank(str(target))
