import json
import os
import tempfile

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")
import mlx.nn as nn


def test_manager_writes_jsonl_and_sleep_changes():
    from mlx_plastic_rank.lowrank import PlasticBlock
    from mlx_plastic_rank.plasticity_manager import PlasticityManager

    d_model = 64
    model = nn.Sequential(PlasticBlock(d_model))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "plasticity.jsonl")
        mgr = PlasticityManager(
            model, delta=1e-6, strategy="stable", target_compression=0.1, log_path=path
        )

        # Ensure some rank to start with
        for m in model.modules():
            if hasattr(m, "add_rank"):
                m.add_rank(8)

        # Simulate plateau by calling with nearly identical losses
        mgr.step({"val_loss": 1.234})
        mgr.step({"val_loss": 1.23400005})

        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        evt = json.loads(line)
        assert all(k in evt for k in ["layer", "r0", "r_star", "action", "strategy"])
