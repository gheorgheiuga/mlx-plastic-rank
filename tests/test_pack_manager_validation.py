import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")
import mlx.nn as nn

from mlx_plastic_rank.packs.io import save_pack, save_pack_metadata
from mlx_plastic_rank.packs.manager import LoRAManager, PackApplicationError


class FusedAttention:
    def __init__(self, hidden: int):
        self.c_attn = nn.Linear(hidden, hidden * 3, bias=False)
        self.c_proj = nn.Linear(hidden, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        return self.c_proj(q + k + v)


class FusedBlock:
    def __init__(self, hidden: int):
        self.attn = FusedAttention(hidden)


class FusedModel:
    def __init__(self, layers: int = 1, hidden: int = 8):
        self.model = {"h": [FusedBlock(hidden) for _ in range(layers)]}
        self.config = types.SimpleNamespace(n_embd=hidden)
        self.model_type = "gpt2"


def _write_pack(tmp_path):
    source_model = FusedModel(hidden=8)
    source_manager = LoRAManager(source_model)
    source_manager.initialize_adapters(["attn.q_proj"], rank=4, alpha=8.0, seed=0)
    tensors, metadata = source_manager.export_active_pack("demo", tmp_path)
    pack_dir = tmp_path / "demo"
    save_pack(tensors, pack_dir / "pack.safetensors")
    save_pack_metadata(metadata, pack_dir / "meta.json")
    return pack_dir, tensors


def test_apply_pack_rejects_unexpected_tensor(tmp_path):
    pack_dir, tensors = _write_pack(tmp_path)
    tampered = dict(tensors)
    tampered["junk.tensor"] = np.zeros((1,), dtype=np.float32)
    save_pack(tampered, pack_dir / "pack.safetensors")

    manager = LoRAManager(FusedModel(hidden=8))
    with pytest.raises(PackApplicationError, match="unexpected tensors"):
        manager.apply_pack(pack_dir)


def test_apply_pack_rejects_rank_mismatch(tmp_path):
    pack_dir, tensors = _write_pack(tmp_path)
    tampered = dict(tensors)
    key = "blocks.0.attn.q_proj.lora.A"
    tampered[key] = tampered[key][:, :2]
    save_pack(tampered, pack_dir / "pack.safetensors")

    manager = LoRAManager(FusedModel(hidden=8))
    with pytest.raises(PackApplicationError, match="rank mismatch"):
        manager.apply_pack(pack_dir)


def test_manager_set_dropout_rejects_invalid_values():
    manager = LoRAManager(FusedModel(hidden=8))
    with pytest.raises(PackApplicationError):
        manager.set_dropout(-0.01)
    with pytest.raises(PackApplicationError):
        manager.set_dropout(1.0)
