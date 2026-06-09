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


def test_applied_pack_adapters_remain_trainable_and_reexportable(tmp_path):
    pack_dir, _ = _write_pack(tmp_path)
    manager = LoRAManager(FusedModel(hidden=8))

    source_metadata = manager.apply_pack(pack_dir)
    params = manager.trainable_parameters()

    assert len(params) == 2
    tensors, metadata = manager.export_active_pack("phase-two", tmp_path, profile=source_metadata.profile)
    assert metadata.rank_map == source_metadata.rank_map
    assert metadata.alpha_map == source_metadata.alpha_map
    assert tensors["blocks.0.attn.q_proj.lora.A"].shape == (8, 4)
    assert tensors["blocks.0.attn.q_proj.lora.B"].shape == (4, 8)


def test_initialize_adapters_accepts_per_adapter_rank_map(tmp_path):
    manager = LoRAManager(FusedModel(layers=2, hidden=8))

    manager.initialize_adapters(
        ["attn.q_proj"],
        rank=4,
        alpha=8.0,
        seed=0,
        rank_map={
            "blocks.0.attn.q_proj": 2,
            "blocks.1.attn.q_proj": 4,
        },
        alpha_map={
            "blocks.0.attn.q_proj": 4.0,
            "blocks.1.attn.q_proj": 8.0,
        },
    )
    tensors, metadata = manager.export_active_pack("hetero", tmp_path)

    assert metadata.rank_map == {
        "blocks.0.attn.q_proj": 2,
        "blocks.1.attn.q_proj": 4,
    }
    assert tensors["blocks.0.attn.q_proj.lora.A"].shape == (8, 2)
    assert tensors["blocks.1.attn.q_proj.lora.A"].shape == (8, 4)


def test_manager_set_dropout_rejects_invalid_values():
    manager = LoRAManager(FusedModel(hidden=8))
    with pytest.raises(PackApplicationError):
        manager.set_dropout(-0.01)
    with pytest.raises(PackApplicationError):
        manager.set_dropout(1.0)
