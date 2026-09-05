import types

import mlx.nn as nn

from mlx_plastic_rank.packs import lora
from mlx_plastic_rank.packs.inspection import size_limit_for, summarize_pack
from mlx_plastic_rank.packs.io import (
    PackMetadata,
    load_pack_metadata,
    save_pack,
    save_pack_metadata,
)
from mlx_plastic_rank.packs.manager import LoRAManager


class DummyAttention:
    def __init__(self, hidden: int):
        self.c_attn = nn.Linear(hidden, hidden * 3, bias=False)
        self.c_proj = nn.Linear(hidden, hidden, bias=False)


class DummyBlock:
    def __init__(self, hidden: int):
        self.attn = DummyAttention(hidden)


class DummyModel:
    def __init__(self, layers: int = 2, hidden: int = 8):
        self.model = {"h": [DummyBlock(hidden) for _ in range(layers)]}
        self.config = types.SimpleNamespace(n_embd=hidden)


def test_export_only_lora_keys(tmp_path, monkeypatch):
    hidden = 8
    monkeypatch.setattr(
        lora,
        "SLICE_MAP",
        {
            "attn.q_proj": (0, hidden),
            "attn.k_proj": (hidden, hidden * 2),
            "attn.v_proj": (hidden * 2, hidden * 3),
        },
    )
    model = DummyModel(hidden=hidden)
    manager = LoRAManager(model)
    manager.initialize_adapters(["attn.q_proj", "attn.k_proj", "attn.v_proj"], rank=4, alpha=8.0)
    tensors, metadata = manager.export_active_pack("dummy", tmp_path, notes="test")
    pack_dir = tmp_path / "dummy"
    save_pack(tensors, pack_dir / "pack.safetensors")
    save_pack_metadata(metadata, pack_dir / "meta.json")

    _, infos, _, total_bytes, non_lora = summarize_pack(pack_dir)
    assert not non_lora
    assert all(".lora." in info.key for info in infos)
    for info in infos:
        if info.key.endswith(".lora.alpha"):
            assert info.dtype in ("float32", "<f4")
        else:
            assert info.dtype in ("float16", "<f2")
    assert total_bytes < 2 * 1024 * 1024


def test_size_limit_for_known_bases():
    meta_gemma = PackMetadata(
        pack_name="demo",
        base_hash="hash",
        base_model="/tmp/gemma-4-12B-mxfp8",
        rank_map={"blocks.0.attn.q_proj": 4},
        alpha_map={"blocks.0.attn.q_proj": 8.0},
        target_layers=["blocks.0.attn.q_proj"],
        created_at="",
        notes="",
    )
    assert size_limit_for(meta_gemma) == 10 * 1024 * 1024

    meta_llama = PackMetadata(
        pack_name="demo",
        base_hash="hash",
        base_model="/tmp/llama-3-8b-instruct-mlx-4bit",
        rank_map={"blocks.0.attn.q_proj": 8},
        alpha_map={"blocks.0.attn.q_proj": 16.0},
        target_layers=["blocks.0.attn.q_proj"],
        created_at="",
        notes="",
    )
    assert size_limit_for(meta_llama) == 18 * 1024 * 1024


def test_pack_metadata_round_trips_training_provenance(tmp_path):
    path = tmp_path / "meta.json"
    metadata = PackMetadata(
        pack_name="demo",
        base_hash="hash",
        base_model="dummy-base",
        rank_map={"blocks.0.attn.q_proj": 4},
        alpha_map={"blocks.0.attn.q_proj": 8.0},
        target_layers=["blocks.0.attn.q_proj"],
        training_data="data/train.jsonl",
        training_config={"steps": 10, "loss_mode": "answer"},
    )

    save_pack_metadata(metadata, path)
    loaded = load_pack_metadata(path)

    assert loaded.training_data == "data/train.jsonl"
    assert loaded.training_config["steps"] == 10
    assert loaded.training_config["loss_mode"] == "answer"
