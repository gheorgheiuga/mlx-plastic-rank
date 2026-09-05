import json
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")
import mlx.nn as nn

from mlx_plastic_rank.packs.io import load_pack_metadata, save_pack, save_pack_metadata
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


def _write_pack(tmp_path, *, base_model=None, base_checkpoint=None):
    source_model = FusedModel(hidden=8)
    source_manager = LoRAManager(
        source_model,
        base_checkpoint=base_checkpoint,
        base_model=base_model,
    )
    source_manager.initialize_adapters(["attn.q_proj"], rank=4, alpha=8.0, seed=0)
    tensors, metadata = source_manager.export_active_pack("demo", tmp_path)
    pack_dir = tmp_path / "demo"
    save_pack(tensors, pack_dir / "pack.safetensors")
    save_pack_metadata(metadata, pack_dir / "meta.json")
    return pack_dir, tensors


def test_apply_pack_rejects_mismatched_remote_base_model(tmp_path):
    pack_dir, _ = _write_pack(tmp_path, base_model="org/model-a")
    manager = LoRAManager(FusedModel(hidden=8), base_model="org/model-b")

    with pytest.raises(PackApplicationError, match="Base model mismatch"):
        manager.apply_pack(pack_dir)


def test_apply_pack_accepts_different_base_labels_when_hashes_match(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"same-checkpoint")
    pack_dir, _ = _write_pack(
        tmp_path,
        base_model="local-alias-a",
        base_checkpoint=checkpoint,
    )
    manager = LoRAManager(
        FusedModel(hidden=8),
        base_model="local-alias-b",
        base_checkpoint=checkpoint,
    )

    metadata = manager.apply_pack(pack_dir)

    assert metadata.pack_name == "demo"


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


def test_apply_pack_rejects_non_scalar_alpha_tensor(tmp_path):
    pack_dir, tensors = _write_pack(tmp_path)
    tampered = dict(tensors)
    key = "blocks.0.attn.q_proj.lora.alpha"
    tampered[key] = np.array([8.0], dtype=np.float32)
    save_pack(tampered, pack_dir / "pack.safetensors")

    manager = LoRAManager(FusedModel(hidden=8))
    with pytest.raises(PackApplicationError, match="scalar"):
        manager.apply_pack(pack_dir)


def test_apply_pack_rejects_non_fp32_alpha_tensor(tmp_path):
    pack_dir, tensors = _write_pack(tmp_path)
    tampered = dict(tensors)
    key = "blocks.0.attn.q_proj.lora.alpha"
    tampered[key] = np.array(8.0, dtype=np.float16)
    save_pack(tampered, pack_dir / "pack.safetensors")

    manager = LoRAManager(FusedModel(hidden=8))
    with pytest.raises(PackApplicationError, match="float32"):
        manager.apply_pack(pack_dir)


def test_apply_pack_rejects_alpha_tensor_metadata_mismatch(tmp_path):
    pack_dir, tensors = _write_pack(tmp_path)
    tampered = dict(tensors)
    key = "blocks.0.attn.q_proj.lora.alpha"
    tampered[key] = np.array(123.0, dtype=np.float32)
    save_pack(tampered, pack_dir / "pack.safetensors")

    manager = LoRAManager(FusedModel(hidden=8))
    with pytest.raises(PackApplicationError, match="alpha mismatch"):
        manager.apply_pack(pack_dir)


def test_failed_pack_validation_preserves_active_adapters(tmp_path):
    valid_dir, _ = _write_pack(tmp_path / "valid")
    invalid_dir, invalid_tensors = _write_pack(tmp_path / "invalid")
    alpha_key = "blocks.0.attn.q_proj.lora.alpha"
    invalid_tensors[alpha_key] = np.array(123.0, dtype=np.float32)
    save_pack(invalid_tensors, invalid_dir / "pack.safetensors")

    manager = LoRAManager(FusedModel(hidden=8))
    manager.apply_pack(valid_dir)
    active_before = dict(manager.iter_adapters())

    with pytest.raises(PackApplicationError, match="alpha mismatch"):
        manager.apply_pack(invalid_dir)

    active_after = dict(manager.iter_adapters())
    assert active_after == active_before


def test_failed_wrapper_prevalidation_preserves_active_adapters(tmp_path, monkeypatch):
    valid_dir, _ = _write_pack(tmp_path / "valid")
    replacement_dir, _ = _write_pack(tmp_path / "replacement")
    manager = LoRAManager(FusedModel(hidden=8))
    manager.apply_pack(valid_dir)
    active_before = dict(manager.iter_adapters())
    wrapper = next(iter(manager._wrappers.values()))

    def reject_attachment(adapter):
        raise ValueError(f"rejected {adapter.name}")

    monkeypatch.setattr(wrapper, "validate_adapter", reject_attachment)

    with pytest.raises(PackApplicationError, match="Cannot attach"):
        manager.apply_pack(replacement_dir)

    assert dict(manager.iter_adapters()) == active_before


def test_apply_pack_accepts_zero_alpha_when_metadata_and_tensor_match(tmp_path):
    pack_dir, tensors = _write_pack(tmp_path)
    adapter_key = "blocks.0.attn.q_proj"
    alpha_key = f"{adapter_key}.lora.alpha"
    tensors[alpha_key] = np.array(0.0, dtype=np.float32)
    save_pack(tensors, pack_dir / "pack.safetensors")
    metadata = load_pack_metadata(pack_dir / "meta.json")
    metadata.alpha_map[adapter_key] = 0.0
    save_pack_metadata(metadata, pack_dir / "meta.json")

    manager = LoRAManager(FusedModel(hidden=8))
    manager.apply_pack(pack_dir)

    assert dict(manager.iter_adapters())[adapter_key].alpha == 0.0


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


@pytest.mark.parametrize("changed_file", ["model-00002.safetensors", "config.json"])
def test_complete_checkpoint_identity_rejects_changed_content(tmp_path, changed_file):
    from mlx_plastic_rank.packs.provenance import content_sha256

    checkpoint = tmp_path / "base"
    checkpoint.mkdir()
    for name in ("model-00001.safetensors", "model-00002.safetensors", "config.json"):
        (checkpoint / name).write_bytes(b"original")
    pack_dir, _ = _write_pack(tmp_path, base_model="alias-a", base_checkpoint=checkpoint)
    metadata = load_pack_metadata(pack_dir / "meta.json")
    assert metadata.base_hash_version == 1
    assert metadata.base_hash == content_sha256(checkpoint)
    (checkpoint / changed_file).write_bytes(b"changed")
    manager = LoRAManager(FusedModel(), base_model="alias-b", base_checkpoint=checkpoint)
    with pytest.raises(PackApplicationError, match="Base hash mismatch"):
        manager.apply_pack(pack_dir)


def test_legacy_pack_requires_whole_checkpoint_provenance(tmp_path):
    from mlx_plastic_rank.packs.provenance import content_sha256

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    pack_dir, _ = _write_pack(tmp_path, base_checkpoint=checkpoint)
    path = pack_dir / "meta.json"
    metadata = json.loads(path.read_text())
    metadata.pop("base_hash_version")
    path.write_text(json.dumps(metadata))
    manager = LoRAManager(FusedModel(), base_checkpoint=checkpoint)
    with pytest.raises(PackApplicationError, match="Legacy pack"):
        manager.apply_pack(pack_dir)
    metadata["training_config"] = {"provenance": {"model_sha256": content_sha256(checkpoint)}}
    path.write_text(json.dumps(metadata))
    assert manager.apply_pack(pack_dir).base_hash_version == 0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), 70000.0])
def test_invalid_factor_cannot_replace_active_pack_or_be_exported(tmp_path, bad_value):
    from safetensors.numpy import save_file

    valid_dir, _ = _write_pack(tmp_path / "valid")
    bad_dir, tensors = _write_pack(tmp_path / "bad")
    manager = LoRAManager(FusedModel())
    manager.apply_pack(valid_dir)
    active = dict(manager.iter_adapters())
    key = "blocks.0.attn.q_proj.lora.B"
    tensors[key] = tensors[key].astype(np.float32)
    tensors[key][0, 0] = bad_value
    save_file(tensors, str(bad_dir / "pack.safetensors"))  # Deliberately malformed external artifact.
    with pytest.raises(PackApplicationError, match="finite"):
        manager.apply_pack(bad_dir)
    assert dict(manager.iter_adapters()) == active
    next(iter(active.values())).B = mx.array(tensors[key])
    with pytest.raises(PackApplicationError, match="finite"):
        manager.export_active_pack("invalid-export", tmp_path)
    assert not (tmp_path / "invalid-export").exists()
