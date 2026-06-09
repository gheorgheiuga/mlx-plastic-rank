import types

import pytest

mx = pytest.importorskip("mlx.core", reason="MLX not installed; skipping")
import mlx.nn as nn

from mlx_plastic_rank.packs.inspection import ALLOWED_RANKS, allowed_ranks_for
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
    def __init__(self, hidden: int = 32):
        self.model = {"h": [FusedBlock(hidden)]}
        self.config = types.SimpleNamespace(n_embd=hidden)
        self.model_type = "gpt2"


def test_allowed_ranks_for_profiles():
    assert allowed_ranks_for("lite") == ALLOWED_RANKS
    heavy = allowed_ranks_for("heavy")
    assert 16 in heavy and 32 in heavy and 64 in heavy
    with pytest.raises(ValueError):
        allowed_ranks_for("unknown")


def test_heavy_profile_allows_higher_rank():
    model = FusedModel(hidden=32)
    manager = LoRAManager(model)

    with pytest.raises(PackApplicationError):
        manager.initialize_adapters(["attn.q_proj"], rank=16, alpha=32.0, seed=0)

    adapters = manager.initialize_adapters(
        ["attn.q_proj"],
        rank=16,
        alpha=32.0,
        seed=0,
        allowed_ranks=allowed_ranks_for("heavy"),
    )
    adapter = adapters["blocks.0.attn.q_proj"]
    assert adapter.rank == 16
    assert adapter.alpha == pytest.approx(32.0)


def test_export_heavy_profile_in_metadata(tmp_path):
    model = FusedModel(hidden=32)
    manager = LoRAManager(model)
    manager.initialize_adapters(
        ["attn.q_proj"],
        rank=16,
        alpha=32.0,
        seed=0,
        allowed_ranks=allowed_ranks_for("heavy"),
    )

    _, metadata = manager.export_active_pack(
        "heavy-demo",
        tmp_path,
        profile="heavy",
    )
    assert metadata.profile == "heavy"
