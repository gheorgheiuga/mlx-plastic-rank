import json
import types
from pathlib import Path

import mlx.nn as nn
import numpy as np
import pytest

from mlx_plastic_rank.packs.io import PackMetadata, save_pack, save_pack_metadata
from mlx_plastic_rank.packs.rank_budget import (
    AdapterShape,
    RankBudgetError,
    adapter_shapes_from_model,
    adapter_shapes_from_pack,
    consumable_rank_map_payload,
    fixed_rank_map,
    normalize_rank_map_to_target,
    random_same_budget_rank_map,
    rank_map_budget_report,
    shuffled_discovered_rank_map,
    validate_rank_map,
)


def _shapes() -> dict[str, AdapterShape]:
    return {
        "blocks.0.attn.q_proj": AdapterShape(
            adapter="blocks.0.attn.q_proj",
            target="attn.q_proj",
            layer=0,
            in_dim=8,
            out_dim=16,
        ),
        "blocks.0.attn.k_proj": AdapterShape(
            adapter="blocks.0.attn.k_proj",
            target="attn.k_proj",
            layer=0,
            in_dim=8,
            out_dim=4,
        ),
        "blocks.1.attn.q_proj": AdapterShape(
            adapter="blocks.1.attn.q_proj",
            target="attn.q_proj",
            layer=1,
            in_dim=8,
            out_dim=16,
        ),
    }


def test_byte_accounting_uses_adapter_shapes():
    shapes = _shapes()
    report = rank_map_budget_report(
        {
            "blocks.0.attn.q_proj": 4,
            "blocks.0.attn.k_proj": 4,
        },
        shapes,
        profile="heavy",
    )

    rows = {row["adapter"]: row for row in report["adapters"]}
    assert rows["blocks.0.attn.q_proj"]["params"] == 4 * (16 + 8)
    assert rows["blocks.0.attn.q_proj"]["tensor_bytes"] == 4 * (16 + 8) * 2 + 4
    assert rows["blocks.0.attn.k_proj"]["params"] == 4 * (4 + 8)
    assert rows["blocks.0.attn.k_proj"]["tensor_bytes"] == 4 * (4 + 8) * 2 + 4
    assert report["summary"]["tensor_bytes"] == 196 + 100


def test_validation_rejects_bad_ranks_and_unsupported_keys():
    shapes = _shapes()

    with pytest.raises(RankBudgetError, match="unsupported rank"):
        validate_rank_map({"blocks.0.attn.q_proj": 3}, shapes, profile="heavy")

    with pytest.raises(RankBudgetError, match="integer rank"):
        validate_rank_map({"blocks.0.attn.q_proj": "4"}, shapes, profile="heavy")

    with pytest.raises(RankBudgetError, match="Unsupported adapter key"):
        validate_rank_map({"blocks.9.attn.q_proj": 4}, shapes, profile="heavy")


def test_validation_checks_alpha_map_convention():
    shapes = _shapes()

    validate_rank_map(
        {"blocks.0.attn.q_proj": 4},
        shapes,
        profile="heavy",
        alpha_map={"blocks.0.attn.q_proj": 0.0},
    )
    validate_rank_map(
        {"blocks.0.attn.q_proj": 4},
        shapes,
        profile="heavy",
        alpha_map={"blocks.0.attn.q_proj": 8.0},
    )
    with pytest.raises(RankBudgetError, match="must be 0.0 or 2\\*rank"):
        validate_rank_map(
            {"blocks.0.attn.q_proj": 4},
            shapes,
            profile="heavy",
            alpha_map={"blocks.0.attn.q_proj": 7.0},
        )
    with pytest.raises(RankBudgetError, match="not present in rank_map"):
        validate_rank_map(
            {"blocks.0.attn.q_proj": 4},
            shapes,
            profile="heavy",
            alpha_map={"blocks.1.attn.q_proj": 8.0},
        )


def test_fixed_r16_and_r32_budget_calculation():
    shapes = _shapes()
    r16 = fixed_rank_map(shapes, 16, profile="heavy")
    r32 = fixed_rank_map(shapes, 32, profile="heavy")

    r16_report = rank_map_budget_report(r16, shapes, profile="heavy")
    r32_report = rank_map_budget_report(r32, shapes, profile="heavy")

    unit_params = sum(shape.unit_params for shape in shapes.values())
    assert r16_report["summary"]["tensor_bytes"] == unit_params * 16 * 2 + 3 * 4
    assert r32_report["summary"]["tensor_bytes"] == unit_params * 32 * 2 + 3 * 4
    assert r32_report["summary"]["tensor_bytes"] > r16_report["summary"]["tensor_bytes"]


def test_normalization_never_exceeds_target_budget():
    shapes = _shapes()
    source = fixed_rank_map(shapes, 32, profile="heavy")

    report = normalize_rank_map_to_target(
        source,
        shapes,
        profile="heavy",
        target="fixed-r16",
    )

    assert report["normalized_summary"]["total_bytes"] <= report["target"]["budget_bytes"]
    assert report["normalized_summary"]["budget_slack_bytes"] >= 0
    assert any(row["action"] == "demote" for row in report["changes"])
    validate_rank_map(report["rank_map"], shapes, profile="heavy", alpha_map=report["alpha_map"])


def test_normalization_is_deterministic_for_same_inputs():
    shapes = _shapes()
    source = {
        "blocks.0.attn.q_proj": 32,
        "blocks.0.attn.k_proj": 32,
        "blocks.1.attn.q_proj": 16,
    }

    first = normalize_rank_map_to_target(
        source,
        shapes,
        profile="heavy",
        target="fixed-r32-percent",
        fixed_r32_percent=0.45,
    )
    second = normalize_rank_map_to_target(
        source,
        shapes,
        profile="heavy",
        target="fixed-r32-percent",
        fixed_r32_percent=0.45,
    )

    assert first["rank_map"] == second["rank_map"]
    assert first["alpha_map"] == second["alpha_map"]
    assert first["changes"] == second["changes"]


def test_random_same_budget_control_is_budget_safe_and_deterministic():
    shapes = _shapes()
    source = {
        "blocks.0.attn.q_proj": 32,
        "blocks.0.attn.k_proj": 4,
        "blocks.1.attn.q_proj": 16,
    }

    first = random_same_budget_rank_map(source, shapes, profile="heavy", seed=11)
    second = random_same_budget_rank_map(source, shapes, profile="heavy", seed=11)

    assert first["control"] == "random_same_budget"
    assert first["rank_map"] == second["rank_map"]
    assert first["changes"] == second["changes"]
    assert first["normalized_summary"]["total_bytes"] <= first["reference_summary"]["total_bytes"]
    assert first["normalized_summary"]["budget_slack_bytes"] >= 0
    validate_rank_map(first["rank_map"], shapes, profile="heavy", alpha_map=first["alpha_map"])


def test_shuffled_discovered_control_shuffles_then_normalizes_to_budget():
    shapes = _shapes()
    source = {
        "blocks.0.attn.q_proj": 32,
        "blocks.0.attn.k_proj": 4,
        "blocks.1.attn.q_proj": 16,
    }

    report = shuffled_discovered_rank_map(source, shapes, profile="heavy", seed=3)

    assert report["control"] == "shuffled_discovered"
    assert report["initial_rank_multiset_matches_reference"] is True
    assert sorted(report["initial_rank_map"].values()) == sorted(source.values())
    assert report["shuffle_changed_adapters"] > 0
    assert report["normalized_summary"]["total_bytes"] <= report["reference_summary"]["total_bytes"]
    assert report["normalized_summary"]["budget_slack_bytes"] >= 0
    validate_rank_map(report["rank_map"], shapes, profile="heavy", alpha_map=report["alpha_map"])


def _write_pack(tmp_path: Path) -> Path:
    pack_dir = tmp_path / "hetero"
    rank_map = {
        "blocks.0.attn.q_proj": 4,
        "blocks.0.attn.k_proj": 8,
    }
    tensors = {
        "blocks.0.attn.q_proj.lora.A": np.ones((16, 4), dtype=np.float16),
        "blocks.0.attn.q_proj.lora.B": np.ones((4, 8), dtype=np.float16),
        "blocks.0.attn.q_proj.lora.alpha": np.array(8.0, dtype=np.float32),
        "blocks.0.attn.k_proj.lora.A": np.ones((4, 8), dtype=np.float16),
        "blocks.0.attn.k_proj.lora.B": np.ones((8, 8), dtype=np.float16),
        "blocks.0.attn.k_proj.lora.alpha": np.array(16.0, dtype=np.float32),
    }
    save_pack(tensors, pack_dir / "pack.safetensors")
    save_pack_metadata(
        PackMetadata(
            pack_name="hetero",
            base_hash="",
            base_model="dummy",
            profile="heavy",
            rank_map=rank_map,
            alpha_map={key: 2.0 * rank for key, rank in rank_map.items()},
            target_layers=list(rank_map),
        ),
        pack_dir / "meta.json",
    )
    return pack_dir


def test_existing_pack_shapes_can_validate_discovered_map(tmp_path: Path):
    pack_dir = _write_pack(tmp_path)
    shapes = adapter_shapes_from_pack(pack_dir)

    report = rank_map_budget_report(
        {
            "blocks.0.attn.q_proj": 4,
            "blocks.0.attn.k_proj": 8,
        },
        shapes,
        profile="heavy",
    )

    assert report["summary"]["adapter_count"] == 2
    assert report["by_target"][0]["target"] == "attn.k_proj"
    assert report["by_target"][1]["target"] == "attn.q_proj"


def test_model_shape_discovery_uses_existing_target_specs():
    class _Attention:
        def __init__(self, hidden: int):
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, hidden, bias=False)
            self.v_proj = nn.Linear(hidden, hidden, bias=False)

    class _Block:
        def __init__(self, hidden: int):
            self.self_attn = _Attention(hidden)

    model = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=[_Block(8), _Block(8)]),
        config=types.SimpleNamespace(hidden_size=8),
        model_type="gemma",
    )

    shapes = adapter_shapes_from_model(model, ["q", "attn.k_proj"])

    assert set(shapes) == {
        "blocks.0.attn.q_proj",
        "blocks.0.attn.k_proj",
        "blocks.1.attn.q_proj",
        "blocks.1.attn.k_proj",
    }
    assert shapes["blocks.0.attn.q_proj"].in_dim == 8
    assert shapes["blocks.0.attn.q_proj"].out_dim == 8


def test_normalized_payload_is_consumable_rank_map_json():
    payload = consumable_rank_map_payload(
        {"blocks.0.attn.q_proj": 4},
        {"blocks.0.attn.q_proj": 8.0},
    )

    encoded = json.loads(json.dumps(payload))
    assert encoded == {
        "rank_map": {"blocks.0.attn.q_proj": 4},
        "alpha_map": {"blocks.0.attn.q_proj": 8.0},
    }
