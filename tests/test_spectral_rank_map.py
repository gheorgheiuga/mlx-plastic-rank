import json
from pathlib import Path

import numpy as np

from mlx_plastic_rank.packs.io import PackMetadata, save_pack, save_pack_metadata
from mlx_plastic_rank.packs.rank_map import (
    SpectralRankMapConfig,
    build_spectral_rank_map_candidate,
)


def _adapter_tensors(key: str, out_dim: int, in_dim: int, rank: int) -> dict[str, np.ndarray]:
    return {
        f"{key}.lora.A": np.ones((out_dim, rank), dtype=np.float16),
        f"{key}.lora.B": np.ones((rank, in_dim), dtype=np.float16),
        f"{key}.lora.alpha": np.array(2.0 * rank, dtype=np.float32),
    }


def _write_pack(root: Path, rank_map: dict[str, int]) -> Path:
    tensors: dict[str, np.ndarray] = {}
    for key, rank in rank_map.items():
        tensors.update(_adapter_tensors(key, out_dim=4, in_dim=4, rank=rank))
    pack_dir = root / "source-pack"
    save_pack(tensors, pack_dir / "pack.safetensors")
    save_pack_metadata(
        PackMetadata(
            pack_name="source-pack",
            base_hash="",
            base_model="dummy-base",
            profile="heavy",
            rank_map=rank_map,
            alpha_map={key: 2.0 * rank for key, rank in rank_map.items()},
            target_layers=list(rank_map),
            created_at="",
        ),
        pack_dir / "meta.json",
    )
    return pack_dir


def _write_spectral(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path


def test_spectral_rank_map_promotes_key_and_demotes_weakest_queries(tmp_path: Path):
    key_adapter = "blocks.5.attn.k_proj"
    weak_query = "blocks.2.attn.q_proj"
    next_query = "blocks.3.attn.q_proj"
    pack_dir = _write_pack(
        tmp_path,
        {
            key_adapter: 8,
            weak_query: 8,
            next_query: 8,
        },
    )
    spectral_path = _write_spectral(
        tmp_path / "spectral.json",
        [
            {
                "adapter": key_adapter,
                "layer": 5,
                "layer_type": "full_attention",
                "low_8_lift": 2.2,
            },
            {
                "adapter": weak_query,
                "layer": 2,
                "layer_type": "sliding_attention",
                "low_8_lift": 0.7,
            },
            {
                "adapter": next_query,
                "layer": 3,
                "layer_type": "sliding_attention",
                "low_8_lift": 0.9,
            },
        ],
    )

    report = build_spectral_rank_map_candidate(
        pack_dir,
        [spectral_path],
        SpectralRankMapConfig(
            allowed_ranks=(4, 8, 16),
            promote_rank=16,
            demote_min_rank=4,
        ),
    )

    assert report["rank_map"][key_adapter] == 16
    assert report["rank_map"][weak_query] == 4
    assert report["rank_map"][next_query] == 4
    assert report["candidate_estimated_lora_params"] == report["original_estimated_lora_params"]
    assert [row["adapter"] for row in report["promotions"]] == [key_adapter]
    assert [row["adapter"] for row in report["reductions"]] == [weak_query, next_query]
    assert report["alpha_map"][key_adapter] == 32.0


def test_spectral_rank_map_leaves_budget_unchanged_without_promotions(tmp_path: Path):
    key_adapter = "blocks.5.attn.k_proj"
    query_adapter = "blocks.2.attn.q_proj"
    pack_dir = _write_pack(tmp_path, {key_adapter: 8, query_adapter: 8})
    spectral_path = _write_spectral(
        tmp_path / "spectral.json",
        [
            {
                "adapter": key_adapter,
                "layer": 5,
                "layer_type": "full_attention",
                "low_8_lift": 1.1,
            },
            {
                "adapter": query_adapter,
                "layer": 2,
                "layer_type": "sliding_attention",
                "low_8_lift": 0.7,
            },
        ],
    )

    report = build_spectral_rank_map_candidate(
        pack_dir,
        [spectral_path],
        SpectralRankMapConfig(allowed_ranks=(4, 8, 16), promote_rank=16),
    )

    assert report["rank_map"] == {key_adapter: 8, query_adapter: 8}
    assert report["promotions"] == []
    assert report["reductions"] == []
