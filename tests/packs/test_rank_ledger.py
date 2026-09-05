import math
from pathlib import Path

import numpy as np

from mlx_plastic_rank.packs.io import PackMetadata, save_pack, save_pack_metadata
from mlx_plastic_rank.packs.rank_ledger import (
    compare_pack_rank_ledgers,
    comparison_rows_for_csv,
    ledger_rows_for_csv,
    pack_rank_ledger,
)


def _write_pack(
    root: Path,
    name: str,
    tensors: dict[str, np.ndarray],
    rank_map: dict[str, int],
    alpha_map: dict[str, float],
) -> Path:
    pack_dir = root / name
    save_pack(tensors, pack_dir / "pack.safetensors")
    save_pack_metadata(
        PackMetadata(
            pack_name=name,
            base_hash="",
            base_model="dummy-base",
            profile="heavy",
            rank_map=rank_map,
            alpha_map=alpha_map,
            target_layers=list(rank_map),
            created_at="",
        ),
        pack_dir / "meta.json",
    )
    return pack_dir


def _adapter_tensors(
    key: str,
    left: np.ndarray,
    right: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    return {
        f"{key}.lora.A": left.astype(np.float16),
        f"{key}.lora.B": right.astype(np.float16),
        f"{key}.lora.alpha": np.array(alpha, dtype=np.float32),
    }


def test_pack_rank_ledger_tracks_effective_rank_and_slack(tmp_path: Path):
    full_key = "blocks.0.attn.q_proj"
    slack_key = "blocks.1.attn.q_proj"
    tensors = {}
    tensors.update(
        _adapter_tensors(
            full_key,
            np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            alpha=4.0,
        )
    )
    tensors.update(
        _adapter_tensors(
            slack_key,
            np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            alpha=4.0,
        )
    )
    pack_dir = _write_pack(
        tmp_path,
        "ledger-demo",
        tensors,
        rank_map={full_key: 2, slack_key: 2},
        alpha_map={full_key: 4.0, slack_key: 4.0},
    )

    report = pack_rank_ledger(pack_dir)

    assert report["summary"]["declared_rank"] == 4
    assert report["summary"]["effective_rank"] == 3
    assert report["summary"]["rank_slack"] == 1
    by_adapter = {row["adapter"]: row for row in report["adapters"]}
    assert by_adapter[full_key]["effective_rank"] == 2
    assert by_adapter[slack_key]["effective_rank"] == 1
    assert by_adapter[slack_key]["rank_slack"] == 1
    assert report["by_target"][0]["target"] == "attn.q_proj"
    assert report["by_target"][0]["effective_rank"] == 3

    csv_rows = ledger_rows_for_csv(report)
    assert csv_rows[0]["shape"] == "3x3"
    assert "singular_values" not in csv_rows[0]


def test_compare_pack_rank_ledgers_reports_overlap_for_identical_packs(tmp_path: Path):
    key = "blocks.0.attn.k_proj"
    tensors = _adapter_tensors(
        key,
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        alpha=4.0,
    )
    left = _write_pack(tmp_path, "left", tensors, {key: 2}, {key: 4.0})
    right = _write_pack(tmp_path, "right", tensors, {key: 2}, {key: 4.0})

    report = compare_pack_rank_ledgers(left, right)
    pair = report["pairs"][0]

    assert report["summary"]["shared_adapter_count"] == 1
    assert pair["left_effective_rank"] == 2
    assert pair["right_effective_rank"] == 2
    assert pair["composition_rank"] == 2
    assert pair["rank_savings"] == 2
    assert pair["column_overlap_rank"] == 2
    assert pair["row_overlap_rank"] == 2
    assert math.isclose(pair["fro_cosine"], 1.0, rel_tol=1e-6)

    csv_rows = comparison_rows_for_csv(report)
    assert csv_rows[0]["adapter"] == key
    assert "composition_singular_values" not in csv_rows[0]
