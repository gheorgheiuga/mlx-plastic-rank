from pathlib import Path

import numpy as np

from mlx_plastic_rank.packs.io import PackMetadata, save_pack, save_pack_metadata
from mlx_plastic_rank.pop_polynomial_probe import (
    build_operator,
    project_operator,
    project_weight_operator,
    projection_basis,
    run_probe,
)


def _write_pack(tmp_path: Path) -> Path:
    key = "blocks.0.attn.q_proj"
    tensors = {
        f"{key}.lora.A": np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float16,
        ),
        f"{key}.lora.B": np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float16,
        ),
        f"{key}.lora.alpha": np.array(4.0, dtype=np.float32),
    }
    pack_dir = tmp_path / "probe-pack"
    save_pack(tensors, pack_dir / "pack.safetensors")
    save_pack_metadata(
        PackMetadata(
            pack_name="probe-pack",
            base_hash="",
            base_model="dummy-base",
            profile="heavy",
            rank_map={key: 2},
            alpha_map={key: 4.0},
            target_layers=[key],
            created_at="",
        ),
        pack_dir / "meta.json",
    )
    return pack_dir


def test_toy_probe_verifies_builtin_rank_identities():
    report = run_probe()

    assert report["kind"] == "pop_polynomial_probe"
    assert report["operator"]["rank"] == 3
    assert report["all_identities_hold"] is True
    assert {row["rank_gap"] for row in report["pairs"]} == {0}


def test_probe_reports_lora_adapter_overlap_for_compatible_pack(tmp_path: Path):
    pack_dir = _write_pack(tmp_path)

    report = run_probe(pack_dir=pack_dir)

    assert len(report["adapter_overlaps"]) == 1
    adapter = report["adapter_overlaps"][0]
    assert adapter["adapter"] == "blocks.0.attn.q_proj"
    assert adapter["effective_rank"] == 2
    assert adapter["matched_space"] == "output_column_space"
    first_pair = adapter["pairs"][0]
    assert first_pair["pair"] == "x_vs_x_minus_a"
    assert first_pair["overlap_rank"]["f"] == 2


def test_probe_filters_adapter_overlaps_by_key(tmp_path: Path):
    pack_dir = _write_pack(tmp_path)

    report = run_probe(pack_dir=pack_dir, adapter_keys={"blocks.9.attn.q_proj"})

    assert report["adapter_overlaps"] == []


def test_probe_projects_operator_and_adapter_space(tmp_path: Path):
    pack_dir = _write_pack(tmp_path)

    report = run_probe(pack_dir=pack_dir, projection_dim=2, projection_seed=7)

    assert report["operator"]["shape"] == [2, 2]
    assert report["operator"]["unprojected_shape"] == [4, 4]
    assert report["operator"]["projection_dim"] == 2
    assert report["all_identities_hold"] is True
    assert report["adapter_overlaps"][0]["matched_space"] == "projected_output_column_space"


def test_project_weight_operator_matches_full_projected_row_gram():
    weight = np.array(
        [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    basis = projection_basis(4, 2, seed=11)

    full = project_operator(build_operator(weight, "row-gram"), basis)
    reduced = project_weight_operator(weight, "row-gram", basis)

    np.testing.assert_allclose(reduced, full, rtol=1e-5, atol=1e-5)


def test_spectral_shift_targets_operator_eigenvalue():
    report = run_probe(spectral_shift_specs=["index:1"])

    assert report["spectral_shift_specs"] == ["index:1"]
    assert report["all_identities_hold"] is True
    pair = report["pairs"][0]
    assert pair["pair"] == "x_vs_x_minus_lambda_index_1"
    assert pair["ranks"]["g"] == 3
    assert pair["ranks"]["lcm"] == 2


def test_spectral_notch_targets_multiple_operator_eigenvalues():
    report = run_probe(spectral_notch_specs=["low:2"])

    assert report["spectral_notch_specs"] == ["low:2"]
    assert report["all_identities_hold"] is True
    pair = report["pairs"][0]
    assert pair["pair"] == "x_vs_spectral_notch_low_2"
    assert pair["ranks"]["g"] == 2
    assert pair["ranks"]["lcm"] == 2
