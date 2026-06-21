from pathlib import Path

import numpy as np

from mlx_plastic_rank.packs.device_profiles import (
    build_memory_ledger,
    device_profiles_report,
    gb_to_bytes,
    load_rank_budget_bytes,
    memory_observations_from_eval_batch_report,
    memory_observations_from_eval_report,
    memory_observations_from_generation_report,
    pack_memory_inputs,
    parse_profile_names,
)
from mlx_plastic_rank.packs.io import PackMetadata, save_pack, save_pack_metadata


def test_device_profiles_have_expected_order_and_budgets():
    assert parse_profile_names(None) == ["8gb", "16gb", "32gb", "48gb"]
    assert parse_profile_names("8,16g,32gb,48-gb") == ["8gb", "16gb", "32gb", "48gb"]

    report = device_profiles_report()
    profiles = report["profiles"]

    assert [row["name"] for row in profiles] == ["8gb", "16gb", "32gb", "48gb"]
    assert profiles[0]["rank_ceiling"] == 8
    assert profiles[-1]["rank_ceiling"] == 64
    assert profiles[0]["soft_budget_bytes"] < profiles[0]["hard_budget_bytes"]


def test_memory_ledger_scores_profiles_from_observed_peak():
    report = build_memory_ledger(
        observations=[
            {
                "source": "manual",
                "kind": "manual",
                "metric": "observed_peak_gb",
                "peak_memory_gb": 12.0,
                "peak_memory_bytes": gb_to_bytes(12.0),
                "details": {},
            }
        ],
        adapter_tensor_bytes=1024,
    )

    statuses = {row["name"]: row["fit_status"] for row in report["profiles"]}
    assert statuses == {
        "8gb": "fail",
        "16gb": "warn",
        "32gb": "pass",
        "48gb": "pass",
    }
    assert report["smallest_soft_fit_profile"] == "32gb"
    assert report["smallest_hard_fit_profile"] == "16gb"


def test_memory_ledger_is_unknown_without_runtime_or_base_model_estimate():
    report = build_memory_ledger(adapter_tensor_bytes=2048)

    assert report["status"] == "unknown"
    assert report["memory_inputs"]["adapter_tensor_bytes"] == 2048
    assert all(row["fit_status"] == "unknown" for row in report["profiles"])


def test_memory_observation_extractors_read_current_artifact_shapes():
    eval_observations = memory_observations_from_eval_report(
        [
            {"pack": None, "peak_memory_gb": 9.0, "model": "base"},
            {"pack": "demo", "peak_memory_gb": 10.0, "model": "base"},
        ],
        source="eval.json",
        pack="demo",
    )
    batch_observations = memory_observations_from_eval_batch_report(
        [
            {
                "pack": "demo",
                "domain": "fault-codes",
                "batch_size": 1,
                "vram_peak_base": 8.0,
                "vram_peak_pack": 10.5,
            }
        ],
        source="eval_batch.json",
        pack="demo",
    )
    generation_observations = memory_observations_from_generation_report(
        {"summary": {"pack": "demo", "peak_memory_gb": 11.0, "examples": 8}},
        source="generation.json",
        pack="demo",
    )

    assert [row["peak_memory_gb"] for row in eval_observations] == [9.0, 10.0]
    assert [row["metric"] for row in batch_observations] == ["vram_peak_base", "vram_peak_pack"]
    assert generation_observations[0]["peak_memory_gb"] == 11.0


def test_pack_memory_inputs_reads_adapter_tensor_bytes(tmp_path: Path):
    pack_dir = tmp_path / "demo"
    tensors = {
        "blocks.0.attn.q_proj.lora.A": np.ones((8, 4), dtype=np.float16),
        "blocks.0.attn.q_proj.lora.B": np.ones((4, 8), dtype=np.float16),
        "blocks.0.attn.q_proj.lora.alpha": np.array(8.0, dtype=np.float32),
    }
    save_pack(tensors, pack_dir / "pack.safetensors")
    save_pack_metadata(
        PackMetadata(
            pack_name="demo",
            base_hash="",
            base_model="dummy",
            profile="heavy",
            rank_map={"blocks.0.attn.q_proj": 4},
            alpha_map={"blocks.0.attn.q_proj": 8.0},
            target_layers=["blocks.0.attn.q_proj"],
        ),
        pack_dir / "meta.json",
    )

    info = pack_memory_inputs(pack_dir)

    assert info["adapter_tensor_bytes"] == 8 * 4 * 2 + 4 * 8 * 2 + 4
    assert info["adapter_file_bytes"] >= info["adapter_tensor_bytes"]


def test_rank_budget_report_bytes_can_seed_memory_ledger(tmp_path: Path):
    path = tmp_path / "budget.json"
    path.write_text(
        """
        {
          "kind": "rank_map_budget_normalization",
          "normalized_summary": {
            "tensor_bytes": 1234,
            "total_bytes": 5678
          }
        }
        """,
        encoding="utf-8",
    )

    budget_bytes, summary = load_rank_budget_bytes(path)
    report = build_memory_ledger(
        adapter_tensor_bytes=budget_bytes,
        base_model_bytes=gb_to_bytes(4.0),
        profiles=["8gb"],
        rank_budget_sources=[{"source": str(path), "adapter_tensor_bytes": budget_bytes}],
    )

    assert budget_bytes == 1234
    assert summary["total_bytes"] == 5678
    assert report["memory_inputs"]["adapter_tensor_bytes"] == 1234
    assert report["profiles"][0]["fit_status"] == "pass"
