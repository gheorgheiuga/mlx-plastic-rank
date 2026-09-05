import json
from pathlib import Path

import numpy as np

from mlx_plastic_rank.packs.ablation import (
    ablation_rows_for_csv,
    build_rank_channel_ablation_report,
)
from mlx_plastic_rank.packs.io import PackMetadata, load_pack, save_pack, save_pack_metadata


def _write_pack(tmp_path: Path) -> Path:
    pack_dir = tmp_path / "demo"
    tensors = {
        "blocks.0.attn.q_proj.lora.A": np.array(
            [
                [3.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float16,
        ),
        "blocks.0.attn.q_proj.lora.B": np.array(
            [
                [4.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=np.float16,
        ),
        "blocks.0.attn.q_proj.lora.alpha": np.array(4.0, dtype=np.float32),
        "blocks.0.attn.k_proj.lora.A": np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float16,
        ),
        "blocks.0.attn.k_proj.lora.B": np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float16,
        ),
        "blocks.0.attn.k_proj.lora.alpha": np.array(4.0, dtype=np.float32),
    }
    save_pack(tensors, pack_dir / "pack.safetensors")
    save_pack_metadata(
        PackMetadata(
            pack_name="demo",
            base_hash="",
            base_model="dummy",
            profile="heavy",
            rank_map={
                "blocks.0.attn.q_proj": 2,
                "blocks.0.attn.k_proj": 2,
            },
            alpha_map={
                "blocks.0.attn.q_proj": 4.0,
                "blocks.0.attn.k_proj": 4.0,
            },
            target_layers=[
                "blocks.0.attn.q_proj",
                "blocks.0.attn.k_proj",
            ],
        ),
        pack_dir / "meta.json",
    )
    return pack_dir


def test_channel_ablation_report_ranks_channels_by_proxy(tmp_path: Path):
    pack_dir = _write_pack(tmp_path)

    report = build_rank_channel_ablation_report(pack_dir, unit="channel", top_k=2)

    assert report["kind"] == "causal_rank_channel_ablation_report"
    assert report["evidence_status"] == "mechanistic_proxy_only"
    assert report["summary"]["channel_count"] == 4
    assert report["ablations"][0]["ablation_id"] == "channel-blocks_0_attn_q_proj-c0000"
    assert report["ablations"][0]["proxy_removed_fro"] == 24.0
    assert report["ablations"][1]["ablation_id"] == "channel-blocks_0_attn_q_proj-c0001"
    assert report["ablations"][1]["proxy_removed_fro"] == 4.0


def test_generated_ablation_pack_zeroes_channel_without_mutating_source(tmp_path: Path):
    pack_dir = _write_pack(tmp_path)
    source_before = load_pack(pack_dir / "pack.safetensors")

    report = build_rank_channel_ablation_report(
        pack_dir,
        unit="channel",
        top_k=1,
        ablation_pack_root=tmp_path / "ablations",
    )

    generated_dir = Path(report["ablations"][0]["generated_pack_dir"])
    source_after = load_pack(pack_dir / "pack.safetensors")
    generated = load_pack(generated_dir / "pack.safetensors")

    assert np.array_equal(
        source_before["blocks.0.attn.q_proj.lora.A"],
        source_after["blocks.0.attn.q_proj.lora.A"],
    )
    assert np.array_equal(
        source_before["blocks.0.attn.q_proj.lora.B"],
        source_after["blocks.0.attn.q_proj.lora.B"],
    )
    assert np.all(generated["blocks.0.attn.q_proj.lora.A"][:, 0] == 0)
    assert np.all(generated["blocks.0.attn.q_proj.lora.B"][0, :] == 0)
    assert np.any(generated["blocks.0.attn.q_proj.lora.A"][:, 1] != 0)


def test_prefix_ablation_and_paired_eval_deltas(tmp_path: Path):
    pack_dir = _write_pack(tmp_path)
    baseline = tmp_path / "baseline_eval.json"
    ablated = tmp_path / "ablated_eval.json"
    baseline.write_text(
        json.dumps(
            [
                {"pack": None, "perplexity": 20.0, "token_accuracy": 0.1},
                {
                    "pack": "demo",
                    "perplexity": 10.0,
                    "token_accuracy": 0.5,
                    "domain_metric": 0.5,
                },
            ]
        ),
        encoding="utf-8",
    )
    ablated.write_text(
        json.dumps(
            [
                {"pack": None, "perplexity": 20.0, "token_accuracy": 0.1},
                {
                    "pack": "prefix-blocks_0_attn_q_proj-keep0001",
                    "perplexity": 12.0,
                    "token_accuracy": 0.4,
                    "domain_metric": 0.45,
                },
            ]
        ),
        encoding="utf-8",
    )

    report = build_rank_channel_ablation_report(
        pack_dir,
        unit="prefix",
        prefix_rank=1,
        top_k=1,
        baseline_eval=baseline,
        ablation_eval_reports={
            "prefix-blocks_0_attn_q_proj-keep0001": ablated,
        },
    )

    row = report["ablations"][0]
    assert report["evidence_status"] == "paired_eval"
    assert row["operation"] == "zero_suffix_after_prefix"
    assert row["eval"]["perplexity_delta"] == 2.0
    assert row["eval"]["token_accuracy_delta"] == -0.09999999999999998
    assert ablation_rows_for_csv(report)[0]["perplexity_delta"] == 2.0


def test_target_ablation_groups_matching_adapters(tmp_path: Path):
    pack_dir = _write_pack(tmp_path)

    report = build_rank_channel_ablation_report(
        pack_dir,
        unit="target",
        targets=["attn.q_proj"],
        top_k=0,
    )

    assert len(report["ablations"]) == 1
    assert report["ablations"][0]["target"] == "attn.q_proj"
    assert report["ablations"][0]["adapter_count"] == 1
    assert report["ablations"][0]["channel_count"] == 2
