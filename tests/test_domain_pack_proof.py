import json
from pathlib import Path

import numpy as np

from mlx_plastic_rank.packs.io import PackMetadata, save_pack, save_pack_metadata
from mlx_plastic_rank.packs.proof import DomainPackProofConfig, build_domain_pack_proof


def _write_pack(root: Path, name: str = "domain-pack") -> Path:
    key = "blocks.0.attn.q_proj"
    pack_dir = root / name
    save_pack(
        {
            f"{key}.lora.A": np.ones((4, 2), dtype=np.float16),
            f"{key}.lora.B": np.ones((2, 4), dtype=np.float16),
            f"{key}.lora.alpha": np.array(4.0, dtype=np.float32),
        },
        pack_dir / "pack.safetensors",
    )
    save_pack_metadata(
        PackMetadata(
            pack_name=name,
            base_hash="",
            base_model="dummy-base",
            profile="heavy",
            rank_map={key: 2},
            alpha_map={key: 4.0},
            target_layers=[key],
            training_data="train.jsonl",
            training_config={"steps": 10, "loss_mode": "answer"},
            created_at="",
        ),
        pack_dir / "meta.json",
    )
    return pack_dir


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _passing_config(tmp_path: Path) -> DomainPackProofConfig:
    pack_dir = _write_pack(tmp_path)
    train_data = tmp_path / "train.jsonl"
    eval_data = tmp_path / "eval.jsonl"
    train_data.write_text('{"prompt":"x","answer":"y"}\n', encoding="utf-8")
    eval_data.write_text('{"prompt":"x","answer":"y"}\n', encoding="utf-8")
    eval_report = _write_json(
        tmp_path / "eval-report.json",
        [
            {
                "model": "dummy-base",
                "pack": None,
                "perplexity": 10.0,
                "token_accuracy": 0.5,
                "max_logit_diff": 0.0,
            },
            {
                "model": "dummy-base",
                "pack": "domain-pack",
                "pack_size_bytes": 36,
                "perplexity": 5.0,
                "token_accuracy": 0.7,
                "max_logit_diff": 2.0,
            },
        ],
    )
    generation_report = _write_json(
        tmp_path / "generation-report.json",
        {
            "summary": {
                "pack": "domain-pack",
                "examples": 2,
                "base_solution_keyword_overlap": 0.25,
                "pack_solution_keyword_overlap": 0.5,
                "base_contains_brand_rate": 1.0,
                "pack_contains_brand_rate": 1.0,
                "base_contains_code_rate": 0.5,
                "pack_contains_code_rate": 1.0,
            }
        },
    )
    ledger_report = _write_json(
        tmp_path / "ledger-report.json",
        {
            "metadata": {"pack_name": "domain-pack"},
            "summary": {
                "adapter_count": 1,
                "declared_rank": 2,
                "effective_rank": 2,
                "rank_slack": 0,
                "rank_efficiency": 1.0,
                "bytes": 36,
                "bytes_per_effective_rank": 18.0,
            },
        },
    )
    return DomainPackProofConfig(
        base_model="dummy-base",
        pack="domain-pack",
        domain="demo",
        train_data=train_data,
        eval_data=eval_data,
        pack_dir=pack_dir,
        eval_report=eval_report,
        generation_report=generation_report,
        ledger_report=ledger_report,
        require_generation=True,
        require_ledger=True,
    )


def test_domain_pack_proof_passes_for_improved_attached_pack(tmp_path: Path):
    report = build_domain_pack_proof(_passing_config(tmp_path))

    assert report["status"] == "passed"
    assert report["metrics"]["ppl_improvement_pct"] == 50.0
    assert report["metrics"]["token_accuracy_gain"] == 0.19999999999999996
    assert report["metrics"]["generation"]["solution_keyword_overlap_gain"] == 0.25
    assert {row["status"] for row in report["requirements"]} == {"passed"}


def test_domain_pack_proof_fails_when_pack_does_not_improve(tmp_path: Path):
    config = _passing_config(tmp_path)
    eval_report = _write_json(
        tmp_path / "regression-report.json",
        [
            {
                "model": "dummy-base",
                "pack": None,
                "perplexity": 10.0,
                "token_accuracy": 0.5,
                "max_logit_diff": 0.0,
            },
            {
                "model": "dummy-base",
                "pack": "domain-pack",
                "pack_size_bytes": 36,
                "perplexity": 12.0,
                "token_accuracy": 0.4,
                "max_logit_diff": 2.0,
            },
        ],
    )
    failed_config = DomainPackProofConfig(
        base_model=config.base_model,
        pack=config.pack,
        domain=config.domain,
        train_data=config.train_data,
        eval_data=config.eval_data,
        pack_dir=config.pack_dir,
        eval_report=eval_report,
        generation_report=config.generation_report,
        ledger_report=config.ledger_report,
        require_generation=True,
        require_ledger=True,
    )

    report = build_domain_pack_proof(failed_config)

    assert report["status"] == "failed"
    by_id = {row["id"]: row for row in report["requirements"]}
    assert by_id["domain_eval_improves"]["status"] == "failed"
