import json
from pathlib import Path

import numpy as np
import pytest

from mlx_plastic_rank.packs.io import (
    PackMetadata,
    load_pack_metadata,
    save_pack,
    save_pack_metadata,
)
from mlx_plastic_rank.packs.proof import DomainPackProofConfig, build_domain_pack_proof
from mlx_plastic_rank.packs.provenance import content_sha256, dataset_example_keys, pack_identity


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
    eval_data.write_text('{"prompt":"held-out","answer":"z"}\n', encoding="utf-8")
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
    config = DomainPackProofConfig(
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
    metadata = load_pack_metadata(pack_dir / "meta.json")
    metadata.training_data = str(train_data)
    metadata.training_config["provenance"] = {
        "dataset_sha256": content_sha256(train_data), "model_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "lineage_complete": True, "training_example_keys": sorted(dataset_example_keys(train_data)),
    }
    save_pack_metadata(metadata, pack_dir / "meta.json")
    identity = pack_identity(pack_dir)
    rows = json.loads(eval_report.read_text())
    for row in rows:
        row["provenance"] = {
            "version": 1, "dataset_sha256": content_sha256(eval_data),
            "model_sha256": "a" * 64, "tokenizer_sha256": "b" * 64,
            "tokenized_sha256": "c" * 64,
            "settings": {"loss_mode": "answer", "sequence_length": 64, "chat_template": False, "loader": "mlx-lm", "num_samples": 0},
            "pack": identity if row["pack"] else None,
        }
    _write_json(eval_report, rows)
    for path in (generation_report, ledger_report):
        payload = json.loads(path.read_text())
        payload["provenance"] = {
            "pack": identity, "model_sha256": "a" * 64,
            "dataset_sha256": content_sha256(eval_data),
        }
        _write_json(path, payload)
    return config


def test_domain_pack_proof_passes_for_improved_attached_pack(tmp_path: Path):
    report = build_domain_pack_proof(_passing_config(tmp_path))

    assert report["status"] == "passed"
    assert report["metrics"]["ppl_improvement_pct"] == 50.0
    assert report["metrics"]["token_accuracy_gain"] == 0.19999999999999996
    assert report["metrics"]["generation"]["solution_keyword_overlap_gain"] == 0.25
    assert {row["status"] for row in report["requirements"]} == {"passed"}


@pytest.mark.parametrize("value", [None, "invalid", True, 0.0, -1.0, float("nan"), float("inf")])
def test_proof_rejects_missing_or_impossible_perplexity(tmp_path: Path, value):
    config = _passing_config(tmp_path)
    rows = json.loads(config.eval_report.read_text())
    rows[1]["perplexity"] = value
    config.eval_report.write_text(json.dumps(rows))

    with pytest.raises(ValueError, match="perplexity"):
        build_domain_pack_proof(config)


def test_proof_rejects_ambiguous_evaluation_rows(tmp_path: Path):
    config = _passing_config(tmp_path)
    rows = json.loads(config.eval_report.read_text())
    rows.append(dict(rows[1], perplexity=2.0))
    _write_json(config.eval_report, rows)

    with pytest.raises(ValueError, match="multiple"):
        build_domain_pack_proof(config)


def test_proof_rejects_impossible_generation_rate(tmp_path: Path):
    config = _passing_config(tmp_path)
    payload = json.loads(config.generation_report.read_text())
    payload["summary"]["pack_solution_keyword_overlap"] = 2.0
    _write_json(config.generation_report, payload)

    with pytest.raises(ValueError, match="overlap"):
        build_domain_pack_proof(config)


def test_proof_rejects_training_examples_in_evaluation(tmp_path: Path):
    config = _passing_config(tmp_path)
    config.eval_data.write_text('{"answer":"y","prompt":"x","id":"different-metadata"}\n')
    report = build_domain_pack_proof(config)

    assert report["status"] == "failed"
    requirements = {row["id"]: row for row in report["requirements"]}
    assert requirements["held_out_dataset"]["status"] == "failed"


def test_proof_checks_data_seen_before_a_resumed_training_run(tmp_path: Path):
    config = _passing_config(tmp_path)
    metadata = load_pack_metadata(config.pack_dir / "meta.json")
    metadata.training_config["resume_pack"] = "earlier-training"
    metadata.training_config["provenance"]["training_example_keys"].extend(dataset_example_keys(config.eval_data))
    save_pack_metadata(metadata, config.pack_dir / "meta.json")

    report = build_domain_pack_proof(config)

    requirements = {row["id"]: row for row in report["requirements"]}
    assert requirements["held_out_dataset"]["status"] == "failed"


def test_proof_detects_full_text_overlap_even_when_prompt_fields_differ(tmp_path: Path):
    config = _passing_config(tmp_path)
    config.train_data.write_text('{"prompt":"x","answer":"y","text":"shared training text"}\n')
    config.eval_data.write_text('{"prompt":"z","answer":"w","text":"shared training text"}\n')

    report = build_domain_pack_proof(config)

    requirements = {row["id"]: row for row in report["requirements"]}
    assert requirements["held_out_dataset"]["evidence"]["overlapping_examples"] > 0


@pytest.mark.parametrize("changed", ["train", "eval", "metadata", "settings", "model", "missing"])
def test_proof_rejects_stale_or_unbound_evidence(tmp_path: Path, changed):
    config = _passing_config(tmp_path)
    if changed in {"train", "eval"}:
        path = config.train_data if changed == "train" else config.eval_data
        path.write_text('{"prompt":"changed","answer":"content"}\n')
    elif changed == "metadata":
        metadata = load_pack_metadata(config.pack_dir / "meta.json")
        metadata.notes = "changed after evaluation"
        save_pack_metadata(metadata, config.pack_dir / "meta.json")
    else:
        rows = json.loads(config.eval_report.read_text())
        if changed == "settings":
            rows[1]["provenance"]["settings"]["sequence_length"] = 128
        elif changed == "model":
            rows[1]["provenance"]["model_sha256"] = "d" * 64
        else:
            del rows[1]["provenance"]
        _write_json(config.eval_report, rows)
    assert build_domain_pack_proof(config)["status"] == "failed"


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
