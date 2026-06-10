import json
from pathlib import Path

import pytest

from mlx_plastic_rank.packs.bakeoff import (
    BakeoffError,
    build_bakeoff_plan,
    build_bakeoff_summary,
    validate_bakeoff_spec,
    write_bakeoff_summary,
)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _base_payload(tmp_path: Path) -> dict:
    (tmp_path / "data").mkdir()
    (tmp_path / "data/train.jsonl").write_text('{"prompt":"p","answer":"a"}\n', encoding="utf-8")
    (tmp_path / "data/eval.jsonl").write_text('{"prompt":"p","answer":"a"}\n', encoding="utf-8")
    return {
        "name": "demo-bakeoff",
        "domain": "demo-domain",
        "base": "dummy-base",
        "loader": "mlx-vlm",
        "train_data": "data/train.jsonl",
        "eval_data": "data/eval.jsonl",
        "output_dir": "out/demo-bakeoff",
        "layers": "attn.q_proj,attn.k_proj,attn.v_proj",
        "profile": "heavy",
        "train": {
            "steps": 10,
            "batch_size": 1,
            "learning_rate": 5e-5,
            "sequence_length": 64,
            "loss_mode": "answer",
            "chat_template": True,
            "lora_dropout": 0.05,
        },
        "eval": {
            "num_samples": 4,
            "batch_size": 2,
            "sequence_length": 64,
            "loss_mode": "answer",
            "chat_template": True,
        },
        "promotion_gates": {
            "retain_fixed_r32_improvement": 0.9,
            "max_fixed_r32_size_ratio": 0.6,
        },
        "candidates": [
            {
                "id": "fixed_r16",
                "pack": "demo-r16",
                "mode": "fixed_rank",
                "rank": 16,
                "small_reference": True,
            },
            {
                "id": "fixed_r32",
                "pack": "demo-r32",
                "mode": "fixed_rank",
                "rank": 32,
                "quality_reference": True,
            },
            {
                "id": "dynamic_source",
                "pack": "demo-dynamic",
                "mode": "dynamic_rank",
                "rank": 32,
                "steps": 5,
                "dynamic_initial_rank": 8,
                "dynamic_min_rank": 4,
            },
            {
                "id": "hetero_map",
                "pack": "demo-hetero",
                "mode": "rank_map_from_candidate",
                "rank_map_from_candidate": "dynamic_source",
                "tradeoff_candidate": True,
            },
        ],
    }


def test_bakeoff_parser_rejects_missing_data(tmp_path: Path):
    payload = _base_payload(tmp_path)
    Path(tmp_path / "data/train.jsonl").unlink()

    with pytest.raises(BakeoffError, match="train_data"):
        validate_bakeoff_spec(payload, root=tmp_path)


def test_bakeoff_parser_rejects_duplicate_candidate_ids(tmp_path: Path):
    payload = _base_payload(tmp_path)
    payload["candidates"][1]["id"] = "fixed_r16"

    with pytest.raises(BakeoffError, match="Duplicate"):
        validate_bakeoff_spec(payload, root=tmp_path)


def test_bakeoff_parser_rejects_invalid_candidate_mode(tmp_path: Path):
    payload = _base_payload(tmp_path)
    payload["candidates"][0]["mode"] = "mystery"

    with pytest.raises(BakeoffError, match="Unsupported"):
        validate_bakeoff_spec(payload, root=tmp_path)


def test_bakeoff_plan_emits_deterministic_create_eval_ledger_proof_phases(tmp_path: Path):
    spec = validate_bakeoff_spec(_base_payload(tmp_path), root=tmp_path)

    phases = build_bakeoff_plan(spec)

    assert [phase.phase for phase in phases[:4]] == ["create", "eval", "rank-ledger", "proof"]
    first_create = phases[0].command
    assert first_create[first_create.index("--rank") + 1] == "16"
    assert "--chat-template" in first_create
    dynamic_create = next(phase for phase in phases if phase.candidate_id == "dynamic_source" and phase.phase == "create")
    assert "--dynamic-rank" in dynamic_create.command
    hetero_create = next(phase for phase in phases if phase.candidate_id == "hetero_map" and phase.phase == "create")
    assert hetero_create.command[hetero_create.command.index("--rank-map-from-pack") + 1] == "demo-dynamic"


def test_bakeoff_summary_computes_winners_and_promotion_gate(tmp_path: Path):
    spec = validate_bakeoff_spec(_base_payload(tmp_path), root=tmp_path)
    pack_metrics = {
        "fixed_r16": ("demo-r16", 7.0, 0.62, 10.0, 1600),
        "fixed_r32": ("demo-r32", 5.0, 0.70, 25.0, 3200),
        "dynamic_source": ("demo-dynamic", 6.2, 0.66, 14.0, 2100),
        "hetero_map": ("demo-hetero", 5.4, 0.68, 12.0, 1900),
    }
    for phase in build_bakeoff_plan(spec):
        if phase.output_path is None:
            continue
        pack, ppl, acc, size_mb, rank = pack_metrics[phase.candidate_id]
        if phase.phase == "eval":
            _write_json(
                phase.output_path,
                [
                    {"model": "dummy-base", "pack": None, "perplexity": 10.0, "token_accuracy": 0.5},
                    {
                        "model": "dummy-base",
                        "pack": pack,
                        "pack_size_bytes": int(size_mb * 1024 * 1024),
                        "size_mb": size_mb,
                        "perplexity": ppl,
                        "token_accuracy": acc,
                        "max_logit_diff": 2.0,
                    },
                ],
            )
        elif phase.phase == "rank-ledger":
            _write_json(
                phase.output_path,
                {"summary": {"declared_rank": rank, "effective_rank": rank, "rank_slack": 0}},
            )
        elif phase.phase == "proof":
            _write_json(phase.output_path, {"status": "passed"})

    summary = build_bakeoff_summary(spec)
    write_bakeoff_summary(spec, summary)

    assert summary["winner_quality"] == "fixed_r32"
    assert summary["winner_tradeoff"] == "hetero_map"
    assert summary["promotion_gates"]["passed"] is True
    assert (spec.output_dir / "demo-bakeoff_summary.json").exists()
    assert (spec.output_dir / "demo-bakeoff_summary.csv").exists()
