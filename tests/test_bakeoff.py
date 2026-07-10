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


def _add_control_candidates(
    payload: dict,
    *,
    external_source_pack: str = "external-discovered-pack",
) -> None:
    payload["candidates"].extend(
        [
            {
                "id": "random_control",
                "pack": "demo-random-control",
                "mode": "random_same_budget",
                "control_source_candidate": "hetero_map",
                "control_seed": 17,
            },
            {
                "id": "shuffled_control",
                "pack": "demo-shuffled-control",
                "mode": "shuffled_discovered",
                "control_source_pack": external_source_pack,
                "control_seed": 23,
            },
        ]
    )


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


def test_bakeoff_parser_requires_exactly_one_control_source(tmp_path: Path):
    payload = _base_payload(tmp_path)
    payload["candidates"].append(
        {
            "id": "random_control",
            "pack": "demo-random-control",
            "mode": "random_same_budget",
        }
    )

    with pytest.raises(BakeoffError, match="exactly one"):
        validate_bakeoff_spec(payload, root=tmp_path)

    payload["candidates"][-1]["control_source_candidate"] = "hetero_map"
    payload["candidates"][-1]["control_source_pack"] = "external-pack"
    with pytest.raises(BakeoffError, match="exactly one"):
        validate_bakeoff_spec(payload, root=tmp_path)


def test_bakeoff_parser_requires_control_source_candidate_to_appear_first(tmp_path: Path):
    payload = _base_payload(tmp_path)
    payload["candidates"].insert(
        0,
        {
            "id": "random_control",
            "pack": "demo-random-control",
            "mode": "random_same_budget",
            "control_source_candidate": "hetero_map",
        },
    )

    with pytest.raises(BakeoffError, match="must appear earlier"):
        validate_bakeoff_spec(payload, root=tmp_path)


def test_bakeoff_parser_rejects_invalid_control_seed(tmp_path: Path):
    payload = _base_payload(tmp_path)
    payload["candidates"].append(
        {
            "id": "random_control",
            "pack": "demo-random-control",
            "mode": "random_same_budget",
            "control_source_candidate": "hetero_map",
            "control_seed": -1,
        }
    )

    with pytest.raises(BakeoffError, match="non-negative integer seed"):
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


def test_bakeoff_plan_generates_resumable_control_maps_before_training(tmp_path: Path):
    payload = _base_payload(tmp_path)
    _add_control_candidates(payload)
    spec = validate_bakeoff_spec(payload, root=tmp_path)

    phases = build_bakeoff_plan(spec)
    random_phases = [phase for phase in phases if phase.candidate_id == "random_control"]
    assert [phase.phase for phase in random_phases] == [
        "rank-map",
        "create",
        "eval",
        "rank-ledger",
        "proof",
    ]

    rank_map_phase, create_phase = random_phases[:2]
    assert "random-same-budget" in rank_map_phase.command
    assert rank_map_phase.command[rank_map_phase.command.index("--source-pack") + 1] == "demo-hetero"
    assert rank_map_phase.command[rank_map_phase.command.index("--seed") + 1] == "17"
    rank_map_out = rank_map_phase.command[rank_map_phase.command.index("--rank-map-out") + 1]
    assert create_phase.command[create_phase.command.index("--rank-map-json") + 1] == rank_map_out
    assert rank_map_phase.output_path == Path(rank_map_out)
    assert rank_map_phase.should_skip(force=False) is False
    rank_map_phase.output_path.parent.mkdir(parents=True, exist_ok=True)
    rank_map_phase.output_path.write_text("{}", encoding="utf-8")
    assert rank_map_phase.should_skip(force=False) is False
    assert len(rank_map_phase.additional_skip_paths) == 1
    rank_map_phase.additional_skip_paths[0].write_text("{}", encoding="utf-8")
    assert rank_map_phase.should_skip(force=False) is True
    assert rank_map_phase.should_skip(force=True) is False

    shuffled_phase = next(
        phase
        for phase in phases
        if phase.candidate_id == "shuffled_control" and phase.phase == "rank-map"
    )
    assert "shuffled-discovered" in shuffled_phase.command
    assert (
        shuffled_phase.command[shuffled_phase.command.index("--source-pack") + 1]
        == "external-discovered-pack"
    )


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


def test_bakeoff_summary_records_control_provenance(tmp_path: Path):
    payload = _base_payload(tmp_path)
    external_source = tmp_path / "external-discovered-pack"
    _write_json(
        external_source / "meta.json",
        {
            "rank_map": {"blocks.0.attn.q_proj": 4},
            "alpha_map": {"blocks.0.attn.q_proj": 8.0},
        },
    )
    _add_control_candidates(payload, external_source_pack=str(external_source))
    payload["candidates"].append(
        {
            "id": "target_constant_control",
            "pack": "demo-target-constant",
            "mode": "rank_map_json",
            "rank_map_json": "out/target-constant.json",
            "control_type": "target_constant",
            "control_source": "q16-k16-v8",
            "control_reference_bytes": 12_000,
            "control_candidate_bytes": 11_900,
        }
    )
    spec = validate_bakeoff_spec(payload, root=tmp_path)
    pack_metrics = {
        "fixed_r16": ("demo-r16", 7.0),
        "fixed_r32": ("demo-r32", 5.0),
        "dynamic_source": ("demo-dynamic", 6.2),
        "hetero_map": ("demo-hetero", 5.4),
        "random_control": ("demo-random-control", 5.1),
        "shuffled_control": ("demo-shuffled-control", 7.2),
        "target_constant_control": ("demo-target-constant", 6.8),
    }
    for phase in build_bakeoff_plan(spec):
        if phase.output_path is None:
            continue
        if phase.phase == "rank-map":
            rank_map = {"blocks.0.attn.q_proj": 4}
            alpha_map = {"blocks.0.attn.q_proj": 8.0}
            _write_json(phase.output_path, {"rank_map": rank_map, "alpha_map": alpha_map})
            report_path = Path(phase.command[phase.command.index("--out") + 1])
            _write_json(
                report_path,
                {
                    "control": phase.command[phase.command.index("rank-map") + 1].replace("-", "_"),
                    "seed": int(phase.command[phase.command.index("--seed") + 1]),
                    "reference_summary": {"total_bytes": 12_000},
                    "normalized_summary": {
                        "total_bytes": 11_800,
                        "budget_slack_bytes": 200,
                    },
                    "reference_rank_map": rank_map,
                    "reference_alpha_map": alpha_map,
                    "rank_map": rank_map,
                    "alpha_map": alpha_map,
                },
            )
            continue
        pack, ppl = pack_metrics[phase.candidate_id]
        if phase.phase == "eval":
            _write_json(
                phase.output_path,
                [
                    {"pack": None, "perplexity": 10.0, "token_accuracy": 0.5},
                    {
                        "pack": pack,
                        "size_mb": 12.0,
                        "perplexity": ppl,
                        "token_accuracy": 0.6,
                    },
                ],
            )
        elif phase.phase == "rank-ledger":
            _write_json(
                phase.output_path,
                {
                    "summary": {"declared_rank": 4, "effective_rank": 4, "rank_slack": 0},
                    "adapters": [
                        {
                            "adapter": "blocks.0.attn.q_proj",
                            "declared_rank": 4,
                            "alpha": 8.0,
                        }
                    ],
                },
            )
        elif phase.phase == "proof":
            _write_json(phase.output_path, {"status": "passed"})

    summary = build_bakeoff_summary(spec)
    random_row = next(row for row in summary["rows"] if row["candidate"] == "random_control")

    assert random_row["control_type"] == "random_same_budget"
    assert random_row["control_source"] == "demo-hetero"
    assert random_row["control_seed"] == 17
    assert random_row["control_rank_map"].endswith("random_control_rank_map.json")
    assert random_row["control_report"].endswith("random_control_rank_map_report.json")
    assert random_row["control_reference_bytes"] == 12_000
    assert random_row["control_candidate_bytes"] == 11_800
    assert random_row["control_budget_slack_bytes"] == 200
    target_constant_row = next(
        row for row in summary["rows"] if row["candidate"] == "target_constant_control"
    )
    assert target_constant_row["control_type"] == "target_constant"
    assert target_constant_row["control_source"] == "q16-k16-v8"
    assert target_constant_row["control_rank_map"] == "out/target-constant.json"
    assert target_constant_row["control_budget_slack_bytes"] == 100
    assert summary["winner_quality"] == "fixed_r32"
    assert summary["promotion_gates"]["require_beats_controls"] is True
    assert summary["promotion_gates"]["controls_passed"] is False
    assert any(
        row["control_type"] == "target_constant"
        for row in summary["promotion_gates"]["control_comparisons"]
    )
    assert summary["promotion_gates"]["passed"] is False

    random_ledger_phase = next(
        phase
        for phase in build_bakeoff_plan(spec)
        if phase.candidate_id == "random_control" and phase.phase == "rank-ledger"
    )
    ledger = json.loads(random_ledger_phase.output_path.read_text(encoding="utf-8"))
    ledger["adapters"][0]["declared_rank"] = 8
    _write_json(random_ledger_phase.output_path, ledger)
    with pytest.raises(BakeoffError, match="does not match its generated rank map"):
        build_bakeoff_summary(spec)
    ledger["adapters"][0]["declared_rank"] = 4
    _write_json(random_ledger_phase.output_path, ledger)

    random_phase = next(
        phase
        for phase in build_bakeoff_plan(spec)
        if phase.candidate_id == "random_control" and phase.phase == "rank-map"
    )
    report_path = Path(random_phase.command[random_phase.command.index("--out") + 1])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["seed"] = 999
    _write_json(report_path, report)
    with pytest.raises(BakeoffError, match="stale or mismatched seed"):
        build_bakeoff_summary(spec)
