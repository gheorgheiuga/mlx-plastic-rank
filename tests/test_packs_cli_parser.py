import pytest

from mlx_plastic_rank.packs.cli import build_parser


def test_create_parser_accepts_rank_strategy():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-mxfp8",
            "--data",
            "data/domain_prompts.jsonl",
            "--rank-strategy",
            "stable",
        ]
    )
    assert args.rank_strategy == "stable"


def test_create_parser_keeps_lite_projection_default():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-mxfp8",
            "--data",
            "data/domain_prompts.jsonl",
        ]
    )
    assert args.loader == "auto"
    assert args.layers == "attn.q_proj,attn.k_proj,attn.v_proj"


def test_create_parser_accepts_explicit_vlm_loader():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-mxfp8",
            "--data",
            "data/domain_prompts.jsonl",
            "--loader",
            "mlx-vlm",
        ]
    )
    assert args.loader == "mlx-vlm"


def test_create_parser_accepts_chat_template():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/industrybench_en_train.jsonl",
            "--chat-template",
        ]
    )
    assert args.chat_template is True


def test_create_parser_accepts_answer_loss_mode():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/fault_codes_train.jsonl",
            "--loss-mode",
            "answer",
        ]
    )
    assert args.loss_mode == "answer"


def test_create_parser_accepts_resume_pack():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "phase-two",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/fault_codes_train.jsonl",
            "--resume-pack",
            "phase-one",
        ]
    )
    assert args.resume_pack == "phase-one"


def test_create_parser_accepts_rank_map_from_pack():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "hetero-scratch",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/fault_codes_train.jsonl",
            "--rank-map-from-pack",
            "phase-one",
        ]
    )
    assert args.rank_map_from_pack == "phase-one"


def test_create_parser_accepts_rank_map_json():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "spectral-map",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/fault_codes_train.jsonl",
            "--rank-map-json",
            "out/spectral_key_rank_map.json",
        ]
    )
    assert args.rank_map_json == "out/spectral_key_rank_map.json"


def test_eval_parser_accepts_chat_template():
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data-path",
            "data/industrybench_en_eval.jsonl",
            "--chat-template",
        ]
    )
    assert args.chat_template is True


def test_eval_parser_accepts_answer_loss_mode():
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data-path",
            "data/fault_codes_eval.jsonl",
            "--loss-mode",
            "answer",
        ]
    )
    assert args.loss_mode == "answer"


def test_rank_ledger_parser_accepts_compare_and_outputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "rank-ledger",
            "--name",
            "fault-codes-a",
            "--compare",
            "fault-codes-b",
            "--rank-tol",
            "1e-4",
            "--out",
            "out/ledger.json",
            "--csv",
            "out/ledger.csv",
        ]
    )
    assert args.name == "fault-codes-a"
    assert args.compare == "fault-codes-b"
    assert args.rank_tol == 1e-4
    assert args.out == "out/ledger.json"
    assert args.csv == "out/ledger.csv"


def test_rank_map_spectral_parser_accepts_probe_inputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "rank-map",
            "spectral",
            "--source-pack",
            "hetero-source",
            "--q-spectral",
            "out/q.json",
            "--k-spectral",
            "out/k.json",
            "--v-spectral",
            "out/v.json",
            "--profile",
            "heavy",
            "--policy",
            "balanced",
            "--out",
            "out/rank-map.json",
        ]
    )
    assert args.command == "rank-map"
    assert args.rank_map_command == "spectral"
    assert args.source_pack == "hetero-source"
    assert args.q_spectral == "out/q.json"
    assert args.k_spectral == "out/k.json"
    assert args.v_spectral == "out/v.json"
    assert args.profile == "heavy"
    assert args.out == "out/rank-map.json"


def test_proof_parser_accepts_artifact_inputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "proof",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--pack",
            "domain-pack",
            "--domain",
            "fault-codes",
            "--train-data",
            "data/train.jsonl",
            "--eval-data",
            "data/eval.jsonl",
            "--eval-report",
            "out/eval.json",
            "--generation-report",
            "out/generation.json",
            "--ledger-report",
            "out/ledger.json",
            "--require-generation",
            "--require-ledger",
            "--fail-on-regression",
            "--out",
            "out/proof.json",
        ]
    )
    assert args.command == "proof"
    assert args.pack == "domain-pack"
    assert args.domain == "fault-codes"
    assert args.require_generation is True
    assert args.require_ledger is True
    assert args.fail_on_regression is True


def test_bakeoff_parser_accepts_spec_dry_run_and_force():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bakeoff",
            "--spec",
            "codex/bakeoffs/text_to_sql_gemma4_it_fullscale.json",
            "--dry-run",
            "--force",
        ]
    )
    assert args.command == "bakeoff"
    assert args.spec == "codex/bakeoffs/text_to_sql_gemma4_it_fullscale.json"
    assert args.dry_run is True
    assert args.force is True


def test_create_parser_rejects_invalid_dropout():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "create",
                "--name",
                "demo",
                "--base",
                "mlx-community/gemma-4-12B-mxfp8",
                "--data",
                "data/domain_prompts.jsonl",
                "--lora-dropout",
                "1.0",
            ]
        )


def test_create_parser_accepts_heavy_profile():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-mxfp8",
            "--data",
            "data/domain_prompts.jsonl",
            "--profile",
            "heavy",
        ]
    )
    assert args.profile == "heavy"


def test_create_parser_accepts_min_rank():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-mxfp8",
            "--data",
            "data/domain_prompts.jsonl",
            "--min-rank",
            "16",
        ]
    )
    assert args.min_rank == 16


def test_create_parser_accepts_explicit_rank():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "demo",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/industrybench_en_train.jsonl",
            "--rank",
            "8",
        ]
    )
    assert args.rank == 8


def test_create_parser_accepts_dynamic_rank_controls():
    parser = build_parser()
    args = parser.parse_args(
        [
            "create",
            "--name",
            "dynamic-demo",
            "--base",
            "mlx-community/gemma-4-12B-it-qat-mxfp8",
            "--data",
            "data/fault_codes_train.jsonl",
            "--rank",
            "32",
            "--dynamic-rank",
            "--dynamic-initial-rank",
            "4",
            "--dynamic-min-rank",
            "2",
            "--dynamic-rank-interval",
            "25",
            "--dynamic-rank-warmup",
            "50",
            "--dynamic-grow-threshold",
            "0.4",
            "--dynamic-prune-threshold",
            "0.05",
        ]
    )
    assert args.rank == 32
    assert args.dynamic_rank is True
    assert args.dynamic_initial_rank == 4
    assert args.dynamic_min_rank == 2
    assert args.dynamic_rank_interval == 25
    assert args.dynamic_rank_warmup == 50
    assert args.dynamic_grow_threshold == 0.4
    assert args.dynamic_prune_threshold == 0.05


def test_create_parser_rejects_zero_explicit_rank():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "create",
                "--name",
                "demo",
                "--base",
                "mlx-community/gemma-4-12B-mxfp8",
                "--data",
                "data/domain_prompts.jsonl",
                "--rank",
                "0",
            ]
        )


def test_route_parser_accepts_required_args(tmp_path):
    parser = build_parser()
    mapping = tmp_path / "map.json"
    reqs = tmp_path / "reqs.jsonl"
    mapping.write_text("{}", encoding="utf-8")
    reqs.write_text("", encoding="utf-8")
    args = parser.parse_args(
        [
            "route",
            "--base",
            "mlx-community/gemma-4-12B-mxfp8",
            "--domain-map",
            str(mapping),
            "--input",
            str(reqs),
            "--probe-forward",
        ]
    )
    assert args.command == "route"
    assert args.probe_forward is True


def test_capabilities_parser_accepts_json_check():
    parser = build_parser()
    args = parser.parse_args(["capabilities", "--json", "--check"])
    assert args.command == "capabilities"
    assert args.json is True
    assert args.check is True
