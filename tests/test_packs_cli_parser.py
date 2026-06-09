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
            "qwen3-4b-2507-mlx-4bit",
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


def test_create_parser_rejects_invalid_dropout():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "create",
                "--name",
                "demo",
                "--base",
                "qwen3-4b-2507-mlx-4bit",
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
            "qwen3-4b-2507-mlx-4bit",
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
            "qwen3-4b-2507-mlx-4bit",
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


def test_create_parser_rejects_zero_explicit_rank():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "create",
                "--name",
                "demo",
                "--base",
                "qwen3-4b-2507-mlx-4bit",
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
            "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
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
