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
