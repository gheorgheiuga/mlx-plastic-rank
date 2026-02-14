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
