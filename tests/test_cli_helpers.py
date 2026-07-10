import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_plastic_rank.packs import cli
from mlx_plastic_rank.packs.capabilities import capability_report, missing_capabilities
from mlx_plastic_rank.packs.cli import _load_rank_map_json
from mlx_plastic_rank.packs.eval_utils import (
    apply_thinking_strategy,
    load_domain_prompts,
    parse_batch_sizes,
    parse_thinking_option,
)
from mlx_plastic_rank.packs.train import extract_logits


def test_parse_batch_sizes_deduplicates_and_sorts():
    assert parse_batch_sizes("16,8,16") == [8, 16]
    with pytest.raises(SystemExit):
        parse_batch_sizes("")
    with pytest.raises(SystemExit):
        parse_batch_sizes("foo")


def test_parse_thinking_option():
    assert parse_thinking_option("keep") == ("keep", None)
    assert parse_thinking_option("strip") == ("strip", None)
    assert parse_thinking_option("cap=32") == ("cap", 32)
    with pytest.raises(SystemExit):
        parse_thinking_option("cap=")
    with pytest.raises(SystemExit):
        parse_thinking_option("unknown")


def test_apply_thinking_strategy_strip_and_cap():
    text = "Before <think>internal monologue that is very long</think> after"
    stripped = apply_thinking_strategy(text, "strip", None)
    assert "internal monologue" not in stripped
    assert stripped.startswith("Before") and stripped.endswith("after")

    capped = apply_thinking_strategy(text, "cap", 2)
    assert "internal monologue" in capped
    assert "very" not in capped


def test_load_domain_prompts(tmp_path: Path):
    data = [
        {"domain": "general", "text": "hello"},
        {"domain": "domain", "text": "<<think>>plan<</think>> answer"},
    ]
    path = tmp_path / "prompts.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in data), encoding="utf-8")

    prompts = load_domain_prompts(path, "strip", None)
    assert set(prompts.keys()) == {"general", "domain"}
    assert prompts["domain"][0].strip() == "answer"


def test_load_rank_map_json_defaults_alpha(tmp_path: Path):
    path = tmp_path / "rank-map.json"
    path.write_text(
        json.dumps(
            {
                "rank_map": {
                    "blocks.0.attn.q_proj": 4,
                    "blocks.0.attn.k_proj": 8,
                }
            }
        ),
        encoding="utf-8",
    )

    rank_map, alpha_map = _load_rank_map_json(str(path), (4, 8, 16))

    assert rank_map["blocks.0.attn.q_proj"] == 4
    assert alpha_map["blocks.0.attn.q_proj"] == 8.0
    assert alpha_map["blocks.0.attn.k_proj"] == 16.0


def test_load_rank_map_json_rejects_unsupported_rank(tmp_path: Path):
    path = tmp_path / "rank-map.json"
    path.write_text(json.dumps({"blocks.0.attn.q_proj": 3}), encoding="utf-8")

    with pytest.raises(SystemExit):
        _load_rank_map_json(str(path), (4, 8, 16))


def test_extract_logits_accepts_output_containers():
    raw = object()
    container = type("Output", (), {"logits": raw})()
    assert extract_logits(container) is raw
    assert extract_logits({"logits": raw}) is raw
    assert extract_logits(raw) is raw


def test_capability_report_includes_modality_stack():
    rows = capability_report()
    by_name = {row["name"]: row for row in rows}

    assert {"mlx-lm", "mlx-vlm", "mlx-audio"} <= set(by_name)
    assert "Gemma 4 unified" in by_name["mlx-vlm"]["summary"]
    assert "speech-to-text" in by_name["mlx-audio"]["features"]
    assert missing_capabilities(rows) == [
        row["name"] for row in rows if not row["installed"]
    ]


def test_eval_answer_json_reports_dataset_coverage(tmp_path: Path, monkeypatch):
    class Tokenizer:
        pad_token_id = 0

        def encode(self, text: str):
            return [ord(char) % 255 + 1 for char in text]

    class Manager:
        def __init__(self, model, *, base_checkpoint, base_model):
            pass

    data_path = tmp_path / "eval.jsonl"
    data_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "one", "answer": "aa"}),
                json.dumps({"prompt": "two", "answer": "bbb"}),
                json.dumps({"text": "not supervised"}),
            ]
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "metrics.json"

    monkeypatch.setattr(cli, "_resolve_base_checkpoint", lambda model_ref: None)
    monkeypatch.setattr(
        cli,
        "_load_base_model",
        lambda model_ref, loader: (object(), Tokenizer()),
    )
    monkeypatch.setattr(cli, "LoRAManager", Manager)
    monkeypatch.setattr(
        cli,
        "_evaluate_supervised_perplexity",
        lambda model, tokens, masks, batch_size: {
            "ppl": 1.0,
            "ppl_se": 0.0,
            "ppl_se_method": "example_cluster_delta",
            "token_accuracy": 1.0,
            "tps": 1.0,
            "first_token_ms": 0.0,
            "vram_peak": 0.0,
            "eval_time_s": 0.0,
            "tokens": 2,
            "example_loss_sums": [0.0],
            "example_token_counts": [2],
            "example_correct_counts": [2],
        },
    )
    monkeypatch.setattr(
        cli,
        "model_logits",
        lambda model, inputs: mx.zeros((inputs.shape[0], inputs.shape[1], 256)),
    )

    args = cli.build_parser().parse_args(
        [
            "eval",
            "--base",
            "dummy",
            "--data-path",
            str(data_path),
            "--loss-mode",
            "answer",
            "--sequence-length",
            "64",
            "--num-samples",
            "1",
            "--batch-size",
            "1",
            "--out",
            str(out_path),
        ]
    )

    cli.cmd_eval(args)

    metrics = json.loads(out_path.read_text(encoding="utf-8"))[0]
    assert metrics["source_rows"] == 3
    assert metrics["included_rows"] == 1
    assert metrics["excluded_rows"] == 2
    assert metrics["invalid_rows"] == 1
    assert metrics["sample_limited_rows"] == 1
    assert metrics["truncated_included_rows"] == 0
    assert metrics["reference_answer_tokens_total"] == 5
    assert metrics["reference_answer_tokens_retained"] == 2
    assert metrics["ppl_se_method"] == "example_cluster_delta"
    assert metrics["example_loss_sums"] == [0.0]
    assert metrics["example_token_counts"] == [2]
    assert metrics["example_correct_counts"] == [2]


def test_supervised_eval_computes_loss_and_accuracy_in_one_forward_pass_per_batch():
    class NextTokenModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, inputs):
            self.calls += 1
            targets = inputs + 1
            return mx.eye(4, dtype=mx.float32)[targets] * 20.0

    model = NextTokenModel()
    tokens = mx.array([[0, 1, 2], [0, 1, 2]], dtype=mx.int32)
    masks = mx.ones(tokens.shape, dtype=mx.float32)

    metrics = cli._evaluate_supervised_perplexity(model, tokens, masks, batch_size=1)

    assert model.calls == 2
    assert metrics["token_accuracy"] == pytest.approx(1.0)
    assert metrics["example_token_counts"] == [2, 2]
    assert metrics["example_correct_counts"] == [2, 2]


def test_clustered_perplexity_se_uses_example_clusters():
    perplexity = 2.0
    result = cli._clustered_perplexity_se(
        [1.0, 5.0],
        [1, 2],
        mean_loss=2.0,
        perplexity=perplexity,
    )

    assert result == pytest.approx(perplexity * (2.0 / 3.0))


def test_clustered_perplexity_se_rejects_mismatched_vectors():
    with pytest.raises(ValueError, match="equal length"):
        cli._clustered_perplexity_se(
            [1.0],
            [1, 2],
            mean_loss=1.0,
            perplexity=2.0,
        )
