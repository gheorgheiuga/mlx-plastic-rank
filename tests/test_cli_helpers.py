import json
from pathlib import Path

import pytest

from mlx_plastic_rank.packs.capabilities import capability_report, missing_capabilities
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
