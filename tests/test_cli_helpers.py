import json
from pathlib import Path

import pytest

from mlx_plastic_rank.packs.eval_utils import (
    apply_thinking_strategy,
    load_domain_prompts,
    parse_batch_sizes,
    parse_thinking_option,
)


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
