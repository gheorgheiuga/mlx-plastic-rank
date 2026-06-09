import json
from pathlib import Path

import numpy as np
import pytest

from mlx_plastic_rank.packs.dataset import (
    build_supervised_token_dataset,
    load_jsonl_texts,
    render_chat_text,
)


class TemplateTokenizer:
    pad_token_id = 0

    def encode(self, text: str):
        return [ord(char) % 255 + 1 for char in text]

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):
        assert tokenize is False
        rendered = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered = f"{rendered}\nassistant:"
        return rendered


def test_load_jsonl_texts_prefers_chat_template_messages(tmp_path: Path):
    path = tmp_path / "chat.jsonl"
    path.write_text(
        json.dumps(
            {
                "text": "plain fallback",
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_jsonl_texts(path, TemplateTokenizer(), chat_template=True) == [
        "user:question\nassistant:answer"
    ]


def test_load_jsonl_texts_uses_prompt_answer_for_chat_template(tmp_path: Path):
    path = tmp_path / "prompt_answer.jsonl"
    path.write_text(
        json.dumps({"prompt": "what", "answer": "that", "text": "plain"}),
        encoding="utf-8",
    )

    assert load_jsonl_texts(path, TemplateTokenizer(), chat_template=True) == [
        "user:what\nassistant:that"
    ]


def test_load_jsonl_texts_falls_back_to_text_without_chat_template(tmp_path: Path):
    path = tmp_path / "plain.jsonl"
    path.write_text(json.dumps({"text": "plain"}), encoding="utf-8")

    assert load_jsonl_texts(path) == ["plain"]


def test_render_chat_text_requires_chat_template():
    with pytest.raises(ValueError):
        render_chat_text(object(), {"prompt": "what", "answer": "that"})


def test_build_supervised_token_dataset_masks_prompt_tokens(tmp_path: Path):
    path = tmp_path / "supervised.jsonl"
    path.write_text(
        json.dumps({"prompt": "diagnose code 10", "answer": "adjust axis mode"}),
        encoding="utf-8",
    )

    tokens, masks = build_supervised_token_dataset(path, TemplateTokenizer(), 64)

    assert tokens.shape == (1, 64)
    mask_values = np.array(masks[0])
    assert mask_values.sum() == len(TemplateTokenizer().encode("adjust axis mode"))
    assert mask_values[0] == 0.0
    assert mask_values[-1] == 0.0


def test_build_supervised_token_dataset_uses_chat_prompt_boundary(tmp_path: Path):
    path = tmp_path / "chat_supervised.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "what now"},
                    {"role": "assistant", "content": "replace fuse"},
                ]
            }
        ),
        encoding="utf-8",
    )

    _, masks = build_supervised_token_dataset(
        path,
        TemplateTokenizer(),
        64,
        chat_template=True,
    )

    assert np.array(masks[0]).sum() > 0.0
