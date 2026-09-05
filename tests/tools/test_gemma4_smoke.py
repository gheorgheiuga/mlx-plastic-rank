import json

from scripts import gemma4_smoke


def test_parse_csv_trims_empty_values():
    assert gemma4_smoke.parse_csv(" a, ,b ,, c ") == ["a", "b", "c"]
    assert gemma4_smoke.parse_csv("") == []


def test_load_prompts_accepts_plain_text_and_jsonl(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        "\n".join(
            [
                "plain prompt",
                json.dumps({"prompt": "json prompt"}),
                json.dumps({"text": "json text"}),
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert gemma4_smoke.load_prompts(prompt_file) == [
        "plain prompt",
        "json prompt",
        "json text",
    ]


def test_default_prompts_are_available():
    prompts = gemma4_smoke.load_prompts(None)
    assert prompts
    assert "LoRA skill pack" in prompts[0]


def test_render_chat_prompt_uses_processor_template():
    class Processor:
        def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"user:{messages[0]['content']}|assistant:"

    assert gemma4_smoke.render_chat_prompt(Processor(), "hello") == "user:hello|assistant:"
