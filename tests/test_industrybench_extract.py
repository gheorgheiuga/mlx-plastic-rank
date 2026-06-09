import pytest

from scripts import industrybench_extract

SAMPLE_ROW = {
    "id": "42",
    "_row_idx": 7,
    "question_en": "Which part should be replaced?",
    "answer_en": "Replace the worn bearing.",
    "question": "应该更换哪个部件？",
    "answer": "更换磨损轴承。",
    "difficulty": "medium",
    "_format": "QA",
    "industry_primary": "machinery",
    "capability": "fault diagnosis",
    "knowledge_text": "A" * 12,
    "domain": "industrial procurement",
}


def test_build_example_formats_pack_ready_text():
    example = industrybench_extract.build_example(
        SAMPLE_ROW,
        "en",
        include_knowledge=False,
        knowledge_char_limit=8,
        metadata_mode="full",
        dataset=industrybench_extract.DATASET_ID,
    )

    assert example is not None
    assert example["id"] == "42:en"
    assert example["source_row_idx"] == 7
    assert example["source_dataset_url"] == industrybench_extract.DATASET_URL
    assert example["source_license"] == "MIT"
    assert "arXiv:2605.10267" in example["source_citation"]
    assert example["domain"] == "industrial procurement"
    assert "Question:\nWhich part should be replaced?" in example["text"]
    assert example["text"].endswith("Answer:\nReplace the worn bearing.")
    assert example["messages"][0]["role"] == "user"
    assert example["messages"][1]["content"] == "Replace the worn bearing."


def test_build_example_can_include_truncated_knowledge():
    example = industrybench_extract.build_example(
        SAMPLE_ROW,
        "en",
        include_knowledge=True,
        knowledge_char_limit=8,
        metadata_mode="full",
        dataset="dataset/id",
    )

    assert example is not None
    assert "Context:\nAAAAAAAA\n[truncated]" in example["prompt"]


def test_build_example_can_omit_metadata_from_prompt():
    example = industrybench_extract.build_example(
        SAMPLE_ROW,
        "en",
        include_knowledge=False,
        knowledge_char_limit=8,
        metadata_mode="none",
        dataset="dataset/id",
    )

    assert example is not None
    assert example["prompt"] == "Question:\nWhich part should be replaced?"
    assert example["industry_primary"] == "machinery"


def test_build_examples_skips_missing_translation():
    row = dict(SAMPLE_ROW)
    row["question_ru"] = ""
    row["answer_ru"] = ""

    examples = industrybench_extract.build_examples(
        [row],
        ["en", "ru"],
        include_knowledge=False,
        knowledge_char_limit=100,
        metadata_mode="full",
        dataset="dataset/id",
    )

    assert [example["language"] for example in examples] == ["en"]


def test_split_examples_is_deterministic_and_checks_size():
    rows = [{"id": str(index)} for index in range(10)]
    train, eval_rows = industrybench_extract.split_examples(
        rows,
        train_size=4,
        eval_size=2,
        seed=123,
        shuffle=True,
    )
    train_again, eval_again = industrybench_extract.split_examples(
        rows,
        train_size=4,
        eval_size=2,
        seed=123,
        shuffle=True,
    )

    assert train == train_again
    assert eval_rows == eval_again
    assert len(train) == 4
    assert len(eval_rows) == 2
    with pytest.raises(ValueError):
        industrybench_extract.split_examples(
            rows,
            train_size=9,
            eval_size=2,
            seed=123,
            shuffle=False,
        )
