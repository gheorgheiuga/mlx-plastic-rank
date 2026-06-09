from scripts import fault_codes_extract

SAMPLE_ROW = {
    "_row_idx": 3,
    "code": "10",
    "brand": "ABB",
    "brand_slug": "abb",
    "description": "Incorrect AXISMODE setting preventing next keyword execution.",
    "solution": "Verify and adjust AXISMODE or implement a Pause(IDLE) command.",
    "equipment_type": None,
    "severity": None,
    "permalink": "abb-10-fault-code-solution",
    "source_url": "https://example.test/abb-10",
}


def test_build_example_creates_diagnostic_prompt():
    example = fault_codes_extract.build_example(SAMPLE_ROW, fault_codes_extract.DATASET_ID)

    assert example is not None
    assert example["id"] == "abb-10-fault-code-solution"
    assert example["source_dataset_url"] == fault_codes_extract.DATASET_URL
    assert example["source_license"] == "CC BY-NC 4.0"
    assert example["source_license_url"] == "https://creativecommons.org/licenses/by-nc/4.0/"
    assert example["domain"] == "industrial_fault_diagnostics"
    assert "Manufacturer: ABB" in example["prompt"]
    assert "Fault code: 10" in example["prompt"]
    assert "Recommended action" in example["answer"]
    assert example["messages"][0]["role"] == "user"
    assert example["messages"][1]["role"] == "assistant"


def test_build_example_skips_incomplete_rows():
    row = dict(SAMPLE_ROW)
    row["solution"] = ""

    assert fault_codes_extract.build_example(row, "dataset/id") is None


def test_split_examples_is_deterministic():
    rows = [{"id": str(index)} for index in range(10)]
    train, eval_rows = fault_codes_extract.split_examples(
        rows,
        train_size=4,
        eval_size=2,
        seed=7,
        shuffle=True,
    )
    train_again, eval_again = fault_codes_extract.split_examples(
        rows,
        train_size=4,
        eval_size=2,
        seed=7,
        shuffle=True,
    )

    assert train == train_again
    assert eval_rows == eval_again
