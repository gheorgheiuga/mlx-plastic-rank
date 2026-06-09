import json

import pytest

from scripts import fault_codes_generate_check

SAMPLE_EXAMPLE = {
    "id": "abb-10-fault-code-solution",
    "brand": "ABB",
    "code": "10",
    "solution": "Verify and adjust AXISMODE or implement a Pause(IDLE) command.",
}


def test_load_examples_honors_offset_and_limit(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": str(index)}) for index in range(5)),
        encoding="utf-8",
    )

    rows = fault_codes_generate_check.load_examples(path, limit=2, offset=1)

    assert [row["id"] for row in rows] == ["1", "2"]


def test_load_examples_rejects_empty_selection(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit):
        fault_codes_generate_check.load_examples(path, limit=2, offset=0)


def test_evidence_score_tracks_brand_code_and_solution_overlap():
    output = "For ABB fault code 10, verify AXISMODE and use a Pause command."

    score = fault_codes_generate_check.evidence_score(SAMPLE_EXAMPLE, output)

    assert score["contains_brand"] is True
    assert score["contains_code"] is True
    assert score["solution_keyword_hits"] >= 3
    assert "axismode" in score["matched_solution_keywords"]


def test_flatten_for_csv_serializes_keyword_lists():
    row = {
        "base_matched_solution_keywords": ["axismode", "pause"],
        "pack_matched_solution_keywords": ["verify"],
    }

    flattened = fault_codes_generate_check.flatten_for_csv(row)

    assert flattened["base_matched_solution_keywords"] == "axismode,pause"
    assert flattened["pack_matched_solution_keywords"] == "verify"
