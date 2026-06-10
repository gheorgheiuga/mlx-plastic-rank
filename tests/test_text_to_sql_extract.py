from scripts import text_to_sql_extract

SAMPLE_ROW = {
    "_row_idx": 5,
    "id": 5097,
    "domain": "forestry",
    "domain_description": "Sustainable forest management data.",
    "sql_complexity": "single join",
    "sql_task_type": "analytics and reporting",
    "sql_prompt": "What is the total timber volume sold by each salesperson?",
    "sql_context": "CREATE TABLE salesperson (salesperson_id INT, name TEXT);",
    "sql": "SELECT salesperson_id, SUM(volume) FROM timber_sales GROUP BY salesperson_id;",
    "sql_explanation": "Groups timber sales by salesperson.",
}


def test_build_example_creates_text_to_sql_prompt_with_license():
    example = text_to_sql_extract.build_example(SAMPLE_ROW, text_to_sql_extract.DATASET_ID)

    assert example is not None
    assert example["id"] == "5097"
    assert example["source_dataset_url"] == text_to_sql_extract.DATASET_URL
    assert example["source_license"] == "Apache-2.0"
    assert example["source_license_url"] == "https://www.apache.org/licenses/LICENSE-2.0"
    assert example["domain"] == "forestry"
    assert "Schema and context:" in example["prompt"]
    assert "CREATE TABLE salesperson" in example["prompt"]
    assert "Return only the SQL query." in example["prompt"]
    assert example["answer"].startswith("SELECT salesperson_id")
    assert example["messages"][0]["role"] == "user"
    assert example["messages"][1]["role"] == "assistant"


def test_build_example_skips_incomplete_text_to_sql_rows():
    row = dict(SAMPLE_ROW)
    row["sql"] = ""

    assert text_to_sql_extract.build_example(row, text_to_sql_extract.DATASET_ID) is None


def test_select_examples_is_deterministic():
    rows = [{"id": str(index)} for index in range(10)]
    selected = text_to_sql_extract.select_examples(rows, size=4, seed=11, shuffle=True)
    selected_again = text_to_sql_extract.select_examples(rows, size=4, seed=11, shuffle=True)

    assert selected == selected_again
    assert len(selected) == 4
