"""Convert Gretel synthetic text-to-SQL rows into pack-ready JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "gretelai/synthetic_text_to_sql"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
DATASET_LICENSE = "Apache-2.0"
DATASET_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def fetch_rows_from_datasets(
    dataset: str,
    config: str,
    split: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    loaded = load_dataset(dataset, config, split=split)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(loaded):
        copied = dict(row)
        copied["_row_idx"] = index
        rows.append(copied)
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def dataset_notice(dataset: str) -> dict[str, str]:
    notice = {"source_dataset_url": f"https://huggingface.co/datasets/{dataset}"}
    if dataset == DATASET_ID:
        notice.update(
            {
                "source_license": DATASET_LICENSE,
                "source_license_url": DATASET_LICENSE_URL,
                "source_attribution": DATASET_ID,
            }
        )
    return notice


def build_prompt(row: dict[str, Any]) -> str:
    sql_context = clean_text(row.get("sql_context"))
    sql_prompt = clean_text(row.get("sql_prompt"))
    domain = clean_text(row.get("domain"))
    parts = [
        "You are a SQL assistant.",
        "Use the provided schema and context to write one valid SQL query.",
    ]
    if domain:
        parts.append(f"Domain: {domain}")
    parts.extend(
        [
            "Schema and context:",
            sql_context,
            "Request:",
            sql_prompt,
            "Return only the SQL query.",
        ]
    )
    return "\n".join(part for part in parts if part)


def build_answer(row: dict[str, Any]) -> str:
    return clean_text(row.get("sql"))


def build_example(row: dict[str, Any], dataset: str) -> dict[str, Any] | None:
    sql_context = clean_text(row.get("sql_context"))
    sql_prompt = clean_text(row.get("sql_prompt"))
    sql = build_answer(row)
    if not sql_context or not sql_prompt or not sql:
        return None

    prompt = build_prompt(row)
    source_id = clean_text(row.get("id")) or f"{dataset}:{row.get('_row_idx')}"
    example = {
        "id": source_id,
        "source_dataset": dataset,
        **dataset_notice(dataset),
        "source_row_idx": row.get("_row_idx"),
        "domain": clean_text(row.get("domain")) or "text_to_sql",
        "domain_description": clean_text(row.get("domain_description")),
        "sql_complexity": clean_text(row.get("sql_complexity")),
        "sql_task_type": clean_text(row.get("sql_task_type")),
        "sql_prompt": sql_prompt,
        "sql_context": sql_context,
        "sql_explanation": clean_text(row.get("sql_explanation")),
        "prompt": prompt,
        "answer": sql,
        "text": f"{prompt}\n\nAnswer:\n{sql}",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": sql},
        ],
    }
    return example


def build_examples(rows: Iterable[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        example = build_example(row, dataset)
        if example is not None:
            examples.append(example)
    return examples


def select_examples(
    examples: list[dict[str, Any]],
    *,
    size: int,
    seed: int,
    shuffle: bool,
) -> list[dict[str, Any]]:
    if size < 0:
        raise ValueError("size must be non-negative")
    if len(examples) < size:
        raise ValueError(f"Need {size} examples, found {len(examples)}")
    selected = list(examples)
    if shuffle:
        random.Random(seed).shuffle(selected)
    return selected[:size]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--config", default="default")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--train-out", type=Path, default=Path("data/text_to_sql_train_10000.jsonl"))
    parser.add_argument("--eval-out", type=Path, default=Path("data/text_to_sql_eval_1000.jsonl"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_rows = fetch_rows_from_datasets(
        args.dataset,
        args.config,
        args.train_split,
        limit=args.train_size,
    )
    eval_rows = fetch_rows_from_datasets(
        args.dataset,
        args.config,
        args.eval_split,
        limit=args.eval_size,
    )
    train_examples = select_examples(
        build_examples(train_rows, args.dataset),
        size=args.train_size,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    eval_examples = select_examples(
        build_examples(eval_rows, args.dataset),
        size=args.eval_size,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    write_jsonl(args.train_out, train_examples)
    write_jsonl(args.eval_out, eval_examples)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "license": DATASET_LICENSE if args.dataset == DATASET_ID else None,
                "train_rows": len(train_examples),
                "eval_rows": len(eval_examples),
                "train_out": str(args.train_out),
                "eval_out": str(args.eval_out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
