"""Convert industrial fault-code rows into pack-ready JSONL."""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "avneetsingla/industrial-fault-codes-sample"
DATASET_SERVER = "https://datasets-server.huggingface.co"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
DATASET_LICENSE = "CC BY-NC 4.0"
DATASET_LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def dataset_viewer_get(endpoint: str, params: dict[str, Any], *, retries: int = 3) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{DATASET_SERVER}/{endpoint}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "mlx-plastic-rank/fault-codes"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_rows_from_datasets(
    dataset: str,
    config: str,
    split: str,
    *,
    source_limit: int | None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    loaded = load_dataset(dataset, config, split=split)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(loaded):
        copied = dict(row)
        copied["_row_idx"] = index
        rows.append(copied)
        if source_limit is not None and len(rows) >= source_limit:
            break
    return rows


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    *,
    source_limit: int | None,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        if source_limit is not None and len(rows) >= source_limit:
            break
        remaining = None if source_limit is None else source_limit - len(rows)
        length = min(page_size, remaining) if remaining is not None else page_size
        payload = dataset_viewer_get(
            "rows",
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
        )
        total = int(payload.get("num_rows_total") or 0)
        page_rows = payload.get("rows") or []
        if not page_rows:
            break
        for entry in page_rows:
            row = entry.get("row", entry)
            if isinstance(row, dict):
                copied = dict(row)
                copied["_row_idx"] = entry.get("row_idx", len(rows))
                rows.append(copied)
        offset += len(page_rows)
    return rows


def build_prompt(row: dict[str, Any]) -> str:
    brand = clean_text(row.get("brand")) or "Unknown manufacturer"
    code = clean_text(row.get("code")) or "Unknown code"
    description = clean_text(row.get("description"))
    prompt = [
        "You are an industrial maintenance assistant.",
        f"Manufacturer: {brand}",
        f"Fault code: {code}",
    ]
    if description:
        prompt.append(f"Observed description: {description}")
    prompt.append("Explain what this fault indicates and the recommended technician action.")
    return "\n".join(prompt)


def build_answer(row: dict[str, Any]) -> str:
    brand = clean_text(row.get("brand")) or "the manufacturer"
    code = clean_text(row.get("code")) or "the reported fault code"
    description = clean_text(row.get("description"))
    solution = clean_text(row.get("solution"))
    parts = [f"For {brand} fault code {code}:"]
    if description:
        parts.append(f"- Meaning: {description}")
    if solution:
        parts.append(f"- Recommended action: {solution}")
    return "\n".join(parts)


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


def build_example(row: dict[str, Any], dataset: str) -> dict[str, Any] | None:
    code = clean_text(row.get("code"))
    brand = clean_text(row.get("brand"))
    description = clean_text(row.get("description"))
    solution = clean_text(row.get("solution"))
    if not code or not brand or not description or not solution:
        return None
    prompt = build_prompt(row)
    answer = build_answer(row)
    text = f"{prompt}\n\nAnswer:\n{answer}"
    source_id = clean_text(row.get("permalink")) or f"{brand}:{code}:{row.get('_row_idx')}"
    return {
        "id": source_id,
        "source_dataset": dataset,
        **dataset_notice(dataset),
        "source_row_idx": row.get("_row_idx"),
        "code": code,
        "brand": brand,
        "brand_slug": clean_text(row.get("brand_slug")),
        "description": description,
        "solution": solution,
        "equipment_type": clean_text(row.get("equipment_type")),
        "severity": clean_text(row.get("severity")),
        "source_url": clean_text(row.get("source_url")),
        "domain": "industrial_fault_diagnostics",
        "prompt": prompt,
        "answer": answer,
        "text": text,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def build_examples(rows: Iterable[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        example = build_example(row, dataset)
        if example is not None:
            examples.append(example)
    return examples


def split_examples(
    examples: list[dict[str, Any]],
    *,
    train_size: int,
    eval_size: int,
    seed: int,
    shuffle: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_size < 0 or eval_size < 0:
        raise ValueError("train_size and eval_size must be non-negative")
    needed = train_size + eval_size
    if len(examples) < needed:
        raise ValueError(f"Need {needed} examples, found {len(examples)}")
    selected = list(examples)
    if shuffle:
        random.Random(seed).shuffle(selected)
    eval_rows = selected[:eval_size]
    train_rows = selected[eval_size : eval_size + train_size]
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--backend",
        choices=["datasets", "viewer"],
        default="datasets",
        help="Use the local Hugging Face datasets loader or Dataset Viewer row pagination.",
    )
    parser.add_argument("--source-limit", type=int)
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--eval-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--train-out", type=Path, default=Path("data/fault_codes_train.jsonl"))
    parser.add_argument("--eval-out", type=Path, default=Path("data/fault_codes_eval.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.backend == "viewer":
        rows = fetch_rows(
            args.dataset,
            args.config,
            args.split,
            source_limit=args.source_limit,
        )
    else:
        rows = fetch_rows_from_datasets(
            args.dataset,
            args.config,
            args.split,
            source_limit=args.source_limit,
        )
    examples = build_examples(rows, args.dataset)
    train_rows, eval_rows = split_examples(
        examples,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    summary = {
        "dataset": args.dataset,
        "backend": args.backend,
        **dataset_notice(args.dataset),
        "source_rows": len(rows),
        "examples": len(examples),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_out": str(args.train_out),
        "eval_out": str(args.eval_out),
        "sample": train_rows[0] if train_rows else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)


if __name__ == "__main__":
    main()
