"""Convert IndustryBench rows into pack-ready JSONL."""

from __future__ import annotations

import argparse
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

DATASET_ID = "alibaba-multimodal-industrial-ai/IndustryBench"
DATASET_SERVER = "https://datasets-server.huggingface.co"
LANGUAGE_FIELDS = {
    "zh": ("question", "answer"),
    "en": ("question_en", "answer_en"),
    "ru": ("question_ru", "answer_ru"),
    "vi": ("question_vi", "answer_vi"),
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def dataset_viewer_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{DATASET_SERVER}/{endpoint}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "mlx-plastic-rank/industrybench"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


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


def selected_languages(raw: str) -> tuple[str, ...]:
    if raw == "all":
        return tuple(LANGUAGE_FIELDS)
    return (raw,)


def truncate_context(text: str, char_limit: int) -> str:
    if char_limit <= 0 or len(text) <= char_limit:
        return text
    return text[:char_limit].rstrip() + "\n[truncated]"


def build_prompt(
    row: dict[str, Any],
    question: str,
    *,
    include_knowledge: bool,
    knowledge_char_limit: int,
    metadata_mode: str,
) -> str:
    parts: list[str] = []
    if metadata_mode == "full":
        parts.extend(
            [
                f"Domain: {clean_text(row.get('domain')) or 'industrial'}",
                f"Industry: {clean_text(row.get('industry_primary')) or 'unknown'}",
                f"Capability: {clean_text(row.get('capability')) or 'unknown'}",
                f"Difficulty: {clean_text(row.get('difficulty')) or 'unknown'}",
                f"Format: {clean_text(row.get('_format')) or 'qa'}",
            ]
        )
    elif metadata_mode == "minimal":
        parts.extend(
            [
                "Task: industrial question answering",
                f"Difficulty: {clean_text(row.get('difficulty')) or 'unknown'}",
            ]
        )

    knowledge = truncate_context(clean_text(row.get("knowledge_text")), knowledge_char_limit)
    if include_knowledge and knowledge:
        parts.extend(["", "Context:", knowledge])
    parts.extend(["", "Question:", question])
    return "\n".join(parts).strip()


def build_example(
    row: dict[str, Any],
    language: str,
    *,
    include_knowledge: bool,
    knowledge_char_limit: int,
    metadata_mode: str,
    dataset: str,
) -> dict[str, Any] | None:
    question_key, answer_key = LANGUAGE_FIELDS[language]
    question = clean_text(row.get(question_key))
    answer = clean_text(row.get(answer_key))
    if not question or not answer:
        return None

    prompt = build_prompt(
        row,
        question,
        include_knowledge=include_knowledge,
        knowledge_char_limit=knowledge_char_limit,
        metadata_mode=metadata_mode,
    )
    text = f"{prompt}\n\nAnswer:\n{answer}"
    source_id = clean_text(row.get("id")) or str(row.get("_row_idx", "unknown"))
    return {
        "id": f"{source_id}:{language}",
        "source_dataset": dataset,
        "source_row_idx": row.get("_row_idx"),
        "source_id": source_id,
        "language": language,
        "domain": clean_text(row.get("domain")) or "industrial",
        "industry_primary": clean_text(row.get("industry_primary")),
        "capability": clean_text(row.get("capability")),
        "difficulty": clean_text(row.get("difficulty")),
        "format": clean_text(row.get("_format")),
        "question": question,
        "answer": answer,
        "prompt": prompt,
        "text": text,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def build_examples(
    rows: Iterable[dict[str, Any]],
    languages: Iterable[str],
    *,
    include_knowledge: bool,
    knowledge_char_limit: int,
    metadata_mode: str,
    dataset: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        for language in languages:
            example = build_example(
                row,
                language,
                include_knowledge=include_knowledge,
                knowledge_char_limit=knowledge_char_limit,
                metadata_mode=metadata_mode,
                dataset=dataset,
            )
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


def default_output(prefix: str, language: str, split: str) -> Path:
    return Path("data") / f"{prefix}_{language}_{split}.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--language", choices=[*LANGUAGE_FIELDS, "all"], default="en")
    parser.add_argument("--source-limit", type=int)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--metadata-mode",
        choices=["none", "minimal", "full"],
        default="full",
        help="Metadata rendered into prompt text; JSON metadata fields are always preserved.",
    )
    parser.add_argument("--include-knowledge", action="store_true")
    parser.add_argument("--knowledge-char-limit", type=int, default=2000)
    parser.add_argument("--train-out", type=Path)
    parser.add_argument("--eval-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    languages = selected_languages(args.language)
    rows = fetch_rows(
        args.dataset,
        args.config,
        args.split,
        source_limit=args.source_limit,
    )
    examples = build_examples(
        rows,
        languages,
        include_knowledge=args.include_knowledge,
        knowledge_char_limit=args.knowledge_char_limit,
        metadata_mode=args.metadata_mode,
        dataset=args.dataset,
    )
    train_rows, eval_rows = split_examples(
        examples,
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    suffix = args.language
    train_out = args.train_out or default_output("industrybench", suffix, "train")
    eval_out = args.eval_out or default_output("industrybench", suffix, "eval")
    summary = {
        "dataset": args.dataset,
        "source_rows": len(rows),
        "examples": len(examples),
        "languages": list(languages),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "include_knowledge": args.include_knowledge,
        "metadata_mode": args.metadata_mode,
        "train_out": str(train_out),
        "eval_out": str(eval_out),
        "sample": train_rows[0] if train_rows else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)


if __name__ == "__main__":
    main()
