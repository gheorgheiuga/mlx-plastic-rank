"""Generate held-out industrial fault-code answers with base and optional pack."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx

from mlx_plastic_rank.packs.manager import LoRAManager

WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "that",
    "the",
    "this",
    "with",
    "your",
}


def load_examples(path: Path, *, limit: int, offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < offset:
                continue
            if limit > 0 and len(rows) >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No examples selected from {path}")
    return rows


def render_chat_prompt(processor: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    raise SystemExit("Loaded processor/tokenizer does not expose apply_chat_template")


def load_vlm(base: str):
    from mlx_vlm.utils import load

    loaded = load(base)
    return loaded[0], loaded[1]


def generate_one(
    model: Any,
    processor: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    chat_template: bool,
) -> tuple[str, dict[str, float | int | None]]:
    from mlx_vlm.generate import stream_generate

    rendered = render_chat_prompt(processor, prompt) if chat_template else prompt
    start = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    last_response: Any | None = None
    for response in stream_generate(
        model,
        processor,
        rendered,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=False,
    ):
        text = getattr(response, "text", "")
        if text and first_token_at is None:
            first_token_at = time.perf_counter()
        pieces.append(text)
        last_response = response
    elapsed = time.perf_counter() - start
    return "".join(pieces).strip(), {
        "generation_time_s": elapsed,
        "first_token_ms": ((first_token_at - start) * 1000.0) if first_token_at else None,
        "prompt_tokens": getattr(last_response, "prompt_tokens", None),
        "generation_tokens": getattr(last_response, "generation_tokens", None),
        "total_tokens": getattr(last_response, "total_tokens", None),
        "generation_tps": getattr(last_response, "generation_tps", None),
    }


def words(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def solution_keywords(solution: str) -> set[str]:
    return {word for word in words(solution) if len(word) >= 4 and word not in STOPWORDS}


def evidence_score(example: dict[str, Any], output: str) -> dict[str, Any]:
    output_words = words(output)
    brand = str(example.get("brand") or "").strip().lower()
    code = str(example.get("code") or "").strip().lower()
    keywords = solution_keywords(str(example.get("solution") or ""))
    hits = sorted(keywords & output_words)
    return {
        "contains_brand": bool(brand and brand in output.lower()),
        "contains_code": bool(code and code in output.lower()),
        "solution_keyword_hits": len(hits),
        "solution_keyword_total": len(keywords),
        "solution_keyword_overlap": (len(hits) / len(keywords)) if keywords else 0.0,
        "matched_solution_keywords": hits,
    }


def summarize(rows: Iterable[dict[str, Any]], prefix: str) -> dict[str, float]:
    collected = list(rows)
    if not collected:
        return {}
    n = float(len(collected))
    return {
        f"{prefix}_contains_brand_rate": sum(row[f"{prefix}_contains_brand"] for row in collected) / n,
        f"{prefix}_contains_code_rate": sum(row[f"{prefix}_contains_code"] for row in collected) / n,
        f"{prefix}_solution_keyword_overlap": sum(
            row[f"{prefix}_solution_keyword_overlap"] for row in collected
        )
        / n,
    }


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(row)
    flattened["base_matched_solution_keywords"] = ",".join(row["base_matched_solution_keywords"])
    if "pack_matched_solution_keywords" in flattened:
        flattened["pack_matched_solution_keywords"] = ",".join(row["pack_matched_solution_keywords"])
    return flattened


def write_outputs(payload: dict[str, Any], out: Path, csv_path: Path | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote generation check JSON to {out}")
    if csv_path is None:
        return
    rows = payload.get("rows") or []
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [flatten_for_csv(row) for row in rows]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote generation check CSV to {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="mlx-community/gemma-4-12B-it-qat-mxfp8")
    parser.add_argument("--eval-data", type=Path, default=Path("data/fault_codes_eval.jsonl"))
    parser.add_argument("--pack", help="Pack name under packs/ to compare against the base")
    parser.add_argument("--pack-root", type=Path, default=Path("packs"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("out/fault_codes_generation_check.json"))
    parser.add_argument("--csv", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    examples = load_examples(args.eval_data, limit=args.limit, offset=args.offset)
    model, processor = load_vlm(args.base)
    rows: list[dict[str, Any]] = []

    mx.reset_peak_memory()
    for index, example in enumerate(examples):
        output, metrics = generate_one(
            model,
            processor,
            str(example["prompt"]),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            chat_template=args.chat_template,
        )
        evidence = evidence_score(example, output)
        rows.append(
            {
                "index": args.offset + index,
                "id": example.get("id"),
                "brand": example.get("brand"),
                "code": example.get("code"),
                "prompt": example.get("prompt"),
                "expected_answer": example.get("answer"),
                "base_output": output,
                **{f"base_{key}": value for key, value in metrics.items()},
                **{f"base_{key}": value for key, value in evidence.items()},
            }
        )

    pack_metadata = None
    if args.pack:
        manager = LoRAManager(model, base_model=args.base)
        pack_metadata = manager.apply_pack(args.pack_root / args.pack).pack_name
        for row, example in zip(rows, examples):
            output, metrics = generate_one(
                model,
                processor,
                str(example["prompt"]),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                chat_template=args.chat_template,
            )
            evidence = evidence_score(example, output)
            row["pack_output"] = output
            row.update({f"pack_{key}": value for key, value in metrics.items()})
            row.update({f"pack_{key}": value for key, value in evidence.items()})

    summary = {
        "base": args.base,
        "pack": pack_metadata,
        "examples": len(rows),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "chat_template": args.chat_template,
        "peak_memory_gb": mx.get_peak_memory() / (1024**3),
        **summarize(rows, "base"),
    }
    if args.pack:
        summary.update(summarize(rows, "pack"))
    write_outputs({"summary": summary, "rows": rows}, args.out, args.csv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
