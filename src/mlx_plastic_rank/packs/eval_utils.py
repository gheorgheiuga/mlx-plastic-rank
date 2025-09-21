"""Helper utilities for batch evaluation CLI commands."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


ANGLE_THINK_PATTERN = re.compile(r"<<\s*(/?)\s*think\s*>>", re.IGNORECASE)
THINK_SPAN_PATTERN = re.compile(
    r"<(?P<tag>think|thinking)>(?P<body>.*?)</(?P=tag)>", re.IGNORECASE | re.DOTALL
)


def parse_batch_sizes(raw: str) -> List[int]:
    entries = [chunk.strip() for chunk in raw.replace(";", ",").split(",") if chunk.strip()]
    if not entries:
        raise SystemExit("At least one batch size must be provided")
    sizes: List[int] = []
    for entry in entries:
        try:
            size = int(entry)
        except ValueError as exc:
            raise SystemExit(f"Invalid batch size '{entry}'") from exc
        if size <= 0:
            raise SystemExit(f"Batch size must be positive, got {size}")
        sizes.append(size)
    return sorted(set(sizes))


def parse_thinking_option(raw: str) -> Tuple[str, int | None]:
    value = raw.strip().lower()
    if value == "keep":
        return "keep", None
    if value == "strip":
        return "strip", None
    if value.startswith("cap="):
        try:
            cap = int(value.split("=", 1)[1])
        except ValueError as exc:
            raise SystemExit(f"Invalid thinking cap value in '{raw}'") from exc
        if cap <= 0:
            raise SystemExit("Thinking cap value must be positive")
        return "cap", cap
    raise SystemExit("Thinking option must be one of: keep, strip, cap=N")


def _normalise_think_tags(text: str) -> str:
    return ANGLE_THINK_PATTERN.sub(lambda m: f"<{ '/' if m.group(1) else '' }think>", text)


def apply_thinking_strategy(text: str, mode: str, cap_tokens: int | None) -> str:
    if mode == "keep":
        return text
    normalised = _normalise_think_tags(text)

    def repl(match: re.Match[str]) -> str:
        tag = match.group("tag")
        body = match.group("body")
        if mode == "strip":
            return ""
        if mode == "cap" and cap_tokens is not None:
            tokens = body.split()
            if len(tokens) > cap_tokens:
                tokens = tokens[:cap_tokens]
            truncated = " ".join(tokens)
            return f"<{tag}>{truncated}</{tag}>"
        return match.group(0)

    processed = THINK_SPAN_PATTERN.sub(repl, normalised)
    if mode == "strip":
        processed = re.sub(r"\n{3,}", "\n\n", processed)
        processed = re.sub(r"[ \t]{2,}", " ", processed)
    return processed.strip()


def load_domain_prompts(path: Path, mode: str, cap_tokens: int | None) -> Dict[str, List[str]]:
    prompts: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {lineno}: {exc}") from exc
            text = obj.get("text") or obj.get("prompt")
            if not isinstance(text, str) or not text.strip():
                continue
            domain = obj.get("domain") or "default"
            processed = apply_thinking_strategy(text, mode, cap_tokens)
            prompts.setdefault(domain, []).append(processed)
    if not prompts:
        raise SystemExit(f"No prompts found in {path}")
    return prompts


__all__ = [
    "apply_thinking_strategy",
    "load_domain_prompts",
    "parse_batch_sizes",
    "parse_thinking_option",
]
