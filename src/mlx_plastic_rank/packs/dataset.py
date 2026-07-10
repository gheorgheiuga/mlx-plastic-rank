"""Dataset utilities for tiny LoRA training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import mlx.core as mx


@dataclass(frozen=True)
class SupervisedDatasetCoverage:
    """Accounting for which supervised source rows and answer tokens are scored."""

    source_rows: int
    included_rows: int
    invalid_rows: int
    answer_outside_sequence_rows: int
    sample_limited_rows: int
    truncated_included_rows: int
    reference_answer_tokens_total: int
    reference_answer_tokens_retained: int

    @property
    def excluded_rows(self) -> int:
        return self.source_rows - self.included_rows

    def as_metrics(self) -> Dict[str, int | float | None]:
        token_retention = (
            self.reference_answer_tokens_retained / self.reference_answer_tokens_total
            if self.reference_answer_tokens_total > 0
            else None
        )
        return {
            "source_rows": self.source_rows,
            "included_rows": self.included_rows,
            "excluded_rows": self.excluded_rows,
            "invalid_rows": self.invalid_rows,
            "answer_outside_sequence_rows": self.answer_outside_sequence_rows,
            "sample_limited_rows": self.sample_limited_rows,
            "truncated_included_rows": self.truncated_included_rows,
            "reference_answer_tokens_total": self.reference_answer_tokens_total,
            "reference_answer_tokens_retained": self.reference_answer_tokens_retained,
            "reference_answer_token_retention": token_retention,
        }


def _chat_template_source(obj: Mapping[str, Any]) -> list[dict[str, str]] | None:
    messages = obj.get("messages")
    if isinstance(messages, list) and messages:
        rendered_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                return None
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                return None
            rendered_messages.append({"role": role, "content": content})
        return rendered_messages

    prompt = obj.get("prompt") or obj.get("question")
    answer = obj.get("answer")
    if isinstance(prompt, str) and prompt.strip() and isinstance(answer, str) and answer.strip():
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    return None


def render_chat_text(tokenizer: Any, obj: Mapping[str, Any]) -> str | None:
    messages = _chat_template_source(obj)
    if messages is None:
        return None
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not expose apply_chat_template")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _prompt_answer(obj: Mapping[str, Any]) -> tuple[str, str] | None:
    messages = _chat_template_source(obj)
    if messages and len(messages) >= 2 and messages[-1]["role"] == "assistant":
        prompt_messages = messages[:-1]
        if prompt_messages:
            prompt = "\n".join(
                f"{message['role']}: {message['content']}" for message in prompt_messages
            )
            return prompt, messages[-1]["content"]

    raw_prompt = obj.get("prompt") or obj.get("question")
    raw_answer = obj.get("answer") or obj.get("response") or obj.get("solution")
    if isinstance(raw_prompt, str) and raw_prompt.strip() and isinstance(raw_answer, str) and raw_answer.strip():
        return raw_prompt.strip(), raw_answer.strip()
    return None


def _render_prompt_and_full(tokenizer: Any, obj: Mapping[str, Any], chat_template: bool) -> tuple[str, str] | None:
    pair = _prompt_answer(obj)
    if pair is None:
        return None
    prompt, answer = pair

    messages = _chat_template_source(obj)
    if chat_template and messages and len(messages) >= 2 and messages[-1]["role"] == "assistant":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not expose apply_chat_template")
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt_text, full_text

    prompt_text = f"{prompt.rstrip()}\n\nAnswer:\n"
    return prompt_text, prompt_text + answer


def load_jsonl_texts(path: Path, tokenizer: Any | None = None, chat_template: bool = False) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = render_chat_text(tokenizer, obj) if chat_template and tokenizer is not None else None
            if text is None:
                text = obj.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    if not texts:
        raise ValueError(f"No text entries found in {path}")
    return texts


def build_token_dataset(
    texts: Sequence[str],
    tokenizer,
    sequence_length: int,
) -> mx.array:
    tokens: List[int] = []
    for text in texts:
        tokens.extend(tokenizer.encode(text))
    if len(tokens) < sequence_length + 1:
        raise ValueError("Not enough tokens to build dataset")
    total = (len(tokens) // sequence_length) * sequence_length
    tokens = tokens[:total]
    arr = mx.array(tokens, dtype=mx.int32)
    return arr.reshape(-1, sequence_length)


def _pad_token_id(tokenizer: Any) -> int:
    token_id = getattr(tokenizer, "pad_token_id", None)
    if token_id is None:
        token_id = getattr(tokenizer, "eos_token_id", None)
    return int(token_id if token_id is not None else 0)


def build_supervised_token_dataset(
    path: Path,
    tokenizer: Any,
    sequence_length: int,
    *,
    chat_template: bool = False,
) -> tuple[mx.array, mx.array]:
    tokens, masks, _ = build_supervised_token_dataset_with_coverage(
        path,
        tokenizer,
        sequence_length,
        chat_template=chat_template,
    )
    return tokens, masks


def build_supervised_token_dataset_with_coverage(
    path: Path,
    tokenizer: Any,
    sequence_length: int,
    *,
    chat_template: bool = False,
    max_samples: int | None = None,
) -> tuple[mx.array, mx.array, SupervisedDatasetCoverage]:
    """Build answer-masked rows and report source-to-evaluation coverage.

    Reference answer token totals include every source row with a renderable
    prompt/answer pair. Retained token counts include only answer tokens that
    remain in rows selected for evaluation after sequence and sample limits.
    """

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")

    token_rows: list[list[int]] = []
    mask_rows: list[list[float]] = []
    pad_id = _pad_token_id(tokenizer)
    source_rows = 0
    invalid_rows = 0
    answer_outside_sequence_rows = 0
    sample_limited_rows = 0
    truncated_included_rows = 0
    reference_answer_tokens_total = 0
    reference_answer_tokens_retained = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            source_rows += 1
            obj = json.loads(line)
            rendered = _render_prompt_and_full(tokenizer, obj, chat_template)
            if rendered is None:
                invalid_rows += 1
                continue
            prompt_text, full_text = rendered
            prompt_tokens = tokenizer.encode(prompt_text)
            full_tokens = tokenizer.encode(full_text)
            if len(full_tokens) < 2:
                invalid_rows += 1
                continue

            answer_start = min(len(prompt_tokens), len(full_tokens))
            reference_answer_tokens_total += max(len(full_tokens) - answer_start, 0)
            tokens = full_tokens[:sequence_length]
            mask = [0.0] * len(tokens)
            for idx in range(answer_start, len(tokens)):
                mask[idx] = 1.0
            retained_answer_tokens = int(sum(mask[1:]))
            if retained_answer_tokens <= 0:
                answer_outside_sequence_rows += 1
                continue
            if max_samples is not None and len(token_rows) >= max_samples:
                sample_limited_rows += 1
                continue

            if len(full_tokens) > sequence_length:
                truncated_included_rows += 1
            reference_answer_tokens_retained += retained_answer_tokens
            if len(tokens) < sequence_length:
                pad_len = sequence_length - len(tokens)
                tokens.extend([pad_id] * pad_len)
                mask.extend([0.0] * pad_len)

            token_rows.append(tokens)
            mask_rows.append(mask)

    if not token_rows:
        raise ValueError(f"No supervised prompt/answer entries found in {path}")
    coverage = SupervisedDatasetCoverage(
        source_rows=source_rows,
        included_rows=len(token_rows),
        invalid_rows=invalid_rows,
        answer_outside_sequence_rows=answer_outside_sequence_rows,
        sample_limited_rows=sample_limited_rows,
        truncated_included_rows=truncated_included_rows,
        reference_answer_tokens_total=reference_answer_tokens_total,
        reference_answer_tokens_retained=reference_answer_tokens_retained,
    )
    return (
        mx.array(token_rows, dtype=mx.int32),
        mx.array(mask_rows, dtype=mx.float32),
        coverage,
    )


def sample_minibatch(
    dataset: mx.array,
    batch_size: int,
) -> mx.array:
    if dataset.shape[0] < batch_size:
        raise ValueError("Batch size exceeds dataset samples")
    indices = mx.random.randint(0, dataset.shape[0], (batch_size,))
    return dataset[indices]


def sample_supervised_minibatch(
    tokens: mx.array,
    masks: mx.array,
    batch_size: int,
) -> tuple[mx.array, mx.array]:
    if tokens.shape[0] < batch_size:
        raise ValueError("Batch size exceeds dataset samples")
    indices = mx.random.randint(0, tokens.shape[0], (batch_size,))
    return tokens[indices], masks[indices]
