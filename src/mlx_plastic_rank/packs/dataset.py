"""Dataset utilities for tiny LoRA training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import mlx.core as mx


def load_jsonl_texts(path: Path) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
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


def sample_minibatch(
    dataset: mx.array,
    batch_size: int,
) -> mx.array:
    if dataset.shape[0] < batch_size:
        raise ValueError("Batch size exceeds dataset samples")
    indices = mx.random.randint(0, dataset.shape[0], (batch_size,))
    return dataset[indices]
