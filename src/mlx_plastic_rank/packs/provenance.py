"""Content identities shared by training, evaluation, and experiment reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def digest_json(value: Any) -> str:
    """Hash a JSON value independently of dictionary order."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()


def content_sha256(path: Path) -> str:
    """Hash a file, or every non-hidden file in a checkpoint directory.

    Directory identities include relative names and all shards, tokenizer files,
    and configuration. Modification times and absolute locations are excluded.
    """
    if path.is_dir():
        files = {
            item.relative_to(path).as_posix(): content_sha256(item)
            for item in sorted(path.rglob("*"))
            if item.is_file() and not any(part.startswith(".") for part in item.relative_to(path).parts)
        }
        if not files:
            raise ValueError(f"Cannot identify an empty checkpoint directory: {path}")
        return digest_json(files)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def pack_identity(pack_dir: Path) -> dict[str, str]:
    """Bind a report to both the adapter weights and their metadata."""
    return {
        "tensors_sha256": content_sha256(pack_dir / "pack.safetensors"),
        "metadata_sha256": content_sha256(pack_dir / "meta.json"),
    }


def tokenizer_sha256(tokenizer: Any) -> str | None:
    """Identify the loaded tokenizer, including runtime template/token options."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        vocabulary = json.loads(backend.to_str())
    elif hasattr(tokenizer, "get_vocab"):
        vocabulary = tokenizer.get_vocab()
    else:
        return None  # Unknown tokenizer interfaces cannot support a proof.
    options = {
        key: getattr(tokenizer, key, None)
        for key in ("chat_template", "special_tokens_map", "padding_side", "truncation_side", "add_bos_token", "add_eos_token")
    }
    return digest_json({"vocabulary": vocabulary, "options": options})


def tokenized_sha256(*arrays: Any) -> str:
    """Identify exactly the tokenized inputs and masks, including row order."""
    return digest_json([
        {"shape": list(array.shape), "dtype": str(array.dtype), "values": array.tolist()}
        for array in arrays
    ])


def resolve_model_checkpoint(reference: str) -> Path:
    """Resolve a local checkpoint or pin a Hub reference before model loading."""
    path = Path(reference).expanduser()
    if path.exists():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        repo_id=reference,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja", "*.py", "*.bin"],
    ))


def dataset_example_keys(path: Path) -> set[str]:
    """Identify exact training content while ignoring per-row source metadata."""
    keys: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON objects in {path}")
            # Retain every supported representation. Full-token training may
            # consume `text` even when the same row also carries prompt fields.
            messages = row.get("messages")
            if isinstance(messages, list) and messages and all(
                isinstance(message, dict) and isinstance(message.get("role"), str)
                and isinstance(message.get("content"), str) for message in messages
            ):
                keys.add(digest_json([[message["role"], message["content"].strip()] for message in messages]))
            prompt = row.get("prompt") or row.get("question")
            answer = row.get("answer") or row.get("response") or row.get("solution")
            if isinstance(prompt, str) and prompt.strip() and isinstance(answer, str) and answer.strip():
                keys.add(digest_json([["user", prompt.strip()], ["assistant", answer.strip()]]))
            if isinstance(row.get("text"), str) and row["text"].strip():
                keys.add(digest_json(row["text"].strip()))
    if not keys:
        raise ValueError(f"No identifiable examples in {path}")
    return keys
