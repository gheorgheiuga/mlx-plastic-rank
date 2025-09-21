"""Pack inspection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .io import PackMetadata, load_pack, load_pack_metadata

MAX_PACK_BYTES = 10 * 1024 * 1024
ALLOWED_RANKS = (2, 4, 8)

_BASE_NAME_NORMALISATION = {
    "qwen3-4b-thinking-2507-mlx-4bit": "qwen3-4b-2507-mlx-4bit",
}

_BASE_SIZE_LIMITS = {
    ("qwen3-4b-2507-mlx-4bit", 2): 4 * 1024 * 1024,
    ("qwen3-4b-2507-mlx-4bit", 4): 6 * 1024 * 1024,
    ("qwen3-4b-2507-mlx-4bit", 8): 12 * 1024 * 1024,
    ("llama-3-8b-instruct-mlx-4bit", 4): 9 * 1024 * 1024,
    ("llama-3-8b-instruct-mlx-4bit", 8): 18 * 1024 * 1024,
}


@dataclass
class TensorInfo:
    key: str
    shape: tuple[int, ...]
    dtype: str
    params: int
    bytes: int


def canonical_base_name(base_model: str | None) -> str | None:
    if not base_model:
        return None
    name = Path(base_model).name
    return _BASE_NAME_NORMALISATION.get(name, name)


def size_limit_for(metadata: PackMetadata) -> int:
    base_name = canonical_base_name(metadata.base_model)
    ranks = {int(v) for v in metadata.rank_map.values() if v is not None}
    if base_name and ranks:
        max_rank = max(ranks)
        limit = _BASE_SIZE_LIMITS.get((base_name, max_rank))
        if limit is not None:
            return limit
    return MAX_PACK_BYTES


def summarize_pack(pack_dir: Path) -> tuple[PackMetadata, List[TensorInfo], int, int, List[str]]:
    tensor_path = pack_dir / "pack.safetensors"
    meta_path = pack_dir / "meta.json"
    if not tensor_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Pack files missing in {pack_dir}")

    metadata = load_pack_metadata(meta_path)
    tensors = load_pack(tensor_path)

    infos: List[TensorInfo] = []
    non_lora: List[str] = []
    total_params = 0
    total_bytes = 0
    for key, arr in tensors.items():
        if ".lora." not in key:
            non_lora.append(key)
        params = int(np.prod(arr.shape))
        bytes_ = arr.nbytes
        infos.append(
            TensorInfo(
                key=key,
                shape=tuple(int(v) for v in arr.shape),
                dtype=str(arr.dtype),
                params=params,
                bytes=bytes_,
            )
        )
        total_params += params
        total_bytes += bytes_
    return metadata, infos, total_params, total_bytes, non_lora
