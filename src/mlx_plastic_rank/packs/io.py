"""Pack serialization helpers using SafeTensors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
from safetensors.numpy import load_file, save_file

PACK_VERSION = "0.1.0"


@dataclass
class PackMetadata:
    pack_name: str
    base_hash: str
    base_model: str | None = None
    rank_map: Dict[str, int] = field(default_factory=dict)
    alpha_map: Dict[str, float] = field(default_factory=dict)
    target_layers: List[str] = field(default_factory=list)
    created_at: str = ""
    notes: str = ""
    version: str = PACK_VERSION

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PackMetadata":
        base_model_val = data.get("base_model")
        base_model = str(base_model_val) if base_model_val not in (None, "") else None
        rank_data = data.get("rank_map") or {}
        alpha_data = data.get("alpha_map") or {}
        target_data = data.get("target_layers") or []
        if isinstance(rank_data, dict):
            rank_map = {str(k): int(v) for k, v in rank_data.items()}
        else:
            rank_map = {}
        if isinstance(alpha_data, dict):
            alpha_map = {str(k): float(v) for k, v in alpha_data.items()}
        else:
            alpha_map = {}
        if isinstance(target_data, list):
            targets = [str(v) for v in target_data]
        else:
            targets = []
        return cls(
            pack_name=str(data.get("pack_name", "")),
            base_hash=str(data.get("base_hash", "")),
            base_model=base_model,
            rank_map=rank_map,
            alpha_map=alpha_map,
            target_layers=targets,
            created_at=str(data.get("created_at", "")),
            notes=str(data.get("notes", "")),
            version=str(data.get("version", PACK_VERSION)),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "pack_name": self.pack_name,
            "base_hash": self.base_hash,
            "base_model": self.base_model,
            "rank_map": self.rank_map,
            "alpha_map": self.alpha_map,
            "target_layers": self.target_layers,
            "created_at": self.created_at,
            "notes": self.notes,
            "version": self.version,
        }


def compute_sha256(path: Path) -> str:
    target = path
    if path.is_dir():
        safetensors = sorted(path.glob("*.safetensors"))
        if not safetensors:
            raise FileNotFoundError(f"No .safetensors file found under {path}")
        target = safetensors[0]
    if not target.exists():
        raise FileNotFoundError(f"Cannot compute sha256 for missing file: {target}")

    hasher = hashlib.sha256()
    with target.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_pack(tensors: Dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))


def load_pack(tensor_path: Path) -> Dict[str, np.ndarray]:
    return load_file(str(tensor_path))


def save_pack_metadata(metadata: PackMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")


def load_pack_metadata(path: Path) -> PackMetadata:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PackMetadata.from_dict(data)
