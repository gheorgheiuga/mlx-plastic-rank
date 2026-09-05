"""Pack serialization helpers using SafeTensors."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# safetensors.numpy requires NumPy arrays at the pack file boundary.
import numpy as np
from safetensors.numpy import load_file, save_file

from .provenance import content_sha256

PACK_VERSION = "0.1.0"


@dataclass
class PackMetadata:
    pack_name: str
    base_hash: str
    base_model: str | None = None
    profile: str = "lite"
    rank_map: Dict[str, int] = field(default_factory=dict)
    alpha_map: Dict[str, float] = field(default_factory=dict)
    target_layers: List[str] = field(default_factory=list)
    training_data: str | None = None
    training_config: Dict[str, object] = field(default_factory=dict)
    created_at: str = ""
    notes: str = ""
    version: str = PACK_VERSION
    base_hash_version: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PackMetadata":
        base_hash_version = data.get("base_hash_version", 0)
        if type(base_hash_version) is not int or base_hash_version not in (0, 1):
            raise ValueError("Unsupported base hash version")
        base_model_val = data.get("base_model")
        base_model = str(base_model_val) if base_model_val not in (None, "") else None
        rank_data = data.get("rank_map") or {}
        alpha_data = data.get("alpha_map") or {}
        target_data = data.get("target_layers") or []
        training_config_data = data.get("training_config") or {}
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
        training_data_val = data.get("training_data")
        training_data = str(training_data_val) if training_data_val not in (None, "") else None
        training_config = training_config_data if isinstance(training_config_data, dict) else {}
        return cls(
            pack_name=str(data.get("pack_name", "")),
            base_hash=str(data.get("base_hash", "")),
            base_model=base_model,
            profile=str(data.get("profile", "lite") or "lite").lower(),
            rank_map=rank_map,
            alpha_map=alpha_map,
            target_layers=targets,
            training_data=training_data,
            training_config=training_config,
            created_at=str(data.get("created_at", "")),
            notes=str(data.get("notes", "")),
            version=str(data.get("version", PACK_VERSION)),
            base_hash_version=base_hash_version,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "pack_name": self.pack_name,
            "base_hash": self.base_hash,
            "base_model": self.base_model,
            "profile": self.profile,
            "rank_map": self.rank_map,
            "alpha_map": self.alpha_map,
            "target_layers": self.target_layers,
            "training_data": self.training_data,
            "training_config": self.training_config,
            "created_at": self.created_at,
            "notes": self.notes,
            "version": self.version,
            "base_hash_version": self.base_hash_version,
        }


def compute_sha256(path: Path) -> str:
    """Hash complete content, including every checkpoint shard and configuration."""
    return content_sha256(path)


def validate_base_identity(
    metadata: PackMetadata, *, checkpoint_hash: str | None, base_model: str | None,
) -> None:
    """Check the full checkpoint identity; legacy hashes are never sufficient."""
    if metadata.base_hash_version not in (0, 1):
        raise ValueError(f"Unsupported base hash version: {metadata.base_hash_version}")
    expected_hash = metadata.base_hash
    if metadata.base_hash_version == 0:
        provenance = metadata.training_config.get("provenance", {})
        expected_hash = provenance.get("model_sha256", "") if isinstance(provenance, dict) else ""
        if not expected_hash:
            raise ValueError("Legacy pack has no whole-checkpoint identity; inspect or recreate it before attachment")
    if expected_hash:
        if not checkpoint_hash:
            raise ValueError("Pack requires a resolved checkpoint for base hash verification")
        if expected_hash != checkpoint_hash:
            raise ValueError("Base hash mismatch: pack and checkpoint content differ")
    elif checkpoint_hash:
        raise ValueError("Pack has no checkpoint identity; recreate it against this checkpoint")
    elif base_model and metadata.base_model and base_model != metadata.base_model:
        raise ValueError(f"Base model mismatch: expected {base_model}, pack built for {metadata.base_model}")


def validate_pack_tensors(tensors: Dict[str, np.ndarray]) -> None:
    """Reject non-real or non-finite values at every pack file boundary."""
    for key, value in tensors.items():
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"Pack tensor {key} must contain finite floating-point values")


def save_pack(tensors: Dict[str, np.ndarray], out_path: Path) -> None:
    validate_pack_tensors(tensors)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))


def load_pack(tensor_path: Path) -> Dict[str, np.ndarray]:
    tensors = load_file(str(tensor_path))
    validate_pack_tensors(tensors)
    return tensors


def save_pack_metadata(metadata: PackMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(metadata.to_dict(), indent=2, allow_nan=False)
    path.write_text(payload, encoding="utf-8")


def load_pack_metadata(path: Path) -> PackMetadata:
    data = json.loads(path.read_text(encoding="utf-8"))
    return PackMetadata.from_dict(data)
