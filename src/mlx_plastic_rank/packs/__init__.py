"""Skill pack utilities for MLX LoRA adapters."""

__all__ = [
    "LoRAManager",
    "PackApplicationError",
    "PackMetadata",
    "load_pack",
    "save_pack_metadata",
]


def __getattr__(name: str):
    if name in {"LoRAManager", "PackApplicationError"}:
        from .manager import LoRAManager, PackApplicationError

        return {"LoRAManager": LoRAManager, "PackApplicationError": PackApplicationError}[name]
    if name in {"PackMetadata", "load_pack", "save_pack_metadata"}:
        from .io import PackMetadata, load_pack, save_pack_metadata

        return {
            "PackMetadata": PackMetadata,
            "load_pack": load_pack,
            "save_pack_metadata": save_pack_metadata,
        }[name]
    raise AttributeError(f"module 'mlx_plastic_rank.packs' has no attribute '{name}'")
