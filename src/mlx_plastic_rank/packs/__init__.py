"""Skill pack utilities for MLX LoRA adapters."""

__all__ = [
    "LoRAManager",
    "PackApplicationError",
    "PackMetadata",
    "load_pack",
    "save_pack_metadata",
    "DomainPackRouter",
    "RouteEvent",
    "load_domain_map",
    "resolve_pack_reference",
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
    if name in {
        "DomainPackRouter",
        "RouteEvent",
        "load_domain_map",
        "resolve_pack_reference",
    }:
        from .router import (
            DomainPackRouter,
            RouteEvent,
            load_domain_map,
            resolve_pack_reference,
        )

        return {
            "DomainPackRouter": DomainPackRouter,
            "RouteEvent": RouteEvent,
            "load_domain_map": load_domain_map,
            "resolve_pack_reference": resolve_pack_reference,
        }[name]
    raise AttributeError(f"module 'mlx_plastic_rank.packs' has no attribute '{name}'")
