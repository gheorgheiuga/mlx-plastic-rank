"""
mlx_plastic_rank package

Exports core primitives:
- RankLayer, PlasticBlock (low-rank, reversible layers)
- PlasticityManager (plasticity triggers and sleep store)
- stable_rank (rank heuristic)
- SVD helpers (factorized_lowrank, svd_lowrank)
"""
from .rank_select import stable_rank, theorem_guided_rank  # noqa: F401
from .lowrank import (
    RankLayer,
    PlasticBlock,
    factorized_lowrank,
    svd_lowrank,
    quantize_factors,
    dequantize_factors,
)  # noqa: F401
from .plasticity_manager import PlasticityManager  # noqa: F401
from .utils import quantise, dequantise, set_seed, get_logger  # noqa: F401

__all__ = [
    "stable_rank",
    "RankLayer",
    "PlasticBlock",
    "factorized_lowrank",
    "svd_lowrank",
    "quantize_factors",
    "dequantize_factors",
    "theorem_guided_rank",
    "PlasticityManager",
    "quantise",
    "dequantise",
    "set_seed",
    "get_logger",
]
