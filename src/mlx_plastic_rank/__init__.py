"""Small, lazy low-rank API.

Legacy manager, theorem-named and vault aliases remain explicitly importable
for compatibility, but are excluded from the advertised core surface.
See DSN-20260905-04 for the active prototype boundary.
"""

__all__ = [
    "stable_rank",
    "RankLayer",
    "PlasticBlock",
    "factorized_lowrank",
    "svd_lowrank",
    "quantize_factors",
    "dequantize_factors",
    "gram_energy_rank",
    "quantise",
    "dequantise",
    "set_seed",
    "get_logger",
]


def __getattr__(name: str):
    if name in {
        "RankLayer",
        "PlasticBlock",
        "factorized_lowrank",
        "svd_lowrank",
        "quantize_factors",
        "dequantize_factors",
    }:
        from .lowrank import (
            PlasticBlock,
            RankLayer,
            dequantize_factors,
            factorized_lowrank,
            quantize_factors,
            svd_lowrank,
        )

        return {
            "RankLayer": RankLayer,
            "PlasticBlock": PlasticBlock,
            "factorized_lowrank": factorized_lowrank,
            "svd_lowrank": svd_lowrank,
            "quantize_factors": quantize_factors,
            "dequantize_factors": dequantize_factors,
        }[name]
    if name in {"stable_rank", "gram_energy_rank", "theorem_guided_rank"}:
        from .rank_select import gram_energy_rank, stable_rank, theorem_guided_rank

        return {
            "stable_rank": stable_rank,
            "gram_energy_rank": gram_energy_rank,
            "theorem_guided_rank": theorem_guided_rank,
        }[name]
    if name in {"quantise", "dequantise", "set_seed", "get_logger"}:
        from .utils import dequantise, get_logger, quantise, set_seed

        return {
            "quantise": quantise,
            "dequantise": dequantise,
            "set_seed": set_seed,
            "get_logger": get_logger,
        }[name]
    if name == "PlasticityManager":
        from .plasticity_manager import PlasticityManager

        return PlasticityManager
    if name in {
        "DeletionCertificate",
        "ForgetPolicy",
        "ForgettingVault",
        "MemoryRecord",
        "VerificationReport",
        "verify_certificate_chain",
    }:
        from .forgetting_vault import (
            DeletionCertificate,
            ForgetPolicy,
            ForgettingVault,
            MemoryRecord,
            VerificationReport,
            verify_certificate_chain,
        )

        return {
            "DeletionCertificate": DeletionCertificate,
            "ForgetPolicy": ForgetPolicy,
            "ForgettingVault": ForgettingVault,
            "MemoryRecord": MemoryRecord,
            "VerificationReport": VerificationReport,
            "verify_certificate_chain": verify_certificate_chain,
        }[name]
    raise AttributeError(f"module 'mlx_plastic_rank' has no attribute '{name}'")
