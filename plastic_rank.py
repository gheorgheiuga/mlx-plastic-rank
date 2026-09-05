"""Bounded low-rank lifecycle demo; no training or controller-quality claim.

The historical class imports remain available for compatibility. Use the
installed ``mlx_plastic_rank`` package for application code.
"""

import argparse
import json

import mlx.core as mx

from mlx_plastic_rank.lowrank import PlasticBlock, RankLayer  # noqa: F401
from mlx_plastic_rank.plasticity_manager import PlasticityManager  # noqa: F401
from mlx_plastic_rank.rank_select import stable_rank  # noqa: F401
from mlx_plastic_rank.utils import set_seed


def run_demo(*, steps: int = 3, seed: int = 42, d_model: int = 32) -> list[dict]:
    """Show four live components, park two, then approximately restore them.

    Errors are measured against the residual itself, so a large frozen backbone
    cannot conceal quantization error. Extra steps hold the restored state.
    """
    if steps < 1 or d_model < 4:
        raise ValueError("steps must be positive and d_model must be at least 4")
    set_seed(seed)
    backbone = mx.eye(d_model)
    layer = RankLayer(backbone)
    layer.add_rank(4)
    # Visible synthetic residuals, not learned task weights.
    layer.S = mx.array([0.1, 0.2, 0.3, 0.4])
    inputs = mx.random.normal((8, d_model))
    base_output = inputs @ backbone.T
    original_residual = layer(inputs) - base_output
    scale = mx.maximum(mx.linalg.norm(original_residual), 1e-12)
    rows = []
    for step in range(steps):
        action = "hold"
        if step == 0:
            action = "initialize"
        elif step == 1:
            layer.prune_to_rank(2)
            action = "park"
        elif step == 2:
            for component in list(layer.sleep_dict):
                layer.wake_rank(component)
            action = "restore"
        output = layer(inputs)
        relative_error = mx.linalg.norm(output - base_output - original_residual) / scale
        base_unchanged = bool(mx.array_equal(layer.W0, backbone).item())
        if not base_unchanged or not bool(mx.all(mx.isfinite(output)).item()):
            raise RuntimeError("rank lifecycle violated the frozen/finite invariant")
        if layer.rank + len(layer.sleep_dict) != 4:
            raise RuntimeError("rank lifecycle lost or created a component")
        rows.append({
            "step": step,
            "action": action,
            "active_rank": layer.rank,
            "dormant_components": len(layer.sleep_dict),
            "residual_relative_error": float(relative_error),
            "base_unchanged": base_unchanged,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=3, help="Lifecycle steps (extra steps hold)")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic input seed")
    parser.add_argument("--d-model", type=int, default=32, help="Layer width (at least 4)")
    args = parser.parse_args()
    if args.steps < 1 or args.d_model < 4:
        parser.error("--steps must be positive and --d-model must be at least 4")
    for row in run_demo(steps=args.steps, seed=args.seed, d_model=args.d_model):
        print(json.dumps(row, allow_nan=False))


if __name__ == "__main__":
    main()
