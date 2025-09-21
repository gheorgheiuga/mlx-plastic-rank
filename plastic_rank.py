"""
Legacy wrapper for backward compatibility with tests.

Re-exports key classes and functions from `src/mlx_plastic_rank` and provides
the demo entrypoint under __main__.
"""
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:  # pragma: no cover - canonical imports for tooling
    from mlx_plastic_rank.lowrank import PlasticBlock, RankLayer  # noqa: F401
    from mlx_plastic_rank.plasticity_manager import PlasticityManager  # noqa: F401
    from mlx_plastic_rank.rank_select import stable_rank  # noqa: F401
    from mlx_plastic_rank.utils import set_seed  # noqa: F401
else:  # pragma: no cover - runtime fallback for src-layout imports
    try:
        from mlx_plastic_rank.lowrank import PlasticBlock, RankLayer  # noqa: F401
        from mlx_plastic_rank.plasticity_manager import PlasticityManager  # noqa: F401
        from mlx_plastic_rank.rank_select import stable_rank  # noqa: F401
        from mlx_plastic_rank.utils import set_seed  # noqa: F401
    except ImportError:
        from src.mlx_plastic_rank.lowrank import PlasticBlock, RankLayer  # noqa: F401
        from src.mlx_plastic_rank.plasticity_manager import PlasticityManager  # noqa: F401
        from src.mlx_plastic_rank.rank_select import stable_rank  # noqa: F401
        from src.mlx_plastic_rank.utils import set_seed  # noqa: F401


def _gather_params(model):
    """Collect (layer_ref, name, array) triples for low-rank params."""
    params = []
    for l in model.modules():
        if hasattr(l, "U"):
            params.append((l, "U", l.U))
            params.append((l, "S", l.S))
            params.append((l, "V", l.V))
    return params


def _assign_params(triples, new_arrays):
    for (layer, name, _), arr in zip(triples, new_arrays):
        setattr(layer, name, arr)


if __name__ == "__main__":
    # Minimal MLX-correct param update using value_and_grad on explicit params
    import argparse

    parser = argparse.ArgumentParser(description="Plastic rank demo")
    parser.add_argument("--steps", type=int, default=3, help="Steps/epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--d-model", type=int, default=128, help="Model width")
    args = parser.parse_args()

    set_seed(args.seed)
    d_model = args.d_model
    model = nn.Sequential(PlasticBlock(d_model), PlasticBlock(d_model))
    mgr = PlasticityManager(model)

    x = mx.random.normal((8, 16, d_model))

    def forward_with_params(param_arrays):
        # Assign params, run, compute scalar loss
        triples = _gather_params(model)
        _assign_params(triples, param_arrays)
        y = model(x)
        return y.mean()

    # Initialize a small rank to enable updates
    for l in model.modules():
        if hasattr(l, "add_rank"):
            l.add_rank(2)

    for epoch in range(args.steps):
        triples = _gather_params(model)
        P = [arr for _, _, arr in triples]
        val_and_grad = mx.value_and_grad(forward_with_params)
        loss, grads = val_and_grad(P)
        # SGD step
        new_P = [p - args.lr * g for p, g in zip(P, grads)]
        _assign_params(triples, new_P)
        mgr.step({"val_loss": float(loss)})
        # Report first RankLayer rank and current sleep storage size
        rank_layers = [m for m in model.modules() if hasattr(m, "rank")]
        r0 = rank_layers[0].rank if rank_layers else 0
        print(epoch, "rank0=", r0, "sleep=", mgr.compress_dict_size(), "bytes")
