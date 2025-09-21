import argparse
import json
import os

import mlx.core as mx
import mlx.nn as nn

from mlx_plastic_rank.lowrank import PlasticBlock
from mlx_plastic_rank.plasticity_manager import PlasticityManager
from mlx_plastic_rank.utils import set_seed

"""
Toy demo script using PlasticBlock.
"""


def read_events(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser(description="Demo script using PlasticBlock modules and PlasticityManager.")
    ap.add_argument("--strategy", choices=["stable", "theorem"], default="stable")
    ap.add_argument("--compress", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--delta", type=float, default=1e-3)
    ap.add_argument("--log", type=str, default="out/plasticity.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    d_model = 128
    model = nn.Sequential(PlasticBlock(d_model), PlasticBlock(d_model))
    mgr = PlasticityManager(
        model,
        delta=args.delta,
        strategy=args.strategy,
        target_compression=1 - args.compress,
        log_path=args.log,
    )

    x = mx.random.normal((4, 8, d_model))

    def loss_fn(x):
        return model(x).mean()

    # Pre-activate some rank for activity
    for l in model.modules():
        if hasattr(l, "add_rank"):
            l.add_rank(2)

    for step in range(args.steps):
        loss = loss_fn(x)
        mgr.step({"val_loss": float(loss)})

    # Print table from JSONL
    rows = read_events(args.log)
    print("Layer      r0   r*   Action    Sleep(bytes)   Residual   Δval")
    for e in rows:
        dv = e["val_loss_after"] - e["val_loss_before"]
        print(
            f"{e['layer']:<8}  {e['r0']:>3}  {e['r_star']:>3}  {e['action']:<7}"
            f"  {e['sleep_bytes']:>12,}B  {e['residual']:>8.2f}  {dv:>+6.3f}"
        )


if __name__ == "__main__":
    main()
