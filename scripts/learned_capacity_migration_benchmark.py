#!/usr/bin/env python3
"""Run the learned dense-router MLX capacity-migration benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from mlx_plastic_rank.packs.learned_capacity_migration import (
    LearnedMigrationConfig,
    run_learned_capacity_migration,
    write_artifacts,
)


def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "development", "evidence"),
        default="smoke",
    )
    parser.add_argument("--protocol", default="tiny_mlx_dense_v1")
    parser.add_argument("--seeds", type=_seeds)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_learned_capacity_migration(
        LearnedMigrationConfig(
            protocol=args.protocol,
            mode=args.mode,
            seeds=args.seeds,
        )
    )
    output_dir = args.output_dir or (
        Path("out/capacity_migration") / args.protocol / args.mode
    )
    paths = write_artifacts(report, output_dir)
    print(f"verdict={report['evidence_status']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 2 if args.require_pass and not report["gates"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
