#!/usr/bin/env python3
"""Run the deterministic A -> B -> A capacity-migration benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from research.capacity_migration import (
    BenchmarkConfig,
    run_benchmark,
    write_artifacts,
)


def _seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("out/capacity_migration"))
    parser.add_argument("--seeds", type=_seeds, default=tuple(range(10)))
    parser.add_argument("--task-rank", type=int, default=2)
    parser.add_argument("--phase-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        seeds=args.seeds,
        task_rank=args.task_rank,
        phase_steps=args.phase_steps,
        learning_rate=args.learning_rate,
    )
    report = run_benchmark(config)
    paths = write_artifacts(report, args.output_dir)
    print(f"verdict={report['evidence_status']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 2 if args.require_pass and not report["gates"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
