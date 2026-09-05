#!/usr/bin/env python3
"""Run the frozen first-opportunity loss-lookahead calibration benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Sequence

from research.loss_lookahead_calibration import (
    LookaheadCalibrationConfig,
    run_loss_lookahead_calibration,
    write_artifacts,
)


def _seeds(value: str) -> tuple[int, ...]:
    if "-" in value and "," not in value:
        start, end = (int(part) for part in value.split("-", 1))
        return tuple(range(start, end + 1))
    return tuple(int(part) for part in value.split(",") if part)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_hash() -> str:
    source = Path("research/loss_lookahead_calibration.py")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("smoke", "development", "evidence"),
        default="smoke",
    )
    parser.add_argument("--seeds", type=_seeds)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = datetime.now(timezone.utc)
    monotonic_start = time.monotonic()
    report = run_loss_lookahead_calibration(
        LookaheadCalibrationConfig(mode=args.mode, seeds=args.seeds)
    )
    completed = datetime.now(timezone.utc)
    output_dir = args.output_dir or (
        Path("out/capacity_migration/loss_lookahead_calibration_v1") / args.mode
    )
    provenance = {
        "command": [sys.executable, *sys.argv],
        "started_utc": started.isoformat(),
        "completed_utc": completed.isoformat(),
        "runtime_seconds": time.monotonic() - monotonic_start,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "source_sha256": _source_hash(),
        "python": sys.version,
        "mlx": version("mlx"),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    paths = write_artifacts(report, output_dir, provenance=provenance)
    print(f"verdict={report['evidence_status']}")
    print(json.dumps(report["gates"], sort_keys=True))
    for name, path in paths.items():
        print(f"{name}={path}")
    return 2 if args.require_pass and not report["gates"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
