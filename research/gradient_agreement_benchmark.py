#!/usr/bin/env python3
"""Run the bounded development matrix; preserve partial output and source bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from research.gradient_admission import (
    SPEC_PATH,
    load_spec,
    resolved_seeds,
    run_development,
)

ROOT = Path(__file__).resolve().parents[1]


def source_files() -> list[Path]:
    files = set(ROOT.glob("*.py"))
    for folder in ("src", "scripts", "research"):
        files.update((ROOT / folder).rglob("*.py"))
    files.update((ROOT / "codex/research/gradient-agreement").glob("*"))
    files.update(ROOT / name for name in ("pyproject.toml", "uv.lock", ".python-version"))
    return sorted(p for p in files if p.is_file())


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "development", "evidence"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=1800)
    parser.add_argument("--require-valid", action="store_true")
    args = parser.parse_args(argv)
    # Reject evidence before generating tasks or creating an output directory.
    resolved_seeds(args.mode)
    spec = load_spec()
    if not np.isfinite(args.time_limit_seconds) or not 0 < args.time_limit_seconds <= 1800:
        parser.error("time limit must be positive and at most 1800 seconds")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_bytes = {str(p.relative_to(ROOT)): p.read_bytes() for p in source_files()}
    for name, data in source_bytes.items():
        destination = output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    hashes = {name: digest(data) for name, data in source_bytes.items()}
    provenance = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "parsed_arguments": vars(args) | {"output_dir": str(output)},
        "python": sys.version, "mlx": version("mlx"), "numpy": version("numpy"),
        "platform": platform.platform(), "machine": platform.machine(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "source_sha256": hashes, "protocol_sha256": digest(SPEC_PATH.read_bytes()),
        "generator": "research/gradient_agreement_benchmark.py",
        "data_origin": "repository-authored synthetic fixture; no third-party data or weights",
        "run_status": "running", "evidence_enabled": False,
    }
    write_json(output / "provenance.json", provenance)
    write_json(output / "protocol.json", spec)
    write_json(output / "freeze-receipt.json", {
        "status": "development_snapshot_not_evidence_freeze", "evidence_enabled": False,
        "source_sha256": hashes,
    })
    journals = {name: (output / f"{name}.jsonl").open("x") for name in
                ("trajectory", "events", "failures", "inputs", "preparations", "runs")}

    def emit(kind: str, row: dict[str, Any]) -> None:
        handle = journals[kind]
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        if kind in {"runs", "failures"}:
            print(json.dumps({"kind": kind, "seed": row["seed"],
                              "condition": row["condition"], "error": row.get("error")}), flush=True)

    def save_inputs(seed: int, arrays: dict[str, mx.array]) -> None:
        folder = output / "inputs"
        folder.mkdir(exist_ok=True)
        payload: dict[str, Any] = {k: np.asarray(a) for k, a in arrays.items()}
        np.savez(folder / f"seed_{seed}.npz", **payload)

    try:
        report = run_development(args.mode, emit=emit, save_inputs=save_inputs,
                                 time_limit_seconds=args.time_limit_seconds)
        unchanged = all((ROOT / name).read_bytes() == data for name, data in source_bytes.items())
        report["checks"]["source_unchanged_during_run"] = unchanged
        report["development_valid"] &= unchanged
        if not unchanged:
            report["interpretation"] = "park_before_evidence"
        report["runtime_seconds"] = time.monotonic() - started
        report["mlx_process_peak_bytes"] = mx.get_peak_memory()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
        report["memory_scope"] = "whole process including audits, shadows, source capture and shared A preparation"
        write_json(output / "summary.json", report)
        write_json(output / "paired-intervals.json", {
            "status": "not_computed_for_development", "experimental_unit": "fixture_seed",
        })
        failed = [key for key, value in report["checks"].items() if not value]
        interpretation = (
            "# Gradient-agreement development result\n\n"
            + ("Park before evidence. Failed checks: " + ", ".join(failed) + ".\n"
               if failed else "Development checks passed; an evidence freeze is still required.\n")
            + "\nNo confirmatory inference or controller admission is claimed. "
            "Reserved evidence seeds remain disabled. Full results and partial failures "
            "are preserved in the adjacent JSON and JSONL artifacts.\n"
        )
        (output / "interpretation.md").write_text(interpretation)
        provenance.update(run_status="completed", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        provenance.update(run_status="interrupted", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        for handle in journals.values():
            handle.close()
        write_json(output / "provenance.json", provenance)
        write_json(output / "output-receipt.json", {
            "status": provenance["run_status"], "evidence_enabled": False,
            "sha256": {str(p.relative_to(output)): digest(p.read_bytes())
                       for p in sorted(output.rglob("*")) if p.is_file()
                       and p.name != "output-receipt.json"},
        })
    print(json.dumps({"checks": report["checks"], "development_valid": report["development_valid"],
                      "runtime_seconds": report["runtime_seconds"]}), flush=True)
    return 2 if args.require_valid and not report["development_valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
