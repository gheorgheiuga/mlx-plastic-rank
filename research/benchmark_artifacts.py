"""Shared serialization contract for capacity-migration benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_benchmark_artifacts(
    report: dict[str, Any],
    output_dir: Path,
    *,
    markdown: str,
) -> dict[str, Path]:
    """Write trajectory JSONL and JSON, CSV, and Markdown summaries."""

    aggregate = report.get("aggregate")
    if not aggregate:
        raise ValueError("benchmark report must contain aggregate rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "trajectory_jsonl": output_dir / "trajectory.jsonl",
        "summary_json": output_dir / "summary.json",
        "summary_csv": output_dir / "summary.csv",
        "summary_markdown": output_dir / "summary.md",
    }
    with paths["trajectory_jsonl"].open("w", encoding="utf-8") as handle:
        for row in report["trajectory"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {key: value for key, value in report.items() if key != "trajectory"}
    paths["summary_json"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with paths["summary_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    paths["summary_markdown"].write_text(markdown, encoding="utf-8")
    return paths


__all__ = ["write_benchmark_artifacts"]
