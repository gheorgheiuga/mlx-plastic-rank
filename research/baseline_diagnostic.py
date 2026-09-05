"""DSN-20260905-05: diagnose stored routed data with a dense training-only fit.

Run from the repository root: uv run --locked python -m research.baseline_diagnostic
--output-dir out/baseline_diagnostic/<new-name>. No tasks or seeds are generated.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DSN = ROOT / "codex/dsn/dsn-20260905-baseline-validity-diagnostic.md"
INPUT_PACKAGE = ROOT / "out/capacity_migration/gradient_agreement_v1/development_20260905"
DEVELOPMENT_RECEIPT = "46e38b4684226981b05e1def0253671752454d7aff79cfdc6d0fa415656a5c6e"
SEEDS = (31, 32, 33, 34, 35)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_input_package(package: Path, expected_receipt: str) -> dict[int, bytes]:
    """Pin the original completed package and retain immutable verified inputs."""
    receipt_bytes = (package / "output-receipt.json").read_bytes()
    if digest(receipt_bytes) != expected_receipt:
        raise ValueError("Retained package receipt identity mismatch")
    receipt = json.loads(receipt_bytes)
    if receipt.get("status") != "completed" or receipt.get("evidence_enabled") is not False:
        raise ValueError("Expected a completed development package")
    inputs = {}
    for name, expected in receipt["sha256"].items():
        path = (package / name).resolve()
        if not path.is_relative_to(package.resolve()):
            raise ValueError("Receipt path escapes the input package")
        data = path.read_bytes()
        if digest(data) != expected:
            raise ValueError(f"Retained artifact identity mismatch: {name}")
        for seed in SEEDS:
            if name == f"inputs/seed_{seed}.npz":
                inputs[seed] = data
    if set(inputs) != set(SEEDS):
        raise ValueError("Missing declared development inputs")
    return inputs


def routed_design(features: np.ndarray, routes: np.ndarray) -> np.ndarray:
    """Concatenate route_s(x) * x in site-major order, using saved routes."""
    x, gates = np.asarray(features, dtype=np.float64), np.asarray(routes, dtype=np.float64)
    if x.ndim != 2 or gates.ndim != 2 or x.shape[0] != gates.shape[0] or min(*x.shape, *gates.shape) <= 0:
        raise ValueError("Features/routes must be nonempty matrices sharing rows")
    if not np.isfinite(x).all() or not np.isfinite(gates).all():
        raise ValueError("Features/routes must be finite")
    return (gates[:, :, None] * x[:, None, :]).reshape(x.shape[0], -1)


def fit_reference(design: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit only the supplied training arrays with the declared float64 cutoff."""
    if design.ndim != 2 or targets.ndim != 2 or design.shape[0] != targets.shape[0]:
        raise ValueError("Fit requires training matrices sharing rows")
    if not np.isfinite(design).all() or not np.isfinite(targets).all():
        raise ValueError("Fit requires finite training arrays")
    u, singulars, vh = np.linalg.svd(design.astype(np.float64), full_matrices=False)
    cutoff = max(design.shape) * np.finfo(np.float64).eps * singulars[0]
    keep = singulars > cutoff
    coefficients = (vh[keep].T / singulars[keep]) @ (u[:, keep].T @ targets.astype(np.float64))
    if not np.isfinite(coefficients).all():
        raise ValueError("Non-finite reference solution")
    return coefficients, {
        "design_shape": list(design.shape), "numerical_rank": int(keep.sum()),
        "singular_values": singulars.tolist(), "cutoff": float(cutoff),
        "condition_number": float(singulars[0] / singulars[-1]) if singulars[-1] > 0 else None,
        "retained_condition_number": float(singulars[0] / singulars[keep][-1]) if keep.any() else None,
    }


def measure(design: np.ndarray, targets: np.ndarray, coefficients: np.ndarray) -> dict[str, float]:
    prediction = design @ coefficients
    mse = float(np.mean((prediction - targets) ** 2))
    zero_mse = float(np.mean(targets ** 2))
    score = 1.0 - mse / max(zero_mse, 1e-12)
    if not np.isfinite([mse, zero_mse, score]).all():
        raise ValueError("Non-finite reference measurement")
    return {"mse": mse, "zero_output_mse": zero_mse, "score": score}


def run_diagnostic(input_package: Path, output: Path, time_limit: float = 300) -> dict[str, Any]:
    """Freeze source before fitting; retain complete or partial diagnostic evidence."""
    if not np.isfinite(time_limit) or not 0 < time_limit <= 300:
        raise ValueError("Time limit must be positive and at most 300 seconds")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sources = [Path(__file__).resolve(), DSN, ROOT / "tests/research/test_baseline_diagnostic.py"]
    sources += [ROOT / name for name in ("research/__init__.py", "pyproject.toml", "uv.lock", ".python-version", "NOTICE.md")]
    frozen = {str(path.relative_to(ROOT)): path.read_bytes() for path in sources}
    for name, data in frozen.items():
        destination = output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    provenance: dict[str, Any] = {
        "generator": "research/baseline_diagnostic.py", "protocol": "DSN-20260905-05",
        "scope": "unconstrained development learnability diagnostic; no controller inference",
        "data_origin": "stored repository-authored synthetic arrays; no external dataset or weights",
        "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "source_sha256": {name: digest(data) for name, data in frozen.items()},
        "input_package": str(input_package.resolve()), "input_receipt_sha256": DEVELOPMENT_RECEIPT,
        "seeds": list(SEEDS), "evidence_seeds_used": [], "time_limit_seconds": time_limit,
        "run_status": "running",
    }
    write_json(output / "provenance.json", provenance)
    write_json(output / "freeze-receipt.json", {
        "status": "diagnostic_source_and_protocol_frozen_before_fitting",
        "source_sha256": provenance["source_sha256"], "evidence_enabled": False,
        "gates": {"reference_train_and_heldout_min": 0.95, "shifted_heldout_max_exclusive": 0.50},
    })
    rows: list[dict[str, Any]] = []

    def check_time() -> None:
        if time.monotonic() - started > time_limit:
            raise TimeoutError("Diagnostic time limit exceeded; partial results retained")

    try:
        inputs = verify_input_package(input_package, DEVELOPMENT_RECEIPT)
        (output / "inputs").mkdir()
        (output / "fits").mkdir()
        provenance["input_sha256"] = {str(seed): digest(data) for seed, data in inputs.items()}
        write_json(output / "provenance.json", provenance)
        with (output / "raw_results.jsonl").open("x") as journal:
            for seed in SEEDS:
                check_time()
                (output / "inputs" / f"seed_{seed}.npz").write_bytes(inputs[seed])
                with np.load(io.BytesIO(inputs[seed]), allow_pickle=False) as archive:
                    for task in ("A", "B"):
                        # Explicit whitelist: never read probe rows, teacher
                        # transforms, router matrices or oracle site labels.
                        mask = np.asarray(archive[f"{task}_mask"])
                        if mask.ndim != 1 or not np.isin(mask, [0, 1]).all() or not mask.any():
                            raise ValueError("Expected a nonempty binary head mask")
                        heads = np.flatnonzero(mask)
                        design = routed_design(archive[f"{task}_train_features"], archive[f"{task}_train_routes"])
                        train_targets = np.asarray(archive[f"{task}_train_targets"], dtype=np.float64)[:, heads]
                        for condition in ("reference", "shifted_pairing"):
                            check_time()
                            fit_targets = train_targets if condition == "reference" else np.roll(train_targets, 1, axis=0)
                            coefficients, solver = fit_reference(design, fit_targets)
                            # Held-out values are accessed only after fitting.
                            heldout_design = routed_design(archive[f"{task}_eval_features"], archive[f"{task}_eval_routes"])
                            heldout_targets = np.asarray(archive[f"{task}_eval_targets"], dtype=np.float64)[:, heads]
                            row = {
                                "seed": seed, "task": task, "condition": condition, "solver": solver,
                                "fit_train": measure(design, fit_targets, coefficients),
                                "train": measure(design, train_targets, coefficients),
                                "heldout": measure(heldout_design, heldout_targets, coefficients),
                            }
                            check_time()
                            fit_path = output / "fits" / f"seed_{seed}_{task}_{condition}.npz"
                            np.savez(fit_path, coefficients=coefficients, selected_heads=heads,
                                     singular_values=np.asarray(solver["singular_values"]))
                            row["fit_sha256"] = digest(fit_path.read_bytes())
                            journal.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                            journal.flush()
                            rows.append(row)
        check_time()
        if not all((ROOT / name).read_bytes() == data for name, data in frozen.items()):
            raise ValueError("Source changed during the diagnostic")
        reference = [row for row in rows if row["condition"] == "reference"]
        negatives = [row for row in rows if row["condition"] == "shifted_pairing"]
        checks = {
            "complete_finite_matrix": len(reference) == len(negatives) == 10,
            "reference_training": all(row["train"]["score"] >= .95 for row in reference),
            "reference_heldout": all(row["heldout"]["score"] >= .95 for row in reference),
            "broken_pairing_fails": all(row["heldout"]["score"] < .50 for row in negatives),
        }
        if all(checks.values()):
            decision = "investigate_factorized_optimization_and_allocation"
        elif not checks["broken_pairing_fails"]:
            decision = "inspect_measurement_or_leakage"
        elif checks["reference_training"]:
            decision = "inspect_coverage_conditioning_and_identifiability"
        else:
            decision = "inspect_design_targets_masks_and_solver"
        summary = {
            "checks": checks, "decision": decision, "rows": rows,
            "runtime_seconds": time.monotonic() - started,
            "scope": provenance["scope"], "controller_admission": False,
            "uncertainty": "All five declared development fixtures; per-seed results, no population inference or confidence intervals.",
        }
        write_json(output / "summary.json", summary)
        provenance["run_status"] = "completed"
        return summary
    except BaseException as exc:
        provenance.update(run_status="incomplete", error_type=type(exc).__name__, error=str(exc))
        write_json(output / "failure.json", {"error": str(exc), "completed_fits": len(rows)})
        raise
    finally:
        provenance["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "provenance.json", provenance)
        write_json(output / "output-receipt.json", {
            "status": provenance["run_status"], "evidence_enabled": False,
            "sha256": {str(path.relative_to(output)): digest(path.read_bytes())
                       for path in sorted(output.rglob("*")) if path.is_file() and path.name != "output-receipt.json"},
        })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-package", type=Path, default=INPUT_PACKAGE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    summary = run_diagnostic(args.input_package, args.output_dir, args.time_limit_seconds)
    print(json.dumps({key: summary[key] for key in ("checks", "decision", "runtime_seconds")}))
    return 0 if all(summary["checks"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
