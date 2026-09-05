import io
import json

import numpy as np
import pytest

from research import baseline_diagnostic as diagnostic


def test_routed_design_uses_saved_routes_in_site_order():
    x = np.array([[1., 2.], [3., 4.]])
    routes = np.array([[.25, .75], [.5, .5]])
    np.testing.assert_array_equal(diagnostic.routed_design(x, routes), [[.25, .5, .75, 1.5], [1.5, 2., 1.5, 2.]])


def test_reference_fits_training_only_and_reports_rank_deficiency():
    design = np.array([[1., 0., 1.], [0., 1., 0.], [1., 1., 1.], [2., 1., 2.]])
    targets = design @ np.array([[1.], [3.], [1.]])
    weights, solver = diagnostic.fit_reference(design, targets)
    assert solver["numerical_rank"] == 2
    np.testing.assert_allclose(design @ weights, targets, atol=1e-12)
    original = weights.copy()
    assert diagnostic.measure(design, targets, weights)["score"] > .999
    assert diagnostic.measure(design, targets + 100, weights)["score"] < .5
    np.testing.assert_array_equal(weights, original)


def _test_package(tmp_path, monkeypatch):
    """Independent unit-test fixtures, unrelated to retained development tasks."""
    package = tmp_path / "test-package"
    (package / "inputs").mkdir(parents=True)
    rng = np.random.default_rng(7)
    receipts = {}
    for seed in diagnostic.SEEDS:
        arrays = {}
        for task in ("A", "B"):
            for split in ("train", "eval"):
                x = rng.normal(size=(64, 2)).astype(np.float32)
                arrays[f"{task}_{split}_features"] = x
                arrays[f"{task}_{split}_routes"] = np.ones((64, 1), dtype=np.float32)
                arrays[f"{task}_{split}_targets"] = (x[:, :1] * 2 + x[:, 1:] * 3).astype(np.float32)
            arrays[f"{task}_mask"] = np.ones(1, dtype=np.float32)
            # Reading or validating these unused arrays would fail. They must
            # remain outside both fit inputs and diagnostic measurements.
            arrays[f"{task}_transform"] = np.array([np.nan])
            arrays[f"{task}_probe_targets"] = np.array([np.nan])
        path = package / "inputs" / f"seed_{seed}.npz"
        np.savez(path, **arrays)
        receipts[str(path.relative_to(package))] = diagnostic.digest(path.read_bytes())
    diagnostic.write_json(package / "output-receipt.json", {
        "status": "completed", "evidence_enabled": False, "sha256": receipts,
    })
    expected = diagnostic.digest((package / "output-receipt.json").read_bytes())
    monkeypatch.setattr(diagnostic, "DEVELOPMENT_RECEIPT", expected)
    return package, expected


def test_runner_freezes_source_and_keeps_fits_bound_to_training(tmp_path, monkeypatch):
    package, _ = _test_package(tmp_path, monkeypatch)
    output = tmp_path / "result"
    actual_fit = diagnostic.fit_reference

    def check_freeze_before_fit(*args):
        frozen = json.loads((output / "freeze-receipt.json").read_text())
        assert frozen["status"] == "diagnostic_source_and_protocol_frozen_before_fitting"
        return actual_fit(*args)

    monkeypatch.setattr(diagnostic, "fit_reference", check_freeze_before_fit)
    summary = diagnostic.run_diagnostic(package, output)
    assert all(summary["checks"].values())
    assert len(summary["rows"]) == 20
    assert summary["controller_admission"] is False
    for row in summary["rows"]:
        with np.load(output / "fits" / f"seed_{row['seed']}_{row['task']}_{row['condition']}.npz") as fitted:
            if row["condition"] == "reference":
                np.testing.assert_allclose(fitted["coefficients"].flatten(), [2, 3], atol=1e-6)
    receipt = json.loads((output / "output-receipt.json").read_text())
    for name, expected in receipt["sha256"].items():
        assert diagnostic.digest((output / name).read_bytes()) == expected
    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostic(package, output)


def test_heldout_changes_cannot_change_fitted_weights(tmp_path, monkeypatch):
    package, _ = _test_package(tmp_path, monkeypatch)
    original = diagnostic.run_diagnostic(package, tmp_path / "original")
    receipt_path = package / "output-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    for seed in diagnostic.SEEDS:
        path = package / "inputs" / f"seed_{seed}.npz"
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for task in ("A", "B"):
            arrays[f"{task}_eval_targets"] = arrays[f"{task}_eval_targets"] + 1000
        np.savez(path, **arrays)
        receipt["sha256"][str(path.relative_to(package))] = diagnostic.digest(path.read_bytes())
    diagnostic.write_json(receipt_path, receipt)
    monkeypatch.setattr(diagnostic, "DEVELOPMENT_RECEIPT", diagnostic.digest(receipt_path.read_bytes()))
    changed = diagnostic.run_diagnostic(package, tmp_path / "changed")
    assert original["checks"]["reference_heldout"] is True
    assert changed["checks"]["reference_heldout"] is False
    for row in original["rows"]:
        name = f"seed_{row['seed']}_{row['task']}_{row['condition']}.npz"
        with np.load(tmp_path / "original/fits" / name) as first, np.load(tmp_path / "changed/fits" / name) as second:
            np.testing.assert_array_equal(first["coefficients"], second["coefficients"])


def test_tampered_input_stops_before_fit_and_preserves_failure(tmp_path, monkeypatch):
    package, expected = _test_package(tmp_path, monkeypatch)
    original = diagnostic.verify_input_package(package, expected)
    assert np.load(io.BytesIO(original[31]))["A_train_features"].shape == (64, 2)
    (package / "inputs/seed_31.npz").write_bytes(b"modified")
    monkeypatch.setattr(diagnostic, "fit_reference", lambda *_: pytest.fail("must not fit unverified inputs"))
    with pytest.raises(ValueError, match="identity mismatch"):
        diagnostic.run_diagnostic(package, tmp_path / "rejected")
    failure = json.loads((tmp_path / "rejected/failure.json").read_text())
    assert failure["completed_fits"] == 0


def test_time_limit_preserves_completed_rows(tmp_path, monkeypatch):
    package, _ = _test_package(tmp_path, monkeypatch)
    output = tmp_path / "timeout"
    elapsed = [0.0]
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: elapsed[0])
    actual_fit = diagnostic.fit_reference
    calls = []

    def slow_fit(*args):
        calls.append(1)
        if len(calls) == 2:
            elapsed[0] = 301.0
        return actual_fit(*args)

    monkeypatch.setattr(diagnostic, "fit_reference", slow_fit)
    with pytest.raises(TimeoutError):
        diagnostic.run_diagnostic(package, output)
    assert len((output / "raw_results.jsonl").read_text().splitlines()) == 1
    assert json.loads((output / "output-receipt.json").read_text())["status"] == "incomplete"
