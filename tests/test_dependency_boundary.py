import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NO_NUMPY_IMPORTS = [
    "src/mlx_plastic_rank/packs/rank_budget.py",
    "src/mlx_plastic_rank/packs/device_profiles.py",
    "src/mlx_plastic_rank/packs/memory_ledger.py",
    "src/mlx_plastic_rank/packs/rank_map.py",
    "src/mlx_plastic_rank/packs/ablation.py",
    "src/mlx_plastic_rank/packs/ledger.py",
    "src/mlx_plastic_rank/packs/rank_ledger.py",
    "src/mlx_plastic_rank/packs/train.py",
    "src/mlx_plastic_rank/packs/eval.py",
    "src/mlx_plastic_rank/packs/inspection.py",
    "src/mlx_plastic_rank/packs/lora.py",
    "src/mlx_plastic_rank/rank_select.py",
]

ALLOWLISTED_NUMPY_IMPORTS = {
    "src/mlx_plastic_rank/packs/io.py": "safetensors.numpy pack file boundary",
    "src/mlx_plastic_rank/packs/manager.py": "safetensors.numpy pack export/import boundary",
    "src/mlx_plastic_rank/pop_polynomial_probe.py": "offline legacy diagnostic with local migration TODO",
}

IMPORT_RE = re.compile(r"^\s*(?:import\s+numpy\b|from\s+numpy\b)", re.MULTILINE)


def _numpy_imports(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [line for line in text.splitlines() if IMPORT_RE.match(line)]


def test_poprank_core_modules_do_not_import_numpy():
    offenders = {}
    for relative in NO_NUMPY_IMPORTS:
        path = ROOT / relative
        if not path.exists():
            continue
        imports = _numpy_imports(path)
        if imports:
            offenders[relative] = imports

    assert offenders == {}


def test_numpy_import_allowlist_has_local_justification():
    for relative, reason in ALLOWLISTED_NUMPY_IMPORTS.items():
        path = ROOT / relative
        assert path.exists(), relative
        assert reason
        assert _numpy_imports(path), f"{relative} is allowlisted but no longer imports NumPy"
