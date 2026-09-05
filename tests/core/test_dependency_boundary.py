import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NO_NUMPY_IMPORTS = [
    "src/mlx_plastic_rank/factorization.py",
    "src/mlx_plastic_rank/lowrank.py",
    "src/mlx_plastic_rank/packs/rank_budget.py",
    "src/mlx_plastic_rank/packs/device_profiles.py",
    "src/mlx_plastic_rank/packs/rank_map.py",
    "src/mlx_plastic_rank/packs/ablation.py",
    "src/mlx_plastic_rank/packs/rank_ledger.py",
    "src/mlx_plastic_rank/packs/train.py",
    "src/mlx_plastic_rank/packs/provenance.py",
    "src/mlx_plastic_rank/packs/inspection.py",
    "src/mlx_plastic_rank/packs/lora.py",
    "src/mlx_plastic_rank/rank_select.py",
]

ALLOWLISTED_NUMPY_IMPORTS = {
    "src/mlx_plastic_rank/packs/io.py": "safetensors.numpy pack file boundary",
    "src/mlx_plastic_rank/packs/manager.py": "safetensors.numpy pack export/import boundary",
    "src/mlx_plastic_rank/pop_polynomial_probe.py": "offline legacy diagnostic with local migration TODO",
}


def _numpy_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return [name for name in modules if name == "numpy" or name.startswith("numpy.")]


def test_poprank_core_modules_do_not_import_numpy():
    offenders = {}
    for relative in NO_NUMPY_IMPORTS:
        path = ROOT / relative
        assert path.is_file(), f"Dependency boundary references a missing module: {relative}"
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


def test_core_imports_do_not_load_optional_loaders_or_parked_experiments():
    # Isolate imports from other tests that deliberately exercise these modules.
    result = subprocess.run(
        [sys.executable, "-c", """
import builtins
import sys
optional = {'datasets', 'sympy', 'mlx_lm', 'mlx_vlm', 'mlx_audio'}
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    assert name.split('.')[0] not in optional, name
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
from mlx_plastic_rank import *
from mlx_plastic_rank.packs.cli import build_parser
build_parser().parse_args(['list'])
for name in sys.modules:
    assert name.split('.')[0] not in optional, name
    assert not name.startswith(('mlx_plastic_rank.forgetting_vault',
                               'mlx_plastic_rank.packs.gradient_',
                               'mlx_plastic_rank.packs.learned_capacity_migration',
                               'mlx_plastic_rank.packs.multibatch_controller')), name
"""],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
