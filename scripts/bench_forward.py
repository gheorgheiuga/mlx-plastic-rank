"""Measure one forward implementation per fresh process on synthetic tensors.

Compare --implementation previous/current with identical arguments. The previous
implementation is loaded from the review revision; no checkpoint is downloaded.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_plastic_rank.lowrank import RankLayer
from mlx_plastic_rank.packs.lora import SliceLoRA

ROOT = Path(__file__).resolve().parents[1]
REVIEW_REVISION = "1c308ec"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("layer", "gated"), required=True)
    parser.add_argument("--implementation", choices=("previous", "current"), required=True)
    parser.add_argument("--tokens", type=int, choices=(1, 64), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; use a fresh case path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    relative = "src/mlx_plastic_rank/lowrank.py" if args.kind == "layer" else "src/mlx_plastic_rank/packs/lora.py"
    previous = subprocess.check_output(["git", "show", f"{REVIEW_REVISION}:{relative}"], cwd=ROOT)
    sources = args.out.parent / "sources"
    sources.mkdir(exist_ok=True)
    baseline_path = sources / f"previous_{args.kind}.py"
    baseline_path.write_bytes(previous)
    current_source = (ROOT / relative).read_bytes()
    (sources / f"current_{args.kind}.py").write_bytes(current_source)
    (sources / "bench_forward.py").write_bytes(Path(__file__).read_bytes())
    name = "mlx_plastic_rank._forward_baseline" if args.kind == "layer" else "mlx_plastic_rank.packs._forward_baseline"
    spec = importlib.util.spec_from_file_location(name, baseline_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    mx.random.seed(42)
    width, physical, active = 1024, 32, 8
    x = mx.random.normal((args.tokens, width))
    if args.kind == "layer":
        layer = RankLayer(mx.random.normal((width, width)) / width**.5)
        layer.add_rank(physical)
        layer.S = mx.ones(physical)
        prior = lambda: module.RankLayer.__call__(layer, x)
        current = lambda: layer(x)
        arrays = [x, layer.W0, layer.U, layer.S, layer.V]
    else:
        adapter = SliceLoRA("bench", 0, width, physical, 64,
                            (mx.random.normal((width, physical)) / physical**.5).astype(mx.float16),
                            (mx.random.normal((physical, width)) / width**.5).astype(mx.float16), width, width)
        adapter.set_active_rank(active)
        assert adapter.gates is not None
        prior = lambda: module.SliceLoRA.delta(adapter, x)
        current = lambda: adapter.delta(x)
        arrays = [x, adapter.A, adapter.B, adapter.gates]
    mx.eval(*arrays)
    input_hash = hashlib.sha256(b"".join(np.asarray(value).tobytes() for value in arrays)).hexdigest()
    left, right = prior(), current()
    relative_error = float(mx.linalg.norm(left - right) / mx.maximum(mx.linalg.norm(left), 1e-12))
    if relative_error > 1e-5:
        raise ValueError(f"Forward equivalence failed: {relative_error}")
    del left, right
    operation = prior if args.implementation == "previous" else current
    for _ in range(3):
        mx.eval(operation())
    gc.collect()
    mx.clear_cache()
    resident = mx.get_active_memory()
    mx.reset_peak_memory()
    timings = []
    for _ in range(30):
        started = time.perf_counter()
        mx.eval(operation())
        timings.append((time.perf_counter() - started) * 1000)
    payload = {
        "kind": args.kind, "implementation": args.implementation, "tokens": args.tokens,
        "width": width, "physical_rank": physical, "active_rank": active if args.kind == "gated" else physical,
        "median_ms": statistics.median(timings), "timings_ms": timings,
        "mlx_resident_before_bytes": resident, "mlx_peak_increment_bytes": mx.get_peak_memory() - resident,
        "input_sha256": input_hash, "relative_output_error": relative_error,
        "source_sha256": hashlib.sha256(current_source if args.implementation == "current" else previous).hexdigest(),
        "baseline_revision": REVIEW_REVISION, "mlx": version("mlx"), "python": sys.version,
        "platform": platform.platform(), "command": sys.argv,
        "scope": "one synthetic forward; allocator increment excludes resident tensors and is not whole-model memory",
        "data_origin": "repository-generated synthetic tensors; seed 42; no external data or model",
    }
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: payload[key] for key in ("kind", "implementation", "tokens", "median_ms", "mlx_peak_increment_bytes")}))


if __name__ == "__main__":
    main()
