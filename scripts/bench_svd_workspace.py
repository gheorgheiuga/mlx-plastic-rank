"""Measure one synthetic SVD in a fresh process, including CPU peak RSS.

Use a separate invocation for each case. ``--implementation-file`` can load an
explicitly saved local version of lowrank.py for before/after comparisons.
No dataset or model checkpoint is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import resource
import sys
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--oversamples", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--implementation-file", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np

    if args.implementation_file:
        spec = importlib.util.spec_from_file_location(
            "mlx_plastic_rank._svd_benchmark", args.implementation_file.resolve(),
        )
        if spec is None or spec.loader is None:
            parser.error("Cannot load the supplied implementation file")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        factorize = module.randomized_svd
        source_path = args.implementation_file.resolve()
    else:
        from mlx_plastic_rank.lowrank import randomized_svd

        factorize = randomized_svd
        source_file = importlib.import_module(factorize.__module__).__file__
        if source_file is None:
            raise RuntimeError("Cannot identify the factorization implementation")
        source_path = Path(source_file)

    mx.random.seed(args.seed)
    with mx.stream(mx.Device(mx.cpu)):
        matrix = mx.random.normal((args.m, args.n))
        mx.eval(matrix)
    input_sha = hashlib.sha256(np.asarray(matrix).tobytes()).hexdigest()
    rss_before = _peak_rss_bytes()
    mx.reset_peak_memory()
    original_svd = mx.linalg.svd
    decompositions = []

    def measured_svd(
        a: mx.array, compute_uv: bool = True, *, stream: mx.Stream | mx.Device | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        outputs = original_svd(a, compute_uv, stream=stream)
        decompositions.append({
            "input_shape": list(a.shape),
            "output_shapes": [list(output.shape) for output in outputs],
            "output_bytes": sum(output.nbytes for output in outputs),
        })
        return outputs

    mx.linalg.svd = measured_svd
    started = time.perf_counter()
    try:
        u, s, vh = factorize(
            matrix, args.rank, p=args.oversamples, q=args.power_iterations,
            device_stream=mx.stream(mx.Device(mx.cpu if args.device == "cpu" else mx.gpu)),
        )
        mx.eval(u, s, vh)
    finally:
        mx.linalg.svd = original_svd
    elapsed = time.perf_counter() - started
    peak_rss = _peak_rss_bytes()
    mlx_peak = mx.get_peak_memory()
    # Reconstruction is outside the factorization time/memory measurement.
    with mx.stream(mx.Device(mx.cpu)):
        relative_error = float(mx.linalg.norm(matrix - (u * s) @ vh) / mx.linalg.norm(matrix))
    root = Path(__file__).resolve().parent.parent
    report = {
        "kind": "synthetic_svd_workspace", "version": 1,
        "shape": [args.m, args.n], "rank": args.rank, "seed": args.seed,
        "oversamples": args.oversamples, "power_iterations": args.power_iterations,
        "projection_device": args.device, "seconds": elapsed,
        "process_peak_rss_bytes": peak_rss, "rss_high_water_before_bytes": rss_before,
        "rss_high_water_increase_bytes": max(0, peak_rss - rss_before),
        "mlx_peak_bytes": mlx_peak, "relative_frobenius_error": relative_error,
        "svd_decompositions": decompositions,
        "provenance": {
            "fixture": "locally generated Gaussian matrix; no external data or model",
            "input_sha256": input_sha, "implementation_file": str(source_path),
            "implementation_sha256": _sha256(source_path),
            "benchmark_sha256": _sha256(Path(__file__)), "uv_lock_sha256": _sha256(root / "uv.lock"),
            "python": platform.python_version(), "mlx": importlib.metadata.version("mlx"),
            "numpy": np.__version__, "platform": platform.platform(),
        },
    }
    encoded = json.dumps(report, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
