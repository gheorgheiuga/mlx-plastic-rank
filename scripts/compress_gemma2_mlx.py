import argparse
import json
import struct
from pathlib import Path

import mlx.core as mx
from contextlib import nullcontext
from tqdm import tqdm
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file, save_file

from mlx_plastic_rank.lowrank import svd_lowrank_randomized
from mlx_plastic_rank.rank_select import choose_rank


def _cpu_stream():
    """CPU stream context if available; otherwise a no-op context."""
    return mx.stream(mx.cpu) if hasattr(mx, "stream") else nullcontext()


def _device_stream(device: str):
    if device == "gpu" and hasattr(mx, "gpu"):
        return mx.stream(mx.gpu)
    return _cpu_stream()


def _estimate_bytes(m: int, n: int, k: int) -> int:
    """Rough FP32 working set for rSVD matmuls: Y(m,k) + B(k,n)."""
    return 4 * (m * k + k * n)


def pick_rank(A_np, target_energy: float, strategy: str, eps: float) -> int:
    """Pick rank with MLX, pinning SVD work to CPU.

    Falls back to the "stable" strategy for non-square matrices.
    """
    m, n = A_np.shape
    used_strategy = strategy if (m == n) else "stable"
    with _cpu_stream():
        # Ensure MLX float32 for stable SVD behavior
        A = mx.array(A_np, dtype=mx.float32)
        r, _ = choose_rank(A, target_energy, strategy=used_strategy, eps=eps)
    return int(max(1, min(int(r), min(m, n))))


def mlx_svd_truncate(
    A_np,
    r: int,
    *,
    svd_kind: str,
    oversamples: int,
    iters: int,
    device: str,
) -> object:
    """Return a rank-r approximation of A_np using MLX.

    - "randomized": uses library rSVD with matmuls on `device` and CPU SVD steps.
    - "full": computes a full SVD on the CPU stream and truncates.
    """
    stream = _device_stream(device)
    with stream:
        A = mx.array(A_np, dtype=mx.float32)
        if svd_kind == "randomized":
            A_r = svd_lowrank_randomized(
                A, r, n_oversamples=oversamples, n_iter=iters, device_stream=stream
            )
            # Convert to NumPy via __array__ interface for safetensors IO
            import numpy as np

            return np.array(A_r)
    # Full SVD on CPU
    with _cpu_stream():
        U, S, Vh = mx.linalg.svd(A)
        U_r, S_r, Vh_r = U[:, :r], S[:r], Vh[:r, :]
        A_r = (U_r * S_r[None, :]) @ Vh_r
    import numpy as np
    return np.array(A_r)


def _file_contains_bf16(path: Path) -> bool:
    """Lightweight SafeTensors header scan to detect BF16 dtypes without NumPy.

    Reads the header length (little-endian u64) and parses the JSON header,
    then checks tensor entries' dtype for "BF16".
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    for k, v in header.items():
        if k.startswith("__"):  # skip metadata
            continue
        if isinstance(v, dict) and v.get("dtype", "").upper() == "BF16":
            return True
    return False


def compress_safetensors_file(
    in_path: Path,
    out_path: Path,
    target_energy: float,
    strategy: str,
    eps: float,
    min_dim: int,
    svd_kind: str,
    svd_oversamples: int,
    svd_iters: int,
    device: str,
    gpu_max_bytes: int,
    max_rank: int | None,
):
    # Skip files containing BF16 tensors by copying them as-is (no NumPy involved)
    if _file_contains_bf16(in_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(in_path.read_bytes())
        tqdm.write(f"[SKIP] {in_path.name}: contains BF16; copied as-is")
        return 0, 0
    tensors = load_file(str(in_path))
    out = {}
    changed = 0
    names = list(tensors.keys())
    pbar = tqdm(names, desc=f"{in_path.name}", leave=False)
    for name in pbar:
        arr = tensors[name]
        if arr.ndim == 2 and min(arr.shape) >= min_dim:
            r = pick_rank(arr, target_energy, strategy, eps)
            if max_rank is not None:
                r = min(r, max_rank)
            # device choice per tensor to avoid large GPU allocations
            local_device = device
            if svd_kind == "randomized" and device == "gpu":
                m, n = map(int, arr.shape)
                k = int(min(min(m, n), r + svd_oversamples))
                if _estimate_bytes(m, n, k) > gpu_max_bytes:
                    local_device = "cpu"
            arr_c = mlx_svd_truncate(
                arr,
                r,
                svd_kind=svd_kind,
                oversamples=svd_oversamples,
                iters=svd_iters,
                device=local_device,
            ).astype(arr.dtype, copy=False)
            out[name] = arr_c
            changed += 1
            tag = " dev=cpu" if local_device != device else ""
            pbar.set_postfix_str(f"shape={tuple(arr.shape)} r*={r}{tag}")
        else:
            out[name] = arr
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_path))
    return changed, len(tensors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="google/gemma-2-2b")
    ap.add_argument("--out", default="out/gemma2_mlx_compressed")
    ap.add_argument("--strategy", choices=["stable", "theorem"], default="stable")
    ap.add_argument("--compress", type=float, default=0.30, help="fraction removed; 0.30 keeps 70%% energy")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--min-dim", type=int, default=64, help="only compress matrices with min(shape) >= this")
    ap.add_argument("--svd", choices=["full", "randomized"], default="randomized")
    ap.add_argument("--svd-oversamples", type=int, default=10)
    ap.add_argument("--svd-iters", type=int, default=2)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu", help="matmuls device; SVD runs on CPU")
    ap.add_argument("--gpu-max-bytes", type=int, default=2_000_000_000, help="fallback to CPU if projected rSVD working set exceeds this")
    ap.add_argument("--max-rank", type=int, default=None, help="cap chosen rank across tensors")
    args = ap.parse_args()

    target_energy = 1.0 - args.compress
    src_dir = snapshot_download(
        args.hf,
        allow_patterns=["*.safetensors", "tokenizer.*", "config.json", "generation_config.json"],
    )
    src_dir = Path(src_dir)
    dst_dir = Path(args.out)
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy config/tokenizer assets as-is
    for fname in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        p = src_dir / fname
        if p.exists():
            (dst_dir / fname).write_bytes(p.read_bytes())

    changed_total = 0
    total_params = 0
    files = sorted(src_dir.glob("*.safetensors"))
    for f in tqdm(files, desc="Files", position=0):
        out_f = dst_dir / f.name
        changed, total = compress_safetensors_file(
            f,
            out_f,
            target_energy,
            args.strategy,
            args.eps,
            args.min_dim,
            args.svd,
            args.svd_oversamples,
            args.svd_iters,
            args.device,
            args.gpu_max_bytes,
            args.max_rank,
        )
        changed_total += changed
        total_params += total
        tqdm.write(f"[OK] {f.name}: compressed {changed} of {total} tensors")

    meta = {
        "strategy": args.strategy,
        "target_energy": target_energy,
        "eps": args.eps,
        "min_dim": args.min_dim,
        "svd": args.svd,
        "svd_oversamples": args.svd_oversamples,
        "svd_iters": args.svd_iters,
        "device": args.device,
    }
    (dst_dir / "mlx_plastic_rank_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. Output at {dst_dir}")


if __name__ == "__main__":
    main()
