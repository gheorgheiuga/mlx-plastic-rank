"""Export utilities for saving compressed low-rank factors to SafeTensors.

Schema (simple, explicit):
{
  "U.q": uint8[m, r],
  "U.min": float32[m],
  "U.scale": float32[m],
  "S.q": uint8[r],
  "S.min": float32[1],
  "S.scale": float32[1],
  "Vh.q": uint8[r, n],
  "Vh.min": float32[r],
  "Vh.scale": float32[r],
  "__meta__.json": uint8[k]  // utf-8 JSON: {"bits":8,"shape":[m,n],"rank":r,"version":"0.1.0"}
}
"""
from typing import Dict, Tuple, Any
import mlx.core as mx
import json


def to_numpy_tree(tree):
    if isinstance(tree, dict):
        return {k: to_numpy_tree(v) for k, v in tree.items()}
    if isinstance(tree, mx.array):
        # Convert MLX array to NumPy via the __array__ interface
        import numpy as np  # local import to keep NumPy use minimal

        return np.array(tree)
    return tree


def export_safetensors(path: str, weights: Dict[str, mx.array]) -> None:
    """Save a dict of MLX arrays to a .safetensors file.

    Example:
        weights = {"layer0.W0": mx.random.normal((8, 8))}
        export_safetensors("weights.safetensors", weights)
    """
    np_tree = to_numpy_tree(weights)
    from safetensors.numpy import save_file

    save_file(np_tree, path)


def print_gguf_instructions():  # pragma: no cover - guidance text
    print(
        "Convert to GGUF with llama.cpp/convert_hf_to_gguf.py and pass flags\n"
        "to include tokenizer and chat template metadata as per the Grok report."
    )


# ---------- Low-rank packing API ----------
from .lowrank import quantize_factors, dequantize_factors, factorized_lowrank


def pack_lowrank(U: mx.array, S: mx.array, Vh: mx.array, bits: int = 8) -> Dict[str, Any]:
    """Pack quantized low-rank factors into a flat dict matching the schema.

    Returns a dict of numpy arrays ready for safetensors saving.
    """
    m, r = U.shape
    r2, n = Vh.shape
    assert r == r2 == S.shape[0]
    packed = quantize_factors(U, S, Vh, bits=bits)
    (qU, minU, scU) = packed["U"]
    (qS, s_min, s_scale) = packed["S"]
    (qVh, minVh, scVh) = packed["Vh"]
    meta = {"bits": int(bits), "shape": [int(m), int(n)], "rank": int(r), "version": "0.1.0"}
    meta_bytes = json.dumps(meta).encode("utf-8")

    import numpy as np  # local import used only for packaging to safetensors

    return {
        "U.q": np.array(qU),
        "U.min": np.array(minU),
        "U.scale": np.array(scU),
        "S.q": np.array(qS),
        "S.min": np.array([float(s_min)], dtype=np.float32),
        "S.scale": np.array([float(s_scale)], dtype=np.float32),
        "Vh.q": np.array(qVh),
        "Vh.min": np.array(minVh),
        "Vh.scale": np.array(scVh),
        # store raw JSON bytes as uint8 array for portability
        "__meta__.json": np.frombuffer(meta_bytes, dtype=np.uint8),
    }


def save_lowrank(path: str, packed: Dict[str, Any]) -> None:
    from safetensors.numpy import save_file

    save_file(packed, path)


def load_lowrank(path: str) -> Tuple[mx.array, mx.array, mx.array]:
    """Load packed low-rank factors and dequantize to MLX arrays.

    Uses safetensors' NumPy backend for IO only (no NumPy compute in project code).
    """
    from safetensors.numpy import load_file

    tensors = load_file(path)
    qU = mx.array(tensors["U.q"])  # uint8
    minU = mx.array(tensors["U.min"])  # float32
    scU = mx.array(tensors["U.scale"])  # float32
    qS = mx.array(tensors["S.q"])  # uint8
    s_min = float(tensors["S.min"][0])
    s_scale = float(tensors["S.scale"][0])
    qVh = mx.array(tensors["Vh.q"])  # uint8
    minVh = mx.array(tensors["Vh.min"])  # float32
    scVh = mx.array(tensors["Vh.scale"])  # float32

    U, S, Vh = dequantize_factors({
        "U": (qU, minU, scU),
        "S": (qS, mx.array([s_min]), mx.array([s_scale])),
        "Vh": (qVh, minVh, scVh),
    })
    return U, S, Vh


if __name__ == "__main__":  # CLI exporter
    import argparse, os

    parser = argparse.ArgumentParser(description="Export low-rank compressed weights to .safetensors")
    parser.add_argument("--from-weight", required=True, help="Path to weight matrix (.pt)")
    parser.add_argument("--rank", type=int, required=True, help="Target rank r")
    parser.add_argument("--bits", type=int, default=8, help="Quantization bits")
    parser.add_argument("--out", required=True, help="Output .safetensors path")
    args = parser.parse_args()

    src = args.from_weight
    ext = os.path.splitext(src)[1].lower()
    if ext == ".pt":
        try:
            import torch
        except Exception as e:
            raise RuntimeError("PyTorch not available to load .pt file") from e
        obj = torch.load(src, map_location="cpu")
        if hasattr(obj, "numpy"):
            A_np = obj.numpy()
        elif isinstance(obj, dict):
            # pick first tensor-like item
            for v in obj.values():
                if hasattr(v, "detach"):
                    A_np = v.detach().cpu().numpy()
                    break
            else:
                raise RuntimeError("No tensor found in .pt file")
        else:
            raise RuntimeError("Unsupported .pt contents; provide a tensor or state_dict")
    else:
        raise ValueError("Unsupported input format; use .pt")

    A = mx.array(A_np)
    U, S, Vh = factorized_lowrank(A, args.rank)
    packed = pack_lowrank(U, S, Vh, bits=args.bits)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_lowrank(args.out, packed)
    print(f"Saved low-rank factors to {args.out}")
