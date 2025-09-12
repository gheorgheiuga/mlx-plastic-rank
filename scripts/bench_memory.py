import argparse
import mlx.core as mx

from mlx_plastic_rank.lowrank import factorized_lowrank
from mlx_plastic_rank.export_safetensors import pack_lowrank


def estimate_dense_bytes(m: int, n: int) -> int:
    return 4 * m * n


def estimate_qlr_bytes(m: int, n: int, r: int, bits: int = 8) -> int:
    # Quantized: U.q (m*r), S.q (r), Vh.q (r*n) bytes (uint8)
    q = m * r + r + r * n
    # Metadata (mins/scales): U(m)+U(m)+S(1)+S(1)+Vh(r)+Vh(r) float32
    meta = 4 * (m + m + 1 + 1 + r + r)
    return q + meta


def run_once(m: int, n: int, r: int) -> tuple[int, int, float]:
    A = mx.random.normal((m, n))
    U, S, Vh = factorized_lowrank(A, r)
    A_r = (U * S[None, :]) @ Vh
    packed = pack_lowrank(U, S, Vh, bits=8)
    # Error measured between A and its rank-r approximation
    rel = float(mx.linalg.norm(A - A_r) / mx.linalg.norm(A))
    dense = estimate_dense_bytes(m, n)
    qlr = estimate_qlr_bytes(m, n, r, 8)
    return dense, qlr, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--ranks", type=int, nargs="*", default=[4, 8, 16, 32])
    args = ap.parse_args()

    print(f"m={args.m} n={args.n}")
    print("r    dense_bytes    qlr_bytes    savings    rel_error")
    for r in args.ranks:
        dense, qlr, rel = run_once(args.m, args.n, r)
        sav = 100.0 * (1 - qlr / dense)
        print(f"{r:<4} {dense:>12,}  {qlr:>12,}   {sav:>5.1f}%     {rel:>6.3f}")


if __name__ == "__main__":
    main()
