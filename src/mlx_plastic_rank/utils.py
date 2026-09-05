
import mlx.core as mx


def set_seed(seed: int) -> None:
    """Seed MLX RNG for determinism in tests and demos.

    Note: MLX RNG API is global; this sets the default stream.
    """
    try:
        mx.random.seed(seed)
    except Exception:
        # For older MLX versions without mx.random.seed
        pass


def get_logger(name: str = "mlx_plastic_rank"):
    """Minimal stdlib logger configured for concise INFO logs."""
    import logging

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def quantise(x: mx.array, bits: int = 8):
    """Uniformly quantize an array to unsigned integers.

    Returns a tuple (q, min, scale) where q is uint8, and
    dequantization is: q.astype(float32) * scale + min.
    """
    if bits != 8 or isinstance(bits, bool):
        raise ValueError("Only 8-bit quantization is supported")
    if x.size == 0 or not bool(mx.all(mx.isfinite(x)).item()):
        raise ValueError("Quantization requires nonempty finite values")
    x_min, x_max = x.min(), x.max()
    denom = (2 ** bits - 1)
    scale = (x_max - x_min) / denom
    scale = mx.maximum(scale, mx.array(1e-12))
    q = ((x - x_min) / scale).round().astype(mx.uint8)
    return q, float(x_min), float(scale)


def dequantise(q: mx.array, mn: float, scale: float) -> mx.array:
    """Inverse of quantise."""
    return q.astype(mx.float32) * scale + mn
