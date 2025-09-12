import pytest

pytest.importorskip("mlx.core", reason="MLX not installed; skipping")


def test_estimate_qlr_bytes_matches_schema():
    from scripts.bench_memory import estimate_qlr_bytes

    m, n, r = 100, 80, 16
    # Schema math: q bytes + metadata bytes
    expected = (m * r + r + r * n) + 4 * (m + m + 1 + 1 + r + r)
    assert estimate_qlr_bytes(m, n, r, 8) == expected

