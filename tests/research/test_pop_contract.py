"""Small exact references for the Pop contract; no selector-quality claim."""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from research.pop_polynomial_probe import (
    PolynomialPair,
    evaluate_pair,
    evaluate_polynomial,
    numerical_rank,
)


def exact_rank(matrix):
    """Rational row reduction, independent of the probe's floating-point SVD."""
    rows = [[Fraction(value) for value in row] for row in matrix]
    pivot = 0
    for column in range(len(rows[0])):
        candidate = next((i for i in range(pivot, len(rows)) if rows[i][column]), None)
        if candidate is None:
            continue
        rows[pivot], rows[candidate] = rows[candidate], rows[pivot]
        divisor = rows[pivot][column]
        rows[pivot] = [value / divisor for value in rows[pivot]]
        for i in range(len(rows)):
            if i != pivot:
                multiplier = rows[i][column]
                rows[i] = [a - multiplier * b for a, b in zip(rows[i], rows[pivot])]
        pivot += 1
    return pivot


@pytest.mark.parametrize("matrix,pair,expected", [
    (np.diag([1, 1, 2, 3]), PolynomialPair("repeated", (0, 1), (-1, 1), (1,), (0, -1, 1), "", (1,)), (4, 2, 4, 2)),
    (np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]]), PolynomialPair("jordan", (-1, 1), (1, -2, 1), (-1, 1), (1, -2, 1), ""), (2, 1, 2, 1)),
    (np.zeros((2, 2)), PolynomialPair("zero", (0, 1), (0, 0, 1), (0, 1), (0, 0, 1), ""), (0, 0, 0, 0)),
    (np.diag([2, 3]), PolynomialPair("generic", (0, 1), (-1, 1), (1,), (0, -1, 1), ""), (2, 2, 2, 2)),
    (np.diag([0, 1]), PolynomialPair("projector", (0, 1), (-1, 1), (1,), (0, -1, 1), ""), (1, 1, 2, 0)),
])
def test_polynomial_ranks_match_exact_small_references(matrix, pair, expected):
    coefficients = (pair.f, pair.g, pair.gcd, pair.lcm)
    matrices = [evaluate_polynomial(matrix, polynomial) for polynomial in coefficients]
    # These integer cases evaluate exactly even in fp32; rational elimination
    # independently checks every rank, not only the equality of rank sums.
    ranks = tuple(exact_rank(value.astype(int).tolist()) for value in matrices)
    assert ranks == expected
    report, _ = evaluate_pair(matrix.astype(np.float32), pair)
    direct, _ = evaluate_pair(matrix.astype(np.float32), replace(pair, spectral_roots=()))
    assert tuple(report["ranks"].values()) == tuple(direct["ranks"].values()) == expected
    assert ranks[0] + ranks[1] == ranks[2] + ranks[3]


def test_near_zero_singular_value_is_numerical_policy_not_exact_rank():
    matrix = np.diag([Fraction(1, 2**20), Fraction(0), Fraction(1)])
    assert exact_rank(matrix.tolist()) == 2
    assert numerical_rank(np.asarray(matrix, dtype=np.float32), rank_tol=1e-5) == 1
    assert numerical_rank(np.asarray(matrix, dtype=np.float32), rank_tol=1e-8) == 2


def test_shared_polynomial_images_account_for_union_and_intersection():
    # One common diagonal operator, with f=x-1 and g=x-3.
    matrix = np.diag([1, 2, 3])
    pair = PolynomialPair("shared", (-1, 1), (-3, 1), (1,), (3, -4, 1), "")
    report, spaces = evaluate_pair(matrix.astype(np.float32), pair)
    assert report["ranks"] == {"f": 2, "g": 2, "gcd": 3, "lcm": 1}
    assert exact_rank(np.concatenate([spaces["f"], spaces["g"]], axis=1).astype(int).tolist()) == 3
    # The theorem accounts for this shared middle coordinate; it supplies no
    # preference for retaining it in a learning task.
    assert np.flatnonzero(np.diag(spaces["lcm"])).tolist() == [1]
