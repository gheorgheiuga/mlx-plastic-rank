import json
import math

import pytest

from mlx_plastic_rank.packs.statistics import (
    compare_answer_mode_metrics,
    compare_paired_perplexity,
)


def test_paired_bootstrap_reports_signed_effects_and_degenerate_ci():
    result = compare_paired_perplexity(
        [0.0, 0.0, 0.0],
        [1, 1, 1],
        [math.log(4.0), math.log(4.0), math.log(4.0)],
        [1, 1, 1],
        resamples=100,
        seed=17,
    )

    assert result.left_perplexity == pytest.approx(1.0)
    assert result.right_perplexity == pytest.approx(4.0)
    assert result.ppl_difference == pytest.approx(-3.0)
    assert result.ppl_difference_ci == pytest.approx((-3.0, -3.0))
    assert result.relative_advantage == pytest.approx(0.75)
    assert result.relative_advantage_ci == pytest.approx((0.75, 0.75))
    assert result.probability_left_better == 1.0
    assert result.examples == 3
    assert result.total_tokens == 3


def test_paired_bootstrap_is_deterministic_and_json_serializable():
    arguments = (
        [0.3, 2.5, 1.0, 4.2],
        [1, 3, 2, 4],
        [0.8, 1.9, 1.4, 5.0],
        [1, 3, 2, 4],
    )

    first = compare_paired_perplexity(*arguments, resamples=257, seed=73)
    second = compare_paired_perplexity(*arguments, resamples=257, seed=73)

    assert first == second
    encoded = json.dumps(first.to_dict(), sort_keys=True)
    assert '"method": "paired_example_bootstrap"' in encoded
    assert first.ppl_difference_ci[0] <= first.ppl_difference_ci[1]
    assert first.relative_advantage_ci[0] <= first.relative_advantage_ci[1]
    assert 0.0 <= first.probability_left_better <= 1.0


def test_answer_mode_wrapper_reads_per_example_artifact_fields():
    left = {
        "example_loss_sums": [1.0, 2.0],
        "example_token_counts": [2, 3],
    }
    right = {
        "example_loss_sums": [1.5, 2.5],
        "example_token_counts": [2, 3],
    }

    result = compare_answer_mode_metrics(left, right, resamples=20, seed=4)

    assert result.examples == 2
    assert result.total_tokens == 5
    assert result.left_perplexity < result.right_perplexity
    assert result.ppl_difference < 0.0
    assert result.relative_advantage > 0.0
    assert result.probability_left_better == 1.0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (([1.0], [1], [1.0, 2.0], [1, 1]), "equal length"),
        (([], [], [], []), "At least one"),
        (([1.0], [0], [1.0], [0]), "must be positive"),
        (([1.0], [True], [1.0], [True]), "must be an integer"),
        (([1.0], [1], [1.0], [2]), "identical token counts"),
        (([float("nan")], [1], [1.0], [1]), "must be finite"),
        (([-0.1], [1], [1.0], [1]), "must be non-negative"),
    ],
)
def test_paired_bootstrap_rejects_unaligned_or_invalid_vectors(arguments, message):
    with pytest.raises(ValueError, match=message):
        compare_paired_perplexity(*arguments, resamples=10)


def test_paired_bootstrap_validates_configuration():
    vectors = ([1.0], [1], [1.0], [1])
    with pytest.raises(ValueError, match="resamples"):
        compare_paired_perplexity(*vectors, resamples=0)
    with pytest.raises(ValueError, match="confidence_level"):
        compare_paired_perplexity(*vectors, confidence_level=1.0)
    with pytest.raises(ValueError, match="seed"):
        compare_paired_perplexity(*vectors, seed=True)


def test_answer_mode_wrapper_requires_cluster_vectors():
    with pytest.raises(ValueError, match="example_token_counts"):
        compare_answer_mode_metrics(
            {"example_loss_sums": [1.0]},
            {"example_loss_sums": [1.0], "example_token_counts": [1]},
            resamples=10,
        )
