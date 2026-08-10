import numpy as np
import pytest

from src.analysis.shared_axis_head_comparison import (
    compare_head_to_directions,
    shared_axis,
)


def test_shared_axis_is_normalized_oriented_mean() -> None:
    axis = shared_axis(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(axis, [1 / np.sqrt(2), 1 / np.sqrt(2)])
    assert np.isclose(np.linalg.norm(axis), 1.0)


def test_head_comparison_uses_all_pairs_and_cross_dataset_subset() -> None:
    directions = np.array([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.0, 1.0]])
    result = compare_head_to_directions(
        np.array([1.0, 0.0]), directions, source_index=0
    )
    assert np.allclose(result["head_to_dataset_cosines"], [1.0, 0.8, 0.6, 0.0])
    assert np.isclose(result["head_to_dataset_mean"], 0.6)
    assert np.isclose(result["head_to_cross_dataset_mean"], (0.8 + 0.6) / 3)
    expected_pairs = [0.8, 0.6, 0.0, 0.96, 0.6, 0.8]
    assert np.isclose(result["prototype_pairwise_mean"], np.mean(expected_pairs))


def test_head_comparison_preserves_af_minus_nonaf_orientation() -> None:
    result = compare_head_to_directions(
        np.array([1.0, 0.0]),
        np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        source_index=1,
    )
    assert result["head_to_dataset_cosines"] == [-1.0, 1.0, 0.0]


def test_shared_axis_rejects_cancelling_directions() -> None:
    with pytest.raises(ValueError, match="zero norm"):
        shared_axis(np.array([[1.0, 0.0], [-1.0, 0.0]]))
