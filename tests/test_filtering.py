from __future__ import annotations

import numpy as np

from rnacompass.filtering import (
    PairPrediction,
    apply_convolution,
    build_kernel,
    filter_depmap,
    greedy_unique_positions,
    cluster_independent_stems,
    preprocess_ci_matrix,
    prune_isolated_until_stable,
)


def manual_convolution(matrix: np.ndarray) -> np.ndarray:
    padded = np.pad(matrix, 2)
    out = np.empty_like(matrix, dtype=np.float32)
    kernel = build_kernel()
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            out[i, j] = np.sum(padded[i : i + 5, j : j + 5] * kernel)
    return out


def test_kernel_and_boundary_convolution_are_fixed():
    kernel = build_kernel()
    assert kernel.shape == (5, 5)
    assert np.isclose(kernel.sum(), 0.0, atol=1e-5)
    assert np.allclose(np.fliplr(kernel).diagonal(), 1.6)
    other = kernel.copy()
    for row in range(5):
        other[row, 4 - row] = -0.4
    assert np.allclose(other, -0.4)
    matrix = np.arange(49, dtype=np.float32).reshape(7, 7)
    np.testing.assert_allclose(apply_convolution(matrix), manual_convolution(matrix), rtol=0, atol=2e-5)


def test_preprocessing_symmetrizes_channels_and_masks_four_diagonals():
    matrix = np.zeros((8, 8, 3), dtype=np.float32)
    matrix[0, 7, 1] = 3
    matrix[7, 0, 2] = 5
    matrix[1, 4, 0] = 99
    result = preprocess_ci_matrix(matrix)
    assert result[0, 7] == result[7, 0] == 5
    assert result[1, 4] == result[4, 1] == 0
    assert np.all(np.diag(result) == 0)


def test_strict_threshold_canonical_pairing_and_continuity(monkeypatch):
    sequence = "AAAAAAUUUUUU"
    convolution = np.zeros((12, 12), dtype=np.float32)
    convolution[0, 11] = 10.0  # strict > means this is excluded
    convolution[1, 10] = 10.1
    convolution[2, 9] = 12.0
    convolution[3, 8] = 20.0
    convolution[4, 7] = 30.0  # masked raw diagonal, never considered
    monkeypatch.setattr("rnacompass.filtering.apply_convolution", lambda _: convolution)
    result = filter_depmap(np.ones((12, 12), dtype=np.float32), sequence, 10.0)
    assert [(pair.i, pair.j) for pair in result.pairs] == [(3, 8), (2, 9), (1, 10)]
    assert result.statistics["absolute_threshold"] == 10.0
    assert result.statistics["independent_stem_count"] == 1
    assert result.statistics["predicted_positive"] is False


def test_isolation_and_raw_first_stable_greedy():
    pairs = [
        PairPrediction(0, 9, 4, 100),
        PairPrediction(1, 8, 5, 10),
        PairPrediction(1, 7, 6, 20),
        PairPrediction(6, 11, 9, 99),
    ]
    pruned, removed, _ = prune_isolated_until_stable(pairs)
    assert {(p.i, p.j) for p in pruned} == {(0, 9), (1, 8)}
    assert removed == 2
    selected = greedy_unique_positions(pairs, 4)
    assert [(p.i, p.j) for p in selected] == [(6, 11), (1, 7), (0, 9)]


def test_continuity_uses_frozen_radius_two_antidiagonal_support():
    # Radius-two support is part of the Paper-R2 filtering configuration.
    gap_pairs = [
        PairPrediction(0, 11, 4, 20),
        PairPrediction(2, 9, 5, 21),
    ]
    pruned, removed, _ = prune_isolated_until_stable(gap_pairs)
    assert pruned == gap_pairs
    assert removed == 0

    # Same-diagonal neighbors are not stacked-pair support either.
    main_diagonal_pairs = [
        PairPrediction(1, 10, 4, 20),
        PairPrediction(2, 11, 5, 21),
    ]
    pruned, removed, _ = prune_isolated_until_stable(main_diagonal_pairs)
    assert pruned == []
    assert removed == 2


def test_two_separate_antidiagonals_form_two_independent_stems():
    pairs = [
        PairPrediction(0, 11, 4, 20),
        PairPrediction(2, 9, 5, 21),
        PairPrediction(3, 16, 6, 22),
        PairPrediction(4, 15, 7, 23),
        PairPrediction(10, 19, 8, 24),  # singleton: not a stem region
    ]
    regions = cluster_independent_stems(pairs)
    assert [[item.pair for item in region] for region in regions] == [
        [(3, 16), (4, 15)],
    ]
