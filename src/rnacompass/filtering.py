from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


CANONICAL_PAIRS = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"), ("G", "U"), ("U", "G")}
MIN_PAIRS_PER_STEM = 2
# The frozen Paper-R2 configuration accepts anti-diagonal support one or two
# cells away. Independent stem reporting remains stricter and connects only
# immediately adjacent positions.
CONTINUITY_RADIUS = 2
STEM_MAX_STACK_GAP = 1
MIN_INDEPENDENT_STEMS = 2


@dataclass(frozen=True)
class PairPrediction:
    i: int
    j: int
    raw_ci_score: float
    conv_score: float

    @property
    def pair(self) -> tuple[int, int]:
        return normalize_pair(self.i, self.j)


@dataclass(frozen=True)
class FilterResult:
    processed_matrix: np.ndarray
    convolution_matrix: np.ndarray
    pairs: tuple[PairPrediction, ...]
    stem_regions: tuple[tuple[PairPrediction, ...], ...]
    statistics: dict[str, object]


def normalize_pair(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i <= j else (j, i)


def build_kernel() -> np.ndarray:
    kernel = np.full((5, 5), -0.4, dtype=np.float32)
    for row in range(5):
        kernel[row, 4 - row] = 1.6
    return kernel


def preprocess_ci_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim == 3:
        if array.shape[2] not in {3, 4}:
            raise ValueError(f"unsupported 3D matrix shape: {array.shape}")
        array = np.max(array, axis=2)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"expected square matrix, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix contains NaN or infinite values")
    processed = np.maximum(array, array.T).astype(np.float32, copy=False).copy()
    indices = np.arange(processed.shape[0])
    processed[np.abs(indices[:, None] - indices[None, :]) <= 3] = 0.0
    return processed


def apply_convolution(processed_ci: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import convolve2d

        return convolve2d(processed_ci, build_kernel(), mode="same", boundary="fill", fillvalue=0.0).astype(
            np.float32
        )
    except ImportError:
        padded = np.pad(np.asarray(processed_ci), 2, mode="constant")
        kernel = build_kernel()
        output = np.empty_like(processed_ci, dtype=np.float32)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.sum(padded[i : i + 5, j : j + 5] * kernel)
        return output


def _supported(pair: tuple[int, int], lookup: set[tuple[int, int]]) -> bool:
    """Return whether ``pair`` has a frozen radius-two anti-diagonal neighbor.

    For an upper-triangular base-pair matrix, stacked pairs move as
    ``(i - d, j + d)`` or ``(i + d, j - d)`` for ``d`` in ``{1, 2}``.
    Same-diagonal neighbors are not treated as support.
    """
    i, j = pair
    for distance in range(1, CONTINUITY_RADIUS + 1):
        for candidate in ((i - distance, j + distance), (i + distance, j - distance)):
            normalized = normalize_pair(*candidate)
            if normalized != pair and normalized in lookup:
                return True
    return False


def prune_isolated_until_stable(pairs: Sequence[PairPrediction]) -> tuple[list[PairPrediction], int, int]:
    current = list(pairs)
    iterations = 0
    while current:
        iterations += 1
        lookup = {item.pair for item in current}
        retained = [item for item in current if _supported(item.pair, lookup)]
        if len(retained) == len(current):
            break
        current = retained
    return current, len(pairs) - len(current), iterations


def greedy_unique_positions(pairs: Sequence[PairPrediction], max_pairs: int) -> list[PairPrediction]:
    ordered = sorted(pairs, key=lambda item: (-item.raw_ci_score, -item.conv_score, item.i, item.j))
    selected: list[PairPrediction] = []
    used: set[int] = set()
    for prediction in ordered:
        if prediction.i in used or prediction.j in used:
            continue
        selected.append(prediction)
        used.update((prediction.i, prediction.j))
        if len(selected) >= max_pairs:
            break
    return selected


def cluster_independent_stems(
    pairs: Sequence[PairPrediction],
) -> tuple[tuple[PairPrediction, ...], ...]:
    """Cluster retained pairs into independent anti-diagonal stem regions.

    Stacked pairs share ``i + j``. Only immediately consecutive positions are
    connected. Singleton components are excluded, so every returned region
    contains at least two supported pairs.
    """
    by_antidiagonal: dict[int, list[PairPrediction]] = {}
    for prediction in pairs:
        i, j = prediction.pair
        by_antidiagonal.setdefault(i + j, []).append(prediction)

    regions: list[tuple[PairPrediction, ...]] = []
    for group in by_antidiagonal.values():
        ordered = sorted(group, key=lambda item: item.pair[0])
        current: list[PairPrediction] = []
        for prediction in ordered:
            if current:
                previous_i = current[-1].pair[0]
                current_i = prediction.pair[0]
                if current_i - previous_i > STEM_MAX_STACK_GAP:
                    if len(current) >= MIN_PAIRS_PER_STEM:
                        regions.append(tuple(current))
                    current = []
            current.append(prediction)
        if len(current) >= MIN_PAIRS_PER_STEM:
            regions.append(tuple(current))
    return tuple(
        sorted(
            regions,
            key=lambda region: (region[0].pair[0] + region[0].pair[1], region[0].pair[0]),
        )
    )


def filter_depmap(
    matrix: np.ndarray,
    sequence: str,
    threshold: float,
    *,
    convolution_matrix: np.ndarray | None = None,
) -> FilterResult:
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("absolute threshold must be finite and non-negative")
    sequence = "".join(sequence.split()).upper().replace("T", "U")
    processed = preprocess_ci_matrix(matrix)
    if processed.shape[0] != len(sequence):
        raise ValueError(f"matrix length {processed.shape[0]} != sequence length {len(sequence)}")
    if convolution_matrix is None:
        convolution = apply_convolution(processed)
    else:
        convolution = np.asarray(convolution_matrix)
        if convolution.shape != processed.shape:
            raise ValueError(
                f"convolution length {convolution.shape} != processed matrix shape {processed.shape}"
            )
        if not np.all(np.isfinite(convolution)):
            raise ValueError("convolution matrix contains NaN or infinite values")
        convolution = convolution.astype(np.float32, copy=False)
    upper_i, upper_j = np.triu_indices(len(sequence), k=4)
    candidates = [
        PairPrediction(int(i), int(j), float(processed[i, j]), float(convolution[i, j]))
        for i, j in zip(upper_i, upper_j)
        if (sequence[i], sequence[j]) in CANONICAL_PAIRS and convolution[i, j] > threshold
    ]
    initial_count = len(candidates)
    candidates, first_removed, first_iterations = prune_isolated_until_stable(candidates)
    supported_count = len(candidates)
    selected = greedy_unique_positions(candidates, len(sequence) // 2)
    greedy_count = len(selected)
    selected, final_removed, final_iterations = prune_isolated_until_stable(selected)
    final = tuple(sorted(selected, key=lambda item: (-item.conv_score, item.i, item.j)))
    stem_regions = cluster_independent_stems(final)
    predicted_positive = len(stem_regions) >= MIN_INDEPENDENT_STEMS
    return FilterResult(
        processed_matrix=processed,
        convolution_matrix=convolution,
        pairs=final,
        stem_regions=stem_regions,
        statistics={
            "algorithm": "fixed_zero_sum_5x5_absolute_threshold_structural_cleanup",
            "kernel": build_kernel().tolist(),
            "sequence_length": len(sequence),
            "absolute_threshold": float(threshold),
            "threshold_candidates": int(np.count_nonzero(convolution[upper_i, upper_j] > threshold)),
            "after_base_pairing": initial_count,
            "after_initial_continuity": supported_count,
            "initial_isolated_removed": first_removed,
            "initial_continuity_iterations": first_iterations,
            "after_greedy": greedy_count,
            "final_isolated_removed": final_removed,
            "final_continuity_iterations": final_iterations,
            "final_pairs": len(final),
            "independent_stem_count": len(stem_regions),
            "stem_region_sizes": [len(region) for region in stem_regions],
            "minimum_independent_stems": MIN_INDEPENDENT_STEMS,
            "minimum_pairs_per_stem": MIN_PAIRS_PER_STEM,
            "maximum_stack_gap": STEM_MAX_STACK_GAP,
            "continuity_radius": CONTINUITY_RADIUS,
            "predicted_positive": predicted_positive,
            "candidate_positive_rule": "independent anti-diagonal stem regions >= 2",
        },
    )


def pair_as_dict(pair: PairPrediction) -> dict[str, object]:
    return asdict(pair)
