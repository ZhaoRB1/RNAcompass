#!/usr/bin/env python3
"""Evaluate AUPRC from the DepMap matrices produced by one model.

Input raw DepMap matrices (.npy) and matching base-pair labels (RFxxxxx.tsv).
Report raw and fixed-5x5-convolution AUPRC for each family and their
equal-weight mean (80 labeled families by default for directory input).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.signal import convolve2d


RF_PATTERN = re.compile(r"RF\d{5}")
LABEL_COLUMNS = ("rf_id", "position_1based", "base", "paired_0_1", "partner_1based")
FAMILY_COLUMNS = (
    "model_key", "rf_id", "sequence_length", "candidate_pair_count", "positive_pair_count",
    "raw_auprc", "conv_auprc", "matrix_path", "matrix_sha256", "label_path", "label_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_label_pairs(path: Path, sequence_length: int) -> set[tuple[int, int]]:
    """Validate per-base labels and return positive pairs in zero-based coordinates."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"label file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if tuple(reader.fieldnames or ()) != LABEL_COLUMNS:
            raise ValueError(f"{path}: expected TSV columns {LABEL_COLUMNS}")
        rows = list(reader)
    if len(rows) != sequence_length:
        raise ValueError(f"{path}: {len(rows)} label rows for a {sequence_length}-nt matrix")
    partners: list[int] = []
    for expected_position, row in enumerate(rows, 1):
        if row["rf_id"] != path.stem:
            raise ValueError(f"{path}: RF ID differs from filename at position {expected_position}")
        try:
            position = int(row["position_1based"])
            paired = int(row["paired_0_1"])
            partner = int(row["partner_1based"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: invalid integer at position {expected_position}") from error
        if position != expected_position:
            raise ValueError(f"{path}: positions must run from 1 to {sequence_length} in order")
        if row["base"] not in {"A", "C", "G", "U", "N"}:
            raise ValueError(f"{path}: invalid RNA base at position {position}")
        if paired not in (0, 1) or (paired == 0) != (partner == 0):
            raise ValueError(f"{path}: paired flag and partner disagree at position {position}")
        if partner < 0 or partner > sequence_length or partner == position:
            raise ValueError(f"{path}: invalid partner at position {position}")
        partners.append(partner)
    pairs: set[tuple[int, int]] = set()
    for position, partner in enumerate(partners, 1):
        if partner:
            if partners[partner - 1] != position:
                raise ValueError(f"{path}: nonreciprocal partner at position {position}")
            if position < partner:
                pairs.add((position - 1, partner - 1))
    if not pairs:
        raise ValueError(f"{path}: no positive reference pairs")
    return pairs


def preprocess_raw_matrix(matrix: np.ndarray) -> np.ndarray:
    """Match the paper's DepMap preprocessing: channel max, symmetry, and near-diagonal mask."""
    array = np.asarray(matrix)
    if array.ndim == 3:
        if array.shape[2] not in (3, 4):
            raise ValueError(f"unsupported 3D matrix shape: {array.shape}")
        array = np.max(array, axis=2)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"expected a square raw matrix, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("raw matrix contains NaN or infinity")
    processed = np.maximum(array, array.T).astype(np.float32, copy=False).copy()
    positions = np.arange(processed.shape[0])
    processed[np.abs(positions[:, None] - positions[None, :]) <= 3] = 0.0
    return processed


def convolve_fixed_5x5(processed: np.ndarray) -> np.ndarray:
    """Apply the paper's fixed 5x5 convolution kernel to the processed DepMap."""
    kernel = np.full((5, 5), -0.4, dtype=np.float32)
    for row in range(5):
        kernel[row, 4 - row] = 1.6
    return convolve2d(processed, kernel, mode="same", boundary="fill", fillvalue=0.0).astype(np.float32)


def precision_envelope_right_step_auc(scores: np.ndarray, labels: np.ndarray, positive_count: int) -> float:
    """Compute full-candidate AUPRC with the paper's precision-envelope integral."""
    if positive_count <= 0 or len(scores) == 0:
        return 0.0
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int8)
    if score_array.ndim != 1 or label_array.ndim != 1 or len(score_array) != len(label_array):
        raise ValueError("scores and labels must be equal-length 1D arrays")
    order = np.argsort(-score_array, kind="mergesort")
    label_array = label_array[order]
    true_positives = np.cumsum(label_array, dtype=np.int64)
    predicted_positives = np.arange(1, label_array.size + 1, dtype=float)
    recalls = [0.0, *(true_positives / positive_count).tolist()]
    precisions = [1.0, *(true_positives / predicted_positives).tolist()]
    points = sorted(zip(recalls, precisions), key=lambda point: point[0])
    collapsed_recalls: list[float] = []
    collapsed_precisions: list[float] = []
    for recall, precision in points:
        if not collapsed_recalls or not np.isclose(recall, collapsed_recalls[-1]):
            collapsed_recalls.append(recall)
            collapsed_precisions.append(precision)
        else:
            collapsed_precisions[-1] = max(collapsed_precisions[-1], precision)
    recall_array = np.asarray(collapsed_recalls, dtype=np.float64)
    precision_array = np.asarray(collapsed_precisions, dtype=np.float64)
    precision_envelope = np.maximum.accumulate(precision_array[::-1])[::-1]
    if len(recall_array) < 2:
        return 0.0
    return float(np.sum(np.diff(recall_array) * precision_envelope[1:]))


def score_matrix(matrix_path: Path, label_path: Path, model_key: str) -> dict[str, object]:
    """Score raw and 5x5-convolved DepMaps for one test family."""
    matrix = np.load(matrix_path, allow_pickle=False)
    processed = preprocess_raw_matrix(matrix)
    length = processed.shape[0]
    truth = load_label_pairs(label_path, length)
    upper_i, upper_j = np.triu_indices(length, k=4)
    labels = np.asarray(
        [int((int(i), int(j)) in truth) for i, j in zip(upper_i, upper_j)], dtype=np.int8
    )
    conv = convolve_fixed_5x5(processed)
    return {
        "model_key": model_key,
        "rf_id": label_path.stem,
        "sequence_length": length,
        "candidate_pair_count": len(upper_i),
        "positive_pair_count": len(truth),
        "raw_auprc": precision_envelope_right_step_auc(processed[upper_i, upper_j], labels, len(truth)),
        "conv_auprc": precision_envelope_right_step_auc(conv[upper_i, upper_j], labels, len(truth)),
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": sha256_file(matrix_path),
        "label_path": str(label_path.resolve()),
        "label_sha256": sha256_file(label_path),
    }


def discover_matrices(path: Path) -> list[tuple[str, Path]]:
    """Find flat matrices or RNAcompass results, preferring *_raw.npy in flat input."""
    path = Path(path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted({
            *path.glob("*.npy"),
            *path.glob("RF?????/raw_dep_map.npy"),
            *path.glob("shard_*/results/RF?????/raw_dep_map.npy"),
        })
    else:
        raise FileNotFoundError(f"matrix input not found: {path}")
    if not files:
        raise ValueError(f"no supported .npy matrices found in {path}")
    found: dict[str, Path] = {}
    for matrix in files:
        if matrix.suffix != ".npy":
            raise ValueError(f"matrix must be a .npy file: {matrix}")
        if matrix.name == "raw_dep_map.npy":
            rf_id = matrix.parent.name
            matches = [rf_id] if RF_PATTERN.fullmatch(rf_id) else []
        else:
            matches = RF_PATTERN.findall(matrix.stem)
        if len(matches) != 1:
            raise ValueError(f"matrix filename must contain exactly one RFxxxxx ID: {matrix}")
        rf_id = matches[0]
        if rf_id in found:
            previous = found[rf_id]
            if "raw_dep_map.npy" in (previous.name, matrix.name):
                raise ValueError(f"duplicate matrices for {rf_id}: {previous}, {matrix}")
            previous_is_raw = previous.name.endswith("_raw.npy")
            current_is_raw = matrix.name.endswith("_raw.npy")
            if previous_is_raw == current_is_raw:
                raise ValueError(f"duplicate matrices for {rf_id}: {previous}, {matrix}")
            if current_is_raw:
                found[rf_id] = matrix
            continue
        found[rf_id] = matrix
    return sorted(found.items())


def evaluate_input(
    matrices: Path,
    labels: Path,
    *,
    expected_families: int | None = None,
    output_dir: Path | None = None,
    model_key: str | None = None,
) -> list[dict[str, object]]:
    """Evaluate one model's DepMaps; directory input expects 80 labeled families by default."""
    matrices = Path(matrices)
    labels = Path(labels)
    discovered = discover_matrices(matrices)
    expected = expected_families if expected_families is not None else (1 if matrices.is_file() else 80)
    if labels.is_dir():
        label_ids = {path.stem for path in labels.glob("*.tsv") if RF_PATTERN.fullmatch(path.stem)}
    else:
        label_ids = {labels.stem}
    discovered = [(rf_id, path) for rf_id, path in discovered if rf_id in label_ids]
    if expected < 1 or len(discovered) != expected:
        missing = sorted(label_ids - {rf_id for rf_id, _ in discovered})
        raise ValueError(f"expected {expected} labeled matrix families, found {len(discovered)}; "
                         f"missing label IDs: {missing[:5]}")
    key = model_key or (matrices.parent.name if matrices.is_file() else matrices.name)
    rows = []
    for rf_id, matrix_path in discovered:
        label_path = labels / f"{rf_id}.tsv" if labels.is_dir() else labels
        if label_path.stem != rf_id:
            raise ValueError(f"matrix {rf_id} does not match label {label_path.stem}")
        rows.append(score_matrix(matrix_path, label_path, key))
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "per_family.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FAMILY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "model_key": key,
            "family_count": len(rows),
            "macro_raw_auprc": float(np.mean([row["raw_auprc"] for row in rows])),
            "macro_conv_auprc": float(np.mean([row["conv_auprc"] for row in rows])),
            "candidate_universe": "all upper-triangle pairs with j-i >= 4, without canonical filter",
            "integration": "stable score sort; precision envelope; right-step recall integral",
            "convolution": "fixed zero-sum 5x5, anti-diagonal 1.6, other cells -0.4, zero padding",
            "label_format": "reciprocal 1-based partner coordinates in RFxxxxx.tsv",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrices", type=Path, required=True,
                        help="one raw .npy, a flat directory, output/results, or multi-GPU output")
    parser.add_argument("--labels", type=Path, required=True,
                        help="directory of RFxxxxx.tsv files, or one matching TSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", help="name written to output; defaults to matrix directory name")
    parser.add_argument("--expected-families", type=int, help="default: 80 for a directory, 1 for one file")
    args = parser.parse_args()
    try:
        rows = evaluate_input(
            args.matrices, args.labels, expected_families=args.expected_families,
            output_dir=args.output_dir, model_key=args.model_key,
        )
    except (ValueError, FileNotFoundError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({
        "model_key": rows[0]["model_key"],
        "family_count": len(rows),
        "macro_raw_auprc": float(np.mean([row["raw_auprc"] for row in rows])),
        "macro_conv_auprc": float(np.mean([row["conv_auprc"] for row in rows])),
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
