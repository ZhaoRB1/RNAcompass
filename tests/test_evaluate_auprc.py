"""AUPRC evaluation tests for the released inference output layouts."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_auprc import evaluate_input, load_label_pairs


ROOT = Path(__file__).resolve().parents[1]


def write_labels(path: Path, length: int = 6) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("rf_id", "position_1based", "base", "paired_0_1", "partner_1based"))
        for position in range(1, length + 1):
            partner = 5 if position == 1 else 1 if position == 5 else 0
            writer.writerow((path.stem, position, "A", int(partner > 0), partner))


def write_matrix(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((6, 6), dtype=np.float32)
    matrix[0, 5] = 3.0
    matrix[0, 4] = 2.0
    np.save(path, matrix)


@pytest.mark.parametrize("layout", ["flat", "single", "multi"])
def test_scores_all_supported_matrix_layouts(tmp_path: Path, layout: str) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    write_labels(labels / "RF00001.tsv")
    matrices = tmp_path / "matrices"
    if layout == "flat":
        matrix_path = matrices / "RF00001_full_ci_matrix_raw.npy"
    elif layout == "single":
        matrix_path = matrices / "RF00001" / "raw_dep_map.npy"
    else:
        matrix_path = matrices / "shard_0" / "results" / "RF00001" / "raw_dep_map.npy"
    write_matrix(matrix_path)

    rows = evaluate_input(matrices, labels, expected_families=1, output_dir=tmp_path / "scores")

    assert rows[0]["raw_auprc"] == pytest.approx(0.5)
    assert rows[0]["matrix_path"] == str(matrix_path)
    summary = json.loads((tmp_path / "scores" / "summary.json").read_text())
    assert summary["macro_raw_auprc"] == pytest.approx(0.5)


def test_rejects_same_family_in_two_shards(tmp_path: Path) -> None:
    matrices = tmp_path / "matrices"
    for shard in ("shard_0", "shard_1"):
        write_matrix(matrices / shard / "results" / "RF00001" / "raw_dep_map.npy")
    labels = tmp_path / "labels"
    labels.mkdir()
    write_labels(labels / "RF00001.tsv")

    with pytest.raises(ValueError, match="duplicate"):
        evaluate_input(matrices, labels, expected_families=1)


def test_prefers_raw_flat_matrix_over_processed_copy(tmp_path: Path) -> None:
    matrices = tmp_path / "matrices"
    write_matrix(matrices / "RF00001_full_ci_matrix.npy")
    raw_path = matrices / "RF00001_full_ci_matrix_raw.npy"
    write_matrix(raw_path)
    labels = tmp_path / "labels"
    labels.mkdir()
    write_labels(labels / "RF00001.tsv")

    rows = evaluate_input(matrices, labels, expected_families=1)

    assert rows[0]["matrix_path"] == str(raw_path)


def test_cli_requires_explicit_labels(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_auprc.py"),
         "--matrices", str(tmp_path), "--output-dir", str(tmp_path / "scores")],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert completed.returncode != 0
    assert "--labels" in completed.stderr


def test_rejects_nonreciprocal_labels(tmp_path: Path) -> None:
    label = tmp_path / "RF00001.tsv"
    write_labels(label)
    content = label.read_text().replace("RF00001\t5\tA\t1\t1", "RF00001\t5\tA\t1\t2")
    label.write_text(content)
    with pytest.raises(ValueError, match="nonreciprocal"):
        load_label_pairs(label, 6)
