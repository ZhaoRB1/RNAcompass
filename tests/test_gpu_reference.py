import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rnacompass.pipeline import run_pipeline


pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get("RNACOMPASS_RUN_GPU_TESTS") != "1", reason="GPU release gate is opt-in")
def test_gpu_inference_matches_rf00001_reference(tmp_path: Path):
    output = tmp_path / "gpu-output"
    assert run_pipeline(
        fasta=ROOT / "examples" / "RF00001.fa",
        fasta_dir=None,
        output_dir=output,
        device="cuda:0",
        batch_size=4,
        no_heatmaps=True,
    ) == 0
    observed_dir = output / "results" / "RF00001"
    golden_dir = ROOT / "tests" / "data" / "golden" / "RF00001"
    np.testing.assert_allclose(
        np.load(observed_dir / "raw_dep_map.npy"),
        np.load(golden_dir / "raw_dep_map.npy"),
        rtol=1e-4,
        atol=1e-5,
    )
    with (observed_dir / "pairs.tsv").open() as handle:
        observed = [(row["pos_i_0based"], row["pos_j_0based"]) for row in csv.DictReader(handle, delimiter="\t")]
    with (golden_dir / "pairs.tsv").open() as handle:
        expected = [(row["pos_i_0based"], row["pos_j_0based"]) for row in csv.DictReader(handle, delimiter="\t")]
    assert observed == expected

    with (observed_dir / "helices.tsv").open() as handle:
        observed_helices = list(csv.DictReader(handle, delimiter="\t"))
    with (golden_dir / "helices.tsv").open() as handle:
        expected_helices = list(csv.DictReader(handle, delimiter="\t"))
    coordinate_fields = (
        "helix_id",
        "start_i_1based",
        "end_i_1based",
        "start_j_1based",
        "end_j_1based",
        "pair_count",
    )
    assert [tuple(row[field] for field in coordinate_fields) for row in observed_helices] == [
        tuple(row[field] for field in coordinate_fields) for row in expected_helices
    ]
    for score_field in ("mean_conv5x5_score", "max_conv5x5_score"):
        np.testing.assert_allclose(
            [float(row[score_field]) for row in observed_helices],
            [float(row[score_field]) for row in expected_helices],
            rtol=1e-4,
            atol=1e-5,
        )

    observed_summary = json.loads((observed_dir / "summary.json").read_text())
    expected_summary = json.loads((golden_dir / "summary.json").read_text())
    for field in ("threshold_candidates", "final_pairs"):
        assert observed_summary[field] == expected_summary[field]

    run_summary = json.loads((output / "run_summary.json").read_text())
    assert run_summary["success"] is True
    assert run_summary["checkpoint_sha256"] == "02ae70fe93f3fa3b9b1b1020757b4e032df22becd225a055144cae17b032f9cd"
