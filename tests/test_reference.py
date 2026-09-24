import csv
import json
from pathlib import Path

import numpy as np

from rnacompass.fasta import read_records
from rnacompass.filtering import filter_depmap
from rnacompass.model import MODEL_THRESHOLD
from rnacompass.model import DISTILLED_WEIGHT_SHA256, MODEL_ID
from rnacompass.outputs import write_filter_outputs


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "data" / "golden" / "RF00001"


def _coordinates(path: Path) -> list[tuple[int, int]]:
    with path.open(encoding="utf-8") as handle:
        return [(int(row["pos_i_0based"]), int(row["pos_j_0based"])) for row in csv.DictReader(handle, delimiter="\t")]


def test_rf00001_reference_filtering_is_frozen():
    record = read_records(fasta=ROOT / "examples" / "RF00001.fa")[0]
    raw = np.load(GOLDEN / "raw_dep_map.npy")
    result = filter_depmap(raw, record.sequence, MODEL_THRESHOLD)
    expected_summary = json.loads((GOLDEN / "summary.json").read_text())
    assert raw.shape == (120, 120)
    assert result.statistics["threshold_candidates"] == expected_summary["threshold_candidates"] == 63
    assert result.statistics["final_pairs"] == expected_summary["final_pairs"] == 38
    assert [(pair.i, pair.j) for pair in result.pairs] == _coordinates(GOLDEN / "pairs.tsv")


def test_public_summary_records_model_version_and_checkpoint(tmp_path: Path):
    record = read_records(fasta=ROOT / "examples" / "RF00001.fa")[0]
    raw = np.load(GOLDEN / "raw_dep_map.npy")
    result = filter_depmap(raw, record.sequence, MODEL_THRESHOLD)
    write_filter_outputs(
        tmp_path,
        result=result,
        sequence=record.sequence,
        sequence_id=record.sample_id,
        model_id=MODEL_ID,
        built_in_threshold=MODEL_THRESHOLD,
        threshold_source="built_in_optimized",
        software_version="1.0.0",
        checkpoint_sha256=DISTILLED_WEIGHT_SHA256,
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["software_version"] == "1.0.0"
    assert summary["model_id"] == MODEL_ID
    assert summary["checkpoint_sha256"] == DISTILLED_WEIGHT_SHA256
    assert summary["continuity_radius"] == 2
