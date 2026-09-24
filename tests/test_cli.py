import json
from pathlib import Path

from rnacompass.cli import main
from rnacompass.model import MODEL_ID


def test_info_does_not_require_model_dependencies(capsys):
    assert main(["info", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model_id"] == MODEL_ID
    assert data["threshold"] == 14.168192944655118


def test_dry_run_uses_flat_single_model_output_plan(tmp_path: Path):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">sample one\nACGUTN\n", encoding="utf-8")
    output = tmp_path / "output"
    assert main(["run", "--fasta", str(fasta), "--output-dir", str(output), "--dry-run"]) == 0
    summary = json.loads((output / "run_summary.json").read_text())
    assert summary == {
        "dry_run": True,
        "model_id": MODEL_ID,
        "sequence_count": 1,
        "software": "RNAcompass",
        "software_version": "1.0.0",
        "success": True,
    }
    manifest = (output / "run_manifest.tsv").read_text()
    assert "paper_r2__ernie_rna_giga__epoch002" in manifest
    assert str(tmp_path) not in manifest


def test_existing_result_is_protected_before_preflight(tmp_path: Path, capsys):
    fasta = tmp_path / "input.fa"
    fasta.write_text(">sample\nACGUACGU\n", encoding="utf-8")
    output = tmp_path / "output"
    (output / "results" / "sample").mkdir(parents=True)
    code = main(["run", "--fasta", str(fasta), "--output-dir", str(output), "--no-heatmaps"])
    assert code == 2
    assert "--overwrite" in capsys.readouterr().err
