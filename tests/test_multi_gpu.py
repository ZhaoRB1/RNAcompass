"""CPU tests for the one-command multi-GPU dispatcher."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_multi_gpu.py"


def run_launcher(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LAUNCHER), *args], cwd=ROOT, text=True, capture_output=True
    )


def test_dispatches_multifasta_records_with_unique_ids(tmp_path: Path) -> None:
    fasta = tmp_path / "records.fa"
    fasta.write_text(">same one\nACGUACGU\n>same two\nACGUACGU\n>third\nACGUACGU\n")
    output = tmp_path / "output"

    completed = run_launcher(
        "--fasta", str(fasta), "--output-dir", str(output),
        "--devices", "cuda:0,cuda:1", "--dry-run",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifests = []
    for shard in ("shard_0", "shard_1"):
        with (output / shard / "run_manifest.tsv").open(newline="") as stream:
            manifests.append(list(csv.DictReader(stream, delimiter="\t")))
    assert [row["sequence_id"] for row in manifests[0]] == ["same", "third"]
    assert len(manifests[1]) == 1
    assert manifests[1][0]["sequence_id"].startswith("same_")
    summary = json.loads((output / "multi_gpu_summary.json").read_text())
    assert summary["success"] is True
    assert summary["sequence_count"] == 3
    assert [(item["device"], item["sequence_count"]) for item in summary["shards"]] == [
        ("cuda:0", 2), ("cuda:1", 1)
    ]


@pytest.mark.parametrize("devices", ["cuda:0,cuda:0", "cuda:0,cuda:00"])
def test_rejects_duplicate_devices(tmp_path: Path, devices: str) -> None:
    fasta = tmp_path / "one.fa"
    fasta.write_text(">one\nACGUACGU\n")
    output = tmp_path / "output"

    completed = run_launcher(
        "--fasta", str(fasta), "--output-dir", str(output),
        "--devices", devices, "--dry-run",
    )

    assert completed.returncode != 0
    assert "duplicate" in completed.stderr.lower()
    assert not output.exists()


def test_worker_failure_returns_nonzero_and_reports_shard(tmp_path: Path) -> None:
    fasta = tmp_path / "records.fa"
    fasta.write_text(">one\nACGUACGU\n>two\nACGUACGU\n")
    output = tmp_path / "output"

    completed = run_launcher(
        "--fasta", str(fasta), "--output-dir", str(output),
        "--devices", "cuda:0,cuda:1", "--batch-size", "0", "--dry-run",
    )

    assert completed.returncode != 0
    summary = json.loads((output / "multi_gpu_summary.json").read_text())
    assert summary["success"] is False
    assert all(item["status"] == "FAILED" for item in summary["shards"])
