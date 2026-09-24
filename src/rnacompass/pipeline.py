"""Single-model RNAcompass inference and structural filtering pipeline."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .fasta import read_records
from .filtering import filter_depmap
from .heatmaps import write_heatmaps
from .model import DISTILLED_WEIGHT_SHA256, MODEL_ID, MODEL_MAX_SEQUENCE_LENGTH, MODEL_THRESHOLD
from .outputs import write_filter_outputs
from .preflight import run_preflight
from .weights import resolve_weights_dir


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _replace_result_directory(staging: Path, destination: Path, overwrite: bool) -> None:
    if not destination.exists():
        staging.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"result already exists; use --overwrite to replace it: {destination}")
    backup = destination.with_name(destination.name + ".previous")
    if backup.exists():
        shutil.rmtree(backup)
    destination.replace(backup)
    try:
        staging.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    else:
        shutil.rmtree(backup)


def run_pipeline(
    *,
    fasta: str | Path | None,
    fasta_dir: str | Path | None,
    output_dir: str | Path,
    device: str = "cuda:0",
    batch_size: int = 4,
    max_seq_len: int = MODEL_MAX_SEQUENCE_LENGTH,
    weights_dir: str | Path | None = None,
    overwrite: bool = False,
    no_heatmaps: bool = False,
    dry_run: bool = False,
) -> int:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    records = read_records(fasta=fasta, fasta_dir=fasta_dir, max_sequence_length=max_seq_len)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_root = output / "results"
    results_root.mkdir(exist_ok=True)
    logs_root = output / "logs"
    logs_root.mkdir(exist_ok=True)
    selected_weights = resolve_weights_dir(weights_dir)
    started = _now()
    log_path = logs_root / "rnacompass.log"

    with log_path.open("w", encoding="utf-8") as log_handle:
        def log(message: str) -> None:
            timestamped = f"{_now()} {message}"
            print(timestamped)
            log_handle.write(timestamped + "\n")
            log_handle.flush()

        log(f"RNAcompass {__version__}; model={MODEL_ID}; samples={len(records)}")
        for record in records:
            for warning in record.warnings:
                log(f"[warning] {record.sample_id}: {warning}")
            if (results_root / record.sample_id).exists() and not overwrite and not dry_run:
                raise FileExistsError(f"result already exists; use --overwrite: {results_root / record.sample_id}")

        if dry_run:
            rows = [
                {
                    "sequence_id": record.sample_id,
                    "source_id": record.source_id,
                    "source_file": record.source_file.name,
                    "sequence_length": len(record.sequence),
                    "model_id": MODEL_ID,
                    "status": "DRY_RUN",
                }
                for record in records
            ]
            _write_tsv(output / "run_manifest.tsv", rows)
            (output / "run_summary.json").write_text(
                json.dumps({"software": "RNAcompass", "software_version": __version__, "model_id": MODEL_ID, "dry_run": True, "sequence_count": len(records), "success": True}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return 0

        report = run_preflight(selected_weights, device=device)
        if not report["success"]:
            failed = [row["check"] for row in report["checks"] if row["status"] != "OK"]
            raise RuntimeError("preflight failed: " + ", ".join(str(item) for item in failed))

        from .inference import generate_raw_dependency_maps

        rows: list[dict[str, object]] = []
        for record, raw, final_batch_size in generate_raw_dependency_maps(
            records,
            weights_dir=selected_weights,
            device=device,
            batch_size=batch_size,
            log=log,
        ):
            final_dir = results_root / record.sample_id
            staging = Path(tempfile.mkdtemp(prefix=f".{record.sample_id}-", dir=results_root))
            try:
                with (staging / "raw_dep_map.npy").open("wb") as handle:
                    np.save(handle, raw)
                filtered = filter_depmap(raw, record.sequence, MODEL_THRESHOLD)
                write_filter_outputs(
                    staging,
                    result=filtered,
                    sequence=record.sequence,
                    sequence_id=record.sample_id,
                    model_id=MODEL_ID,
                    built_in_threshold=MODEL_THRESHOLD,
                    threshold_source="built_in_optimized",
                    software_version=__version__,
                    checkpoint_sha256=DISTILLED_WEIGHT_SHA256,
                )
                if not no_heatmaps:
                    write_heatmaps(
                        staging,
                        raw=raw,
                        processed=filtered.processed_matrix,
                        convolution=filtered.convolution_matrix,
                        pairs=filtered.pairs,
                        title=f"{record.sample_id} ({MODEL_ID})",
                    )
                _replace_result_directory(staging, final_dir, overwrite)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            rows.append(
                {
                    "sequence_id": record.sample_id,
                    "source_id": record.source_id,
                    "source_file": record.source_file.name,
                    "sequence_length": len(record.sequence),
                    "model_id": MODEL_ID,
                    "checkpoint_sha256": DISTILLED_WEIGHT_SHA256,
                    "threshold": MODEL_THRESHOLD,
                    "final_batch_size": final_batch_size,
                    "final_pairs": len(filtered.pairs),
                    "independent_stems": len(filtered.stem_regions),
                    "predicted_positive": filtered.statistics["predicted_positive"],
                    "result_dir": f"results/{record.sample_id}",
                    "status": "OK",
                }
            )
        _write_tsv(output / "run_manifest.tsv", rows)
        (output / "run_summary.json").write_text(
            json.dumps(
                {
                    "software": "RNAcompass",
                    "software_version": __version__,
                    "model_id": MODEL_ID,
                    "checkpoint_sha256": DISTILLED_WEIGHT_SHA256,
                    "started_at": started,
                    "finished_at": _now(),
                    "sequence_count": len(records),
                    "success": True,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    return 0

