from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .filtering import FilterResult, cluster_independent_stems


def cluster_helices(pairs) -> list[list]:
    """Compatibility wrapper returning the filter's independent stem regions."""
    return [list(region) for region in cluster_independent_stems(pairs)]


def write_filter_outputs(
    output_dir: Path,
    *,
    result: FilterResult,
    sequence: str,
    sequence_id: str,
    model_id: str,
    built_in_threshold: float,
    threshold_source: str,
    software_version: str,
    checkpoint_sha256: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_path = output_dir / "processed_dep_map.npy"
    convolution_path = output_dir / "convolution_5x5.npy"
    np.save(processed_path, result.processed_matrix)
    np.save(convolution_path, result.convolution_matrix)
    pairs_path = output_dir / "pairs.tsv"
    pair_rows = []
    for index, pair in enumerate(result.pairs, 1):
        pair_rows.append(
            {
                "pair_id": f"pair_{index:04d}",
                "pos_i_0based": pair.i,
                "pos_j_0based": pair.j,
                "pos_i_1based": pair.i + 1,
                "pos_j_1based": pair.j + 1,
                "base_i": sequence[pair.i],
                "base_j": sequence[pair.j],
                "raw_ci_value": pair.raw_ci_score,
                "conv5x5_score": pair.conv_score,
                "model": model_id,
                "sequence_id": sequence_id,
            }
        )
    pair_fields = [
        "pair_id", "pos_i_0based", "pos_j_0based", "pos_i_1based", "pos_j_1based",
        "base_i", "base_j", "raw_ci_value", "conv5x5_score", "model", "sequence_id",
    ]
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pair_rows)
    helices_path = output_dir / "helices.tsv"
    helix_fields = [
        "helix_id", "model", "sequence_id", "start_i_1based", "end_i_1based",
        "start_j_1based", "end_j_1based", "pair_count", "mean_conv5x5_score", "max_conv5x5_score",
    ]
    with helices_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=helix_fields)
        writer.writeheader()
        for index, helix in enumerate(result.stem_regions, 1):
            values = [pair.conv_score for pair in helix]
            writer.writerow(
                {
                    "helix_id": f"helix_{index:04d}",
                    "model": model_id,
                    "sequence_id": sequence_id,
                    "start_i_1based": min(pair.i for pair in helix) + 1,
                    "end_i_1based": max(pair.i for pair in helix) + 1,
                    "start_j_1based": min(pair.j for pair in helix) + 1,
                    "end_j_1based": max(pair.j for pair in helix) + 1,
                    "pair_count": len(helix),
                    "mean_conv5x5_score": float(np.mean(values)),
                    "max_conv5x5_score": float(np.max(values)),
                }
            )
    summary_path = output_dir / "summary.json"
    summary = dict(result.statistics)
    summary.update(
        {
            "software": "RNAcompass",
            "software_version": software_version,
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_sha256,
            "sequence_id": sequence_id,
            "built_in_threshold": built_in_threshold,
            "threshold_source": threshold_source,
            "candidate_interpretation": (
                "positive conserved-structure candidate only when at least two "
                "independent anti-diagonal stem regions are retained"
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "processed_dep_map": str(processed_path),
        "convolution_5x5": str(convolution_path),
        "pairs": str(pairs_path),
        "helices": str(helices_path),
        "summary": str(summary_path),
    }
