#!/usr/bin/env python3
"""Distribute FASTA records across independent RNAcompass GPU processes."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rnacompass.fasta import SequenceRecord, read_records
from rnacompass.pipeline import run_pipeline


def parse_devices(value: str) -> list[str]:
    devices = [part.strip() for part in value.split(",")]
    if not devices or any(re.fullmatch(r"cuda:\d+", device) is None for device in devices):
        raise argparse.ArgumentTypeError("--devices must be a comma-separated list such as cuda:0,cuda:1")
    devices = [f"cuda:{int(device.split(':', 1)[1])}" for device in devices]
    if len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError("--devices contains a duplicate CUDA device")
    return devices


def run_shard(
    records: list[SequenceRecord], device: str, output_dir: Path, args: argparse.Namespace
) -> int:
    return run_pipeline(
        fasta=None,
        fasta_dir=None,
        records=records,
        output_dir=output_dir,
        device=device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        weights_dir=args.weights_dir,
        overwrite=args.overwrite,
        no_heatmaps=args.no_heatmaps,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fasta", type=Path)
    inputs.add_argument("--fasta-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", type=parse_devices, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=1022)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-heatmaps", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        records = read_records(
            fasta=args.fasta, fasta_dir=args.fasta_dir, max_sequence_length=args.max_seq_len
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    groups: list[list[SequenceRecord]] = [[] for _ in args.devices]
    for index, record in enumerate(records):
        groups[index % len(groups)].append(record)
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=min(len(args.devices), len(records)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        jobs = [
            (index, device, group, pool.submit(
                run_shard, group, device, output_root / f"shard_{index}", args
            ))
            for index, (device, group) in enumerate(zip(args.devices, groups)) if group
        ]
        for index, device, group, future in jobs:
            try:
                code = future.result()
                status = "OK" if code == 0 else "FAILED"
                error = None if code == 0 else f"process returned {code}"
            except Exception as exc:
                status, error = "FAILED", str(exc)
            entry: dict[str, object] = {
                "device": device,
                "sequence_count": len(group),
                "output_dir": f"shard_{index}",
                "status": status,
            }
            if error:
                entry["error"] = error
            shards.append(entry)
    summary = {
        "success": all(shard["status"] == "OK" for shard in shards),
        "sequence_count": len(records),
        "shards": shards,
    }
    (output_root / "multi_gpu_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
