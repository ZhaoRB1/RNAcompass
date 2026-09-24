"""Command-line interface for the frozen RNAcompass v1 model."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .model import MODEL_ARCHITECTURE, MODEL_ID, MODEL_MAX_SEQUENCE_LENGTH, MODEL_NAME, MODEL_THRESHOLD
from .weights import expected_archive_manifest, install_archive, resolve_weights_dir, verify_weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rnacompass", description="RNAcompass single-model RNA dependency-map inference")
    parser.add_argument("--version", action="version", version=f"RNAcompass {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    weights = subcommands.add_parser("weights", help="install or inspect the manually downloaded weight archive")
    weights_subcommands = weights.add_subparsers(dest="weights_command", required=True)
    install = weights_subcommands.add_parser("install", help="safely install and verify a weight archive")
    install.add_argument("--archive", required=True)
    install.add_argument("--dest", default=None)
    status = weights_subcommands.add_parser("status", help="show installed weight status")
    status.add_argument("--weights-dir", default=None)
    status.add_argument("--check-hashes", action="store_true")
    status.add_argument("--json", action="store_true")

    preflight = subcommands.add_parser("preflight", help="validate environment, CUDA, resources and weight hashes")
    preflight.add_argument("--weights-dir", default=None)
    preflight.add_argument("--device", default="cuda:0")
    preflight.add_argument("--json", action="store_true")

    info = subcommands.add_parser("info", help="show the frozen public model specification")
    info.add_argument("--json", action="store_true")

    run = subcommands.add_parser("run", help="run inference and fixed structural filtering")
    inputs = run.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fasta")
    inputs.add_argument("--fasta-dir")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--max-seq-len", type=int, default=MODEL_MAX_SEQUENCE_LENGTH)
    run.add_argument("--weights-dir", default=None)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--no-heatmaps", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="validate input and output planning without loading weights")
    return parser


def _print_weight_report(report: dict[str, object]) -> None:
    print(f"weights_dir: {report['weights_dir']}")
    for row in report["weights"]:
        print(f"{row['status']:>13}  {row['name']}  {row['path']}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "weights":
            if args.weights_command == "install":
                destination = install_archive(args.archive, args.dest)
                print(f"Installed and verified RNAcompass weights: {destination}")
                return 0
            report = verify_weights(resolve_weights_dir(args.weights_dir), check_hashes=args.check_hashes)
            print(json.dumps(report, indent=2, sort_keys=True) if args.json else "", end="" if args.json else "")
            if not args.json:
                _print_weight_report(report)
            elif args.json:
                print()
            return 0 if report["success"] else 1
        if args.command == "preflight":
            from .preflight import run_preflight

            report = run_preflight(args.weights_dir, device=args.device)
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                for row in report["checks"]:
                    print(f"{row['status']:>16}  {row['check']}")
                print("PASS" if report["success"] else "FAIL")
            return 0 if report["success"] else 1
        if args.command == "info":
            info = {
                "software": "RNAcompass",
                "version": __version__,
                "model_id": MODEL_ID,
                "model_name": MODEL_NAME,
                "architecture": MODEL_ARCHITECTURE,
                "threshold": MODEL_THRESHOLD,
                "max_sequence_length": MODEL_MAX_SEQUENCE_LENGTH,
                "weights_dir": str(resolve_weights_dir()),
                "weight_archive": expected_archive_manifest(),
            }
            if args.json:
                print(json.dumps(info, indent=2, sort_keys=True))
            else:
                for key, value in info.items():
                    if key != "weight_archive":
                        print(f"{key}: {value}")
            return 0
        if args.command == "run":
            from .pipeline import run_pipeline

            return run_pipeline(
                fasta=args.fasta,
                fasta_dir=args.fasta_dir,
                output_dir=args.output_dir,
                device=args.device,
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len,
                weights_dir=args.weights_dir,
                overwrite=args.overwrite,
                no_heatmaps=args.no_heatmaps,
                dry_run=args.dry_run,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"rnacompass: error: {exc}", file=sys.stderr)
        return 2
    return 2

