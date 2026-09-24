#!/usr/bin/env python3
"""Audit the repository for non-public artifacts and unresolved metadata."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache", "build", "dist"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".txt", ".cff", ".csv", ".tsv"}
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?:^|[\s\"'=])/(?:mnt|home|data|data\d+|scratch)/[A-Za-z0-9_.-]+/", re.MULTILINE),
    re.compile(r"[A-Za-z]:\\\\Users\\\\"),
)
PLACEHOLDER = re.compile(r"__[A-Z][A-Z0-9_]*__")
LEGACY_MODEL_TOKENS = ("aido_rna", "rinalmo", "rna_fm", "rnaernie2", "paper_v1")


def audit(root: Path, *, release: bool) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            any(part in IGNORED_PARTS or part.endswith(".egg-info") for part in relative.parts)
            or not path.is_file()
        ):
            continue
        if path.suffix.lower() in MODEL_SUFFIXES:
            errors.append(f"model-weight file is forbidden: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"file exceeds 10 MiB: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in ABSOLUTE_PATH_PATTERNS:
            if pattern.search(text):
                errors.append(f"machine-specific absolute path in {relative}")
                break
        if relative.as_posix() != "scripts/check_release_tree.py":
            lowered = text.lower()
            for token in LEGACY_MODEL_TOKENS:
                if token in lowered:
                    errors.append(f"legacy model token {token!r} in {relative}")
        if release and PLACEHOLDER.search(text):
            errors.append(f"unresolved release placeholder in {relative}")
    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--release", action="store_true", help="also reject unresolved legal and project metadata")
    args = parser.parse_args()
    errors = audit(args.root.resolve(), release=args.release)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Release tree audit passed." if args.release else "Source tree audit passed; release placeholders were not evaluated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
