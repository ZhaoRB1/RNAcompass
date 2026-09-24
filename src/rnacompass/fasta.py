"""FASTA discovery and deterministic sample naming."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


FASTA_SUFFIXES = {".fa", ".fasta", ".fna"}
RNA_BASES = frozenset("ACGU")


@dataclass(frozen=True)
class SequenceRecord:
    source_id: str
    sample_id: str
    sequence: str
    source_file: Path
    warnings: tuple[str, ...] = ()


def sanitize_sample_id(text: str, fallback: str = "sequence") -> str:
    value = re.sub(r"[^A-Za-z0-9._+-]+", "_", text.strip()).strip("._-")
    return (value or fallback)[:120]


def iter_fasta(path: Path):
    header: str | None = None
    chunks: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            elif header is None:
                raise ValueError(f"sequence data before FASTA header in {path}")
            else:
                chunks.append("".join(line.split()))
    if header is not None:
        yield header, "".join(chunks)


def normalize_sequence(sequence: str) -> tuple[str, tuple[str, ...]]:
    normalized = "".join(sequence.split()).upper()
    warnings: list[str] = []
    if "T" in normalized:
        normalized = normalized.replace("T", "U")
        warnings.append("T bases converted to U")
    ambiguous = sorted(set(normalized) - RNA_BASES)
    if ambiguous:
        warnings.append(
            "non-ACGU bases retained as unknown tokens and excluded from mutation candidates: "
            + ",".join(ambiguous)
        )
    return normalized, tuple(warnings)


def _input_files(fasta: str | Path | None, fasta_dir: str | Path | None) -> list[Path]:
    if bool(fasta) == bool(fasta_dir):
        raise ValueError("provide exactly one of --fasta or --fasta-dir")
    if fasta is not None:
        files = [Path(fasta).expanduser().resolve()]
    else:
        root = Path(fasta_dir).expanduser().resolve()  # type: ignore[arg-type]
        if not root.is_dir():
            raise FileNotFoundError(f"FASTA directory not found: {root}")
        files = sorted(path.resolve() for path in root.iterdir() if path.is_file() and path.suffix.lower() in FASTA_SUFFIXES)
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError("no readable FASTA inputs found")
    return files


def read_records(
    *,
    fasta: str | Path | None = None,
    fasta_dir: str | Path | None = None,
    max_sequence_length: int = 1022,
) -> list[SequenceRecord]:
    if max_sequence_length < 1 or max_sequence_length > 1022:
        raise ValueError("--max-seq-len must be between 1 and 1022")
    records: list[SequenceRecord] = []
    used: set[str] = set()
    for path in _input_files(fasta, fasta_dir):
        parsed = list(iter_fasta(path))
        if not parsed:
            raise ValueError(f"no FASTA records in {path}")
        for index, (header, raw_sequence) in enumerate(parsed, 1):
            source_id = header.split()[0] if header.split() else f"{path.stem}_{index}"
            sequence, warnings = normalize_sequence(raw_sequence)
            if not sequence:
                raise ValueError(f"empty FASTA record {source_id!r} in {path}")
            if len(sequence) > max_sequence_length:
                raise ValueError(
                    f"sequence {source_id!r} length {len(sequence)} exceeds max_seq_len={max_sequence_length}"
                )
            sample_id = sanitize_sample_id(source_id, f"record{index:04d}")
            if sample_id in used:
                suffix = hashlib.sha1(f"{path}:{index}:{sequence}".encode()).hexdigest()[:8]
                sample_id = f"{sample_id[:111]}_{suffix}"
            used.add(sample_id)
            records.append(SequenceRecord(source_id, sample_id, sequence, path, warnings))
    return records

