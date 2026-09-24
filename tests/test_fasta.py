from pathlib import Path

import pytest

from rnacompass.fasta import normalize_sequence, read_records


def test_normalization_reports_dna_and_ambiguous_bases():
    sequence, warnings = normalize_sequence("acgtn")
    assert sequence == "ACGUN"
    assert warnings == (
        "T bases converted to U",
        "non-ACGU bases retained as unknown tokens and excluded from mutation candidates: N",
    )


def test_multifasta_ids_are_sanitized_and_deduplicated(tmp_path: Path):
    fasta = tmp_path / "records.fa"
    fasta.write_text(">same id one\nACGU\n>same id two\nACGU\n", encoding="utf-8")
    records = read_records(fasta=fasta)
    assert records[0].sample_id == "same"
    assert records[1].sample_id.startswith("same_")
    assert len({record.sample_id for record in records}) == 2


def test_fasta_rejects_empty_and_excessive_length(tmp_path: Path):
    empty = tmp_path / "empty.fa"
    empty.write_text(">empty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty FASTA"):
        read_records(fasta=empty)
    long = tmp_path / "long.fa"
    long.write_text(">long\n" + "A" * 11 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        read_records(fasta=long, max_sequence_length=10)


def test_exactly_one_input_source_is_required(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one"):
        read_records()
    fasta = tmp_path / "one.fa"
    fasta.write_text(">one\nACGU\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        read_records(fasta=fasta, fasta_dir=tmp_path)

