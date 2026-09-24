from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rnacompass.heatmaps import (
    contact_map_cells,
    write_conserved_pair_contact_map,
    write_raw_and_candidate_heatmaps,
)


def pair(i: int, j: int):
    return SimpleNamespace(i=i, j=j)


def test_contact_map_cells_are_diagonal_and_symmetric():
    cells = contact_map_cells(5, [pair(1, 4)])
    assert cells == {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (1, 4), (4, 1)}


def test_empty_and_one_nt_contact_maps_have_only_the_diagonal():
    assert contact_map_cells(4, []) == {(0, 0), (1, 1), (2, 2), (3, 3)}
    assert contact_map_cells(1, []) == {(0, 0)}


def test_contact_map_rejects_out_of_bounds_pairs():
    try:
        contact_map_cells(3, [pair(0, 3)])
    except ValueError as exc:
        assert "outside matrix length 3" in str(exc)
    else:
        raise AssertionError("out-of-bounds pair was accepted")


def test_contact_map_exports_editable_vectors_and_high_resolution_png(tmp_path):
    result_dir = tmp_path / "output" / "results" / "sample" / "model"
    result_dir.mkdir(parents=True)
    paths = write_conserved_pair_contact_map(
        result_dir / "conserved_pair_contact_map",
        length=6,
        pairs=[pair(1, 4)],
        title="sample (model)",
        dpi=120,
    )

    svg = (result_dir / "conserved_pair_contact_map.svg").read_text(encoding="utf-8")
    pdf = (result_dir / "conserved_pair_contact_map.pdf").read_bytes()
    png = (result_dir / "conserved_pair_contact_map.png").read_bytes()
    assert set(paths) == {"svg", "pdf", "png"}
    assert "<image" not in svg
    assert "Filter Dependency" in svg
    assert "sample (model)" in svg
    assert "#000000" in svg
    assert b"/Subtype /Image" not in pdf
    assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_raw_and_candidate_heatmaps_use_raw_background_and_green_upper_triangle(tmp_path):
    result_dir = tmp_path / "output" / "results" / "sample" / "model"
    result_dir.mkdir(parents=True)
    raw = np.arange(36, dtype=np.float32).reshape(6, 6)
    paths = write_raw_and_candidate_heatmaps(
        result_dir,
        raw=raw,
        pairs=[pair(1, 4), pair(4, 1)],
        title="sample (model)",
    )
    candidate_svg = (result_dir / "conserved_pair_candidates_heatmap.svg").read_text(encoding="utf-8")
    raw_pdf = (result_dir / "raw_dep_map_heatmap.pdf").read_bytes()
    assert paths["raw_dep_map_heatmap_svg"].endswith("raw_dep_map_heatmap.svg")
    assert paths["conserved_pair_candidates_heatmap_svg"].endswith("conserved_pair_candidates_heatmap.svg")
    assert "#00ff00" in candidate_svg.lower()
    assert "Raw dependency map" not in candidate_svg
    assert len(raw_pdf) > 10_000
