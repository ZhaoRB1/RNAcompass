from __future__ import annotations

from pathlib import Path
import os

import numpy as np


CONTACT_MAP_DPI = 600
HEATMAP_DPI = 600


def _configure_matplotlib(matplotlib) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def contact_map_cells(length: int, pairs) -> set[tuple[int, int]]:
    """Return the self diagonal and symmetric retained-pair cells."""
    if length <= 0:
        raise ValueError(f"contact-map length must be positive, got {length}")
    cells = {(position, position) for position in range(length)}
    for pair in pairs:
        i = int(pair.i)
        j = int(pair.j)
        if not (0 <= i < length and 0 <= j < length):
            raise ValueError(f"contact-map pair ({i}, {j}) is outside matrix length {length}")
        cells.add((i, j))
        cells.add((j, i))
    return cells


def _residue_ticks(length: int, intervals: int = 7) -> list[int]:
    if length == 1:
        return [1]
    values = np.linspace(1, length, num=min(intervals + 1, length), dtype=int)
    return sorted(set(int(value) for value in values) | {1, length})


def _contact_map_side(length: int) -> float:
    return min(8.0, max(3.2, length / 32.0))


def write_conserved_pair_contact_map(
    prefix: Path,
    *,
    length: int,
    pairs,
    title: str,
    dpi: int = CONTACT_MAP_DPI,
) -> dict[str, str]:
    """Write a black-and-white binary contact map as SVG, PDF, and PNG."""
    if dpi <= 0:
        raise ValueError(f"contact-map dpi must be positive, got {dpi}")
    cache = prefix.parent.parent.parent.parent / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    _configure_matplotlib(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cells = contact_map_cells(length, pairs)
    side = _contact_map_side(length)
    fig, ax = plt.subplots(figsize=(side, side), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for i, j in sorted(cells):
        ax.add_patch(
            Rectangle(
                (j - 0.5, i - 0.5),
                1.0,
                1.0,
                facecolor="black",
                edgecolor="none",
                linewidth=0,
            )
        )
    ticks = _residue_ticks(length)
    locations = [position - 1 for position in ticks]
    ax.set_xticks(locations, labels=[str(position) for position in ticks])
    ax.set_yticks(locations, labels=[str(position) for position in ticks])
    ax.set_xlim(-0.5, length - 0.5)
    ax.set_ylim(length - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Position j")
    ax.set_ylabel("Position i")
    ax.set_title(f"Filter Dependency\n{title}", fontsize=9, pad=6)
    ax.tick_params(axis="both", labelsize=7, length=2.5, width=0.6)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("black")
    fig.tight_layout(pad=0.8)
    paths = {
        "svg": prefix.with_suffix(".svg"),
        "pdf": prefix.with_suffix(".pdf"),
        "png": prefix.with_suffix(".png"),
    }
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["png"], dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def _draw_pair_overlays(ax, pairs, *, color: str, upper_triangle_only: bool = False) -> None:
    """Draw retained pairs without changing the underlying heatmap values."""
    from matplotlib.patches import Rectangle

    for pair in pairs:
        i = int(pair.i)
        j = int(pair.j)
        if upper_triangle_only:
            if i >= j:
                continue
            inset = 0.12
            ax.add_patch(
                Rectangle(
                    (j - 0.5 + inset, i - 0.5 + inset),
                    1.0 - 2.0 * inset,
                    1.0 - 2.0 * inset,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=0.85,
                    joinstyle="miter",
                    zorder=3,
                )
            )
        else:
            ax.scatter(
                [j, i],
                [i, j],
                s=12,
                facecolors="none",
                edgecolors=color,
                linewidths=0.5,
                zorder=3,
            )


def _save(
    matrix: np.ndarray,
    prefix: Path,
    title: str,
    pairs=(),
    *,
    overlay_color: str | None = None,
    upper_triangle_only: bool = False,
) -> None:
    cache = prefix.parent.parent.parent.parent / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    _configure_matplotlib(matplotlib)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=300)
    image = ax.imshow(matrix, origin="upper", cmap="coolwarm", interpolation="nearest")
    if overlay_color is not None:
        _draw_pair_overlays(
            ax,
            pairs,
            color=overlay_color,
            upper_triangle_only=upper_triangle_only,
        )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Position j (0-based)")
    ax.set_ylabel("Position i (0-based)")
    fig.colorbar(image, ax=ax, label="Dependency / convolution score")
    fig.tight_layout()
    # Explicitly set the export DPI.  Without it, imshow is embedded in PDF at
    # Matplotlib's low default DPI even though the SVG preview appears sharp.
    fig.savefig(prefix.with_suffix(".svg"), dpi=HEATMAP_DPI, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)


def write_raw_and_candidate_heatmaps(
    output_dir: Path,
    *,
    raw: np.ndarray,
    pairs,
    title: str,
) -> dict[str, str]:
    """Write the raw heatmap and raw-plus-candidate overlay pair.

    The candidate panel deliberately uses the raw matrix as its background and
    draws only upper-triangular retained pairs as green cell outlines.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_prefix = output_dir / "raw_dep_map_heatmap"
    sites_prefix = output_dir / "conserved_pair_candidates_heatmap"
    raw_display = np.max(raw, axis=2) if raw.ndim == 3 else raw
    _save(raw_display, raw_prefix, f"Raw dependency map: {title}")
    _save(
        raw_display,
        sites_prefix,
        f"Predicted conserved-pair candidates: {title}",
        pairs,
        overlay_color="#00ff00",
        upper_triangle_only=True,
    )
    return {
        "raw_dep_map_heatmap": str(raw_prefix.with_suffix(".pdf")),
        "raw_dep_map_heatmap_svg": str(raw_prefix.with_suffix(".svg")),
        "conserved_pair_candidates_heatmap": str(sites_prefix.with_suffix(".pdf")),
        "conserved_pair_candidates_heatmap_svg": str(sites_prefix.with_suffix(".svg")),
    }


def write_heatmaps(output_dir: Path, *, raw: np.ndarray, processed: np.ndarray, convolution: np.ndarray, pairs, title: str):
    conv_prefix = output_dir / "conv5x5_heatmap"
    contact_prefix = output_dir / "conserved_pair_contact_map"
    raw_heatmaps = write_raw_and_candidate_heatmaps(
        output_dir,
        raw=raw,
        pairs=pairs,
        title=title,
    )
    _save(
        convolution,
        conv_prefix,
        f"Fixed 5x5 scores: {title}",
        pairs,
        overlay_color="red",
    )
    contact_paths = write_conserved_pair_contact_map(
        contact_prefix,
        length=processed.shape[0],
        pairs=pairs,
        title=title,
    )
    return {
        **raw_heatmaps,
        "conv5x5_heatmap": str(conv_prefix.with_suffix(".pdf")),
        "conserved_pair_contact_map": contact_paths["pdf"],
        "conserved_pair_contact_map_pdf": contact_paths["pdf"],
        "conserved_pair_contact_map_svg": contact_paths["svg"],
        "conserved_pair_contact_map_png": contact_paths["png"],
    }
