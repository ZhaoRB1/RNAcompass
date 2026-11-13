# Weight installation

The public RNAcompass v1 checkpoints are available from
[Google Drive](https://drive.google.com/drive/folders/1qIOZfk1cOgqKFd04wXXeLpuGF18qphBz?usp=sharing).
Follow the commands in the [README](../README.md#model-weights) to download them
under the required `base/` and `distilled/` paths and verify their SHA256
digests before inference. Downloading these two files directly does not require
a `manifest.json` file.

The repository's `src/rnacompass/data/model_manifest.json` records checkpoint
metadata for reference. Inference does not read this file: it uses the model
settings in `src/rnacompass/model.py` and the checkpoint specifications in
`src/rnacompass/weights.py`. The model identifier in results comes from
`model.py`.

## Optional archive installation

The `rnacompass weights install --archive` command expects one gzip-compressed
tar archive with this exact layout:

```text
rnacompass-weights-v1/
├── manifest.json
├── LICENSE
├── base/Base_ernie_rna_giga.pt
└── distilled/Distilled_ernie_rna_giga.pth
```

The `manifest.json` inside the archive is separate from the repository's
`model_manifest.json`. The archive manifest must contain `model_id`, plus the
exact relative path, byte size and SHA256 digest of both checkpoints. Run
`rnacompass info --json` to see the required `weight_archive` specification.

The installer rejects path traversal, absolute members, links, duplicates,
unknown files, missing license files, size mismatches and digest mismatches.
Installation uses a temporary sibling directory followed by an atomic rename.
An existing invalid destination is never overwritten automatically.

The archive's single `LICENSE` covers the RNAcompass-trained parameters and
preserves the MIT notice for the ERNIE-RNA-derived implementation. Its presence
does not indicate that an upstream checkpoint is being redistributed.

The archive is larger than a single GitHub Release asset can be. The current
public distribution therefore provides the two checkpoints separately; the
archive format above remains supported for mirrored or institutional releases.
