# Weight installation

Download the two RNAcompass v1 checkpoints from [Google Drive](https://drive.google.com/drive/folders/1qIOZfk1cOgqKFd04wXXeLpuGF18qphBz?usp=sharing)
using the commands in the [README](../README.md#model-weights). Their expected
paths under the weight directory are:

```text
base/Base_ernie_rna_giga.pt
distilled/Distilled_ernie_rna_giga.pth
```

Run `rnacompass weights status --check-hashes` to check both files against their
expected sizes and SHA-256 digests. `rnacompass preflight` repeats these checks
before GPU inference.

The repository's `src/rnacompass/data/model_manifest.json` records checkpoint
metadata for reference. Inference does not read this file: it uses the model
settings in `src/rnacompass/model.py` and the checkpoint specifications in
`src/rnacompass/weights.py`.
