# Reproducibility and release validation

The reference fixture uses the 120-nt `RF00001` sequence in `examples/`.
The trusted Paper-R2 ERNIE-RNA-giga epoch-2 result is stored under
`tests/data/golden/RF00001/` with machine-specific paths removed.

The CPU test suite validates FASTA handling, filtering, output schemas,
archive safety, path resolution, packaging and release-tree hygiene.

Before claiming an inference-validated release, run on an NVIDIA GPU:

```bash
python -m pip install -e ".[test]"
rnacompass preflight --device cuda:0
rnacompass run --fasta examples/RF00001.fa --output-dir gpu-check --device cuda:0 --no-heatmaps
RNACOMPASS_RUN_GPU_TESTS=1 pytest -m gpu
```

Acceptance criteria are:

- raw dependency map shape `(120, 120)` and numerical agreement with the
  checked-in reference at `rtol=1e-4`, `atol=1e-5`;
- `threshold_candidates == 63`;
- `final_pairs == 38`;
- exact pair and helix coordinates after filtering, with numerical helix scores
  agreeing at `rtol=1e-4`, `atol=1e-5`;
- both checkpoint SHA256 digests pass.

CPU-only development and continuous integration do not satisfy this GPU gate.

## Private v1 validation record

The release gate passed on 2026-09-24 with Python 3.9.16, PyTorch 1.10.0,
CUDA 11.3, Fairseq 0.12.2 and an NVIDIA A800 80 GB GPU. For the 120-nt
RF00001 fixture at batch size 4:

- the raw dependency map had shape `(120, 120)` and maximum absolute
  difference `0.0` from the checked-in reference;
- `threshold_candidates` was 63, `final_pairs` was 38, and 9 qualifying stem
  regions were reported;
- peak PyTorch allocated and reserved memory were approximately 1.259 GiB and
  1.291 GiB, respectively.

The memory measurement applies only to this 120-nt fixture and is not a
minimum-memory claim for sequences up to the 1022-nt model limit.
