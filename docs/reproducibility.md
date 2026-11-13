# Reproducibility and release validation

The 120-nt `RF00001` sequence in `examples/` is the reference input. Validate
the environment and checkpoints, then run inference on an NVIDIA GPU:

```bash
rnacompass preflight --device cuda:0
rnacompass run --fasta examples/RF00001.fa --output-dir gpu-check --device cuda:0 --no-heatmaps
```

Inspect `gpu-check/results/RF00001/summary.json` and the corresponding
dependency map, pairs, and helices. The historical reference run reported:

- raw dependency map shape `(120, 120)`;
- `threshold_candidates == 63`;
- `final_pairs == 38`;
- 9 qualifying stem regions;
- both checkpoint SHA256 digests pass.

The repository's package checks and dry run do not exercise GPU inference.

## Reference validation record

The release gate passed on 2026-09-24 with Python 3.9.16, PyTorch 1.10.0,
CUDA 11.3, Fairseq 0.12.2 and an NVIDIA A800 80 GB GPU. For the 120-nt
RF00001 fixture at batch size 4:

- the raw dependency map had shape `(120, 120)`;
- `threshold_candidates` was 63, `final_pairs` was 38, and 9 qualifying stem
  regions were reported;
- peak PyTorch allocated and reserved memory were approximately 1.259 GiB and
  1.291 GiB, respectively.

The memory measurement applies only to this 120-nt fixture and is not a
minimum-memory claim for sequences up to the 1022-nt model limit.
