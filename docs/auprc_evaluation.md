# AUPRC evaluation

`scripts/evaluate_auprc.py` scores raw dependency maps against base-pair
labels. Run it from the repository root in the RNAcompass environment; the
existing NumPy and SciPy dependencies are sufficient. It does not load model
weights or rerun inference.

## Inputs and examples

Provide a directory of `RFxxxxx.tsv` label files using these five tab-separated
columns: `rf_id`, `position_1based`, `base`, `paired_0_1`, and
`partner_1based`. Positions and partners are 1-based; paired positions must
point to each other. Each label file must match the sequence length of its
matrix and contain at least one positive pair. Labels are supplied separately
with `--labels`.

The `--matrices` path may be:

- a directory of flat files such as `RF00001_full_ci_matrix_raw.npy`;
- `output/results`, containing `RFxxxxx/raw_dep_map.npy` from a single-GPU run;
- `output`, containing `shard_*/results/RFxxxxx/raw_dep_map.npy` from the
  multi-GPU dispatcher; or
- one flat `.npy` file together with one matching label file or label directory.

Example for a multi-GPU run:

```bash
python scripts/evaluate_auprc.py \
  --matrices output \
  --labels /path/to/labels \
  --output-dir auprc_results
```

The matrix must be raw, before convolution, thresholding, or structural
filtering. Both `L×L` and `L×L×3/4` arrays are accepted. If a flat directory
contains both a `*_raw.npy` file and another matrix for the same RF ID, the
raw file is selected. Other duplicates, including the same family in two GPU
shards, are rejected. The directory default is exactly 80 labeled families;
for a small check add `--expected-families N`. A single matrix file defaults
to one family.

## Calculation and outputs

For a multi-channel matrix, the evaluator takes the channel maximum. It then
symmetrizes scores with the elementwise maximum and zeros positions at
distance three or less. Both metrics use all upper-triangle candidates with
`j-i >= 4`, without canonical-pair or prediction-threshold filtering. The
second metric applies a zero-padded 5×5 kernel with weight 1.6 on the
anti-diagonal and -0.4 elsewhere.

Within each family, scores are stably sorted and integrated using a precision
envelope and right-step recall integral. `per_family.csv` records both AUPRC
values, counts, input paths, and SHA-256 digests. `summary.json` reports
equal-weight means across families (`macro_raw_auprc` and
`macro_conv_auprc`). Run separately for each model or checkpoint; the tool
does not select checkpoints or compute F1/MCC.
