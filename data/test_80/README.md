# Held-out 80-family test set

This directory contains 80 Rfam-family consensus RNA sequences and their
matching base-pair reference labels for the RNAcompass Fig. 3 test set. Each
`sequences/RFxxxxx.fa` has one record and matches `labels/RFxxxxx.tsv` by RF
ID. The files contain inputs and reference labels, not model predictions.

Each TSV has five columns: `rf_id`, `position_1based`, `base`, `paired_0_1`,
and `partner_1based`. Coordinates are 1-based. Positive pairs have reciprocal
partner coordinates; zero means that no positive pair was assigned in this
reference. Positive pairs were taken from explicit pairs in the matching
Rfam Stockholm `SS_cons` consensus structure and retained when they fell
inside significant R-scape helix regions (the historical Fig. 3 labeling
rule). They are not direct experimental measurements or a claim that every
unlabeled position is unpaired.

For example, from the repository root:

```bash
rnacompass run \
  --fasta-dir data/test_80/sequences \
  --output-dir output \
  --device cuda:0

python scripts/evaluate_auprc.py \
  --matrices output/results \
  --labels data/test_80/labels \
  --output-dir auprc_results
```

The sequence data originate from [Rfam](https://rfam.org/downloads), which
publishes its data under [CC0](https://docs.rfam.org/en/latest/#license).
The precise Rfam release used for this local test-set snapshot was not
recorded. The paired labels are project-derived annotations.
