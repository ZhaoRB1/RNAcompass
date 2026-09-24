# RNAcompass

<p align="center">
  <img src="docs/assets/rnacompass.png" alt="Overview of the RNAcompass co-distillation framework" width="100%">
</p>

<p align="center">
  <em>Overview of the RNAcompass co-distillation framework.</em>
</p>

RNAcompass v1 is the public, single-model inference package for the
Paper-R2 ERNIE-RNA-giga epoch-2 checkpoint. It generates a raw RNA dependency
map and applies the frozen zero-sum 5×5 structural filter with threshold
`14.168192944655118`.

**Release ownership.** Ruobin Zhao is the stated software-copyright and
model-rights holder; Ruobin Zhao and Xitong Liu are the software authors, and
Lei Sun is the corresponding contact. The evidence separating the
RNAcompass-trained checkpoints from the ERNIE-RNA-derived implementation is
recorded in [model_provenance.md](docs/model_provenance.md).

The repository contains code only. Both the RNAcompass-trained
ERNIE-RNA-giga base checkpoint and the distilled checkpoint are distributed
separately because together they exceed 4.8 GB. Before the permanent public
archive is assigned, the verified weight archive can be requested from the
corresponding author, Lei Sun (`sunlei0227@sdu.edu.cn`), and installed with the
checksum-verifying command below.

## Supported platform

- Linux
- Python 3.9.16
- NVIDIA GPU with a CUDA 11.3-compatible driver
- Validated on an NVIDIA A100 80 GB GPU

The minimum GPU-memory requirement has not yet been characterized. CPU
inference is not supported by the public v1 CLI.

## Installation

```bash
conda env create -f environment.yml
conda activate rnacompass
python -m pip install -e .
```

PyTorch and Fairseq are intentionally installed through the pinned Conda
environment rather than selected dynamically at runtime.

## Install the weights

Download `rnacompass-weights-v1.tar.gz` manually, then run:

```bash
rnacompass weights install --archive /path/to/rnacompass-weights-v1.tar.gz
rnacompass weights status --check-hashes
rnacompass preflight
```

The default destination is `${XDG_CACHE_HOME:-~/.cache}/rnacompass/weights/v1`.
It may be overridden with `RNACOMPASS_WEIGHTS_DIR` or `--weights-dir`; the CLI
argument has highest priority. RNAcompass does not download files or execute
Google Drive helpers.

Both files are required:

| File | Size | SHA256 |
|---|---:|---|
| `base/checkpoint_12_960000.pt` | 3,653,738,558 | `e83ae351adf73ae2c619ff969d91b8545b1dc93745abaffd5eb2ce9cd380f61e` |
| `distilled/vernie_rna_giga_irf_epoch002.pth` | 1,218,010,975 | `02ae70fe93f3fa3b9b1b1020757b4e032df22becd225a055144cae17b032f9cd` |

## Run RNAcompass

```bash
rnacompass run \
  --fasta examples/RF00001.fa \
  --output-dir output \
  --device cuda:0 \
  --batch-size 4
```

For a directory of `.fa`, `.fasta`, or `.fna` files, use `--fasta-dir`.
`--max-seq-len` may reduce, but cannot increase, the model limit of 1022 nt.
DNA `T` is normalized to `U`. Other symbols are mapped to the ERNIE unknown
token, excluded as mutation sites, and reported as warnings.

Use `--dry-run` to validate FASTA parsing and planned sample identifiers
without importing PyTorch or requiring weights:

```bash
rnacompass run --fasta examples/RF00001.fa --output-dir output --dry-run
```

Each sample is written directly under `output/results/<sample>/`:

- `raw_dep_map.npy`
- `processed_dep_map.npy`
- `convolution_5x5.npy`
- `pairs.tsv`
- `helices.tsv`
- `summary.json`
- PDF, SVG and PNG heatmaps unless `--no-heatmaps` is supplied

The top-level `run_manifest.tsv` uses relative result paths and records the
full model ID and checkpoint digest. Existing sample results are protected
unless `--overwrite` is specified.

## Scientific behavior

RNAcompass symmetrizes the raw dependency matrix, masks `|i-j| <= 3`, and
applies a zero-sum 5×5 kernel with anti-diagonal weights 1.6 and all other
weights -0.4. Candidates must be upper-triangular canonical or G–U wobble
pairs and must exceed the frozen absolute threshold. Radius-two anti-diagonal
continuity pruning runs before and after deterministic raw-CI-first positional
deduplication. A prediction is positive only when at least two independent
stem regions, each containing at least two immediately adjacent pairs, remain.

See [reproducibility.md](docs/reproducibility.md) for the reference fixture and
GPU release gate.

## Citation and licenses

The associated manuscript is currently in preparation:

> Ruobin Zhao, Pingping Cao, Xitong Liu, Jingwen Wang, Yuning Liu, Chengqian
> Wang, Jiale He, Qiuzhen Chen, Xiaona Liu and Lei Sun. “Distilling
> complementary knowledge across RNA language models reveals conserved
> structures from single sequences.” Manuscript in preparation.

Ruobin Zhao, Pingping Cao and Xitong Liu contributed equally. Correspondence:
Lei Sun (`sunlei0227@sdu.edu.cn`). Machine-readable citation metadata is in
[`CITATION.cff`](CITATION.cff); a DOI and journal are intentionally omitted
until they exist.

RNAcompass source code is released under the [MIT License](LICENSE). Both
RNAcompass v1 checkpoint files are released under the MIT terms in
[`MODEL_LICENSE.md`](MODEL_LICENSE.md). Xitong Liu is identified as a software
author; Ruobin Zhao is the copyright and model-rights holder. The vendored
ERNIE-RNA runtime is separately attributed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Please also cite Yin et al.,
“ERNIE-RNA: an RNA language model with structure-enhanced representations,”
*Nature Communications* 16, 10076 (2025),
https://doi.org/10.1038/s41467-025-64972-0.
