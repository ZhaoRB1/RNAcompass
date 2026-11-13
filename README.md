# RNAcompass: Conserved RNA structure prediction from single sequences

- [Installation](#installation)
- [Model weights](#model-weights)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Multi-GPU inference](#multi-gpu-inference)
- [Checks and test data](#checks-and-test-data)
- [License](#license)

RNAcompass is a co-distilled RNA language model for detecting conserved
structural signals from a single RNA sequence. It integrates high-confidence
nucleotide dependency signals from multiple RNA language models, combining
their complementary information about RNA stems in one model.

Given a sequence, RNAcompass generates a dependency map and applies fixed
structural filtering to identify candidate base pairs and helices. The
co-distillation framework is described in *Distilling complementary knowledge
across RNA language models reveals conserved structures from single sequences*
and illustrated below.

<p align="center">
  <img src="docs/assets/rnacompass.png" alt="RNAcompass framework" width="800">
</p>

## Installation

Requirements: Linux, Conda, Python 3.9.16, and an NVIDIA GPU with a CUDA
11.3-compatible driver. Create the environment from the repository root:

```bash
git clone https://github.com/ZhaoRB1/RNAcompass.git && cd RNAcompass
conda env create -f environment.yml
conda activate rnacompass
python -m pip install -e .

# fairseq must be installed separately: its default build compiles CUDA
# extensions and pulls torchaudio, which conflicts with pytorch=1.10.0.
python -m pip install cython bitarray regex sacrebleu packaging \
  "tqdm==4.66.2" "hydra-core==1.0.7" "omegaconf==2.0.6"
READTHEDOCS=1 python -m pip install --no-deps --no-build-isolation "fairseq==0.12.2"

# On clusters that export a system CUDA path, use the environment's CUDA
# libraries if the first matrix multiply fails with CUBLAS_STATUS_NOT_SUPPORTED.
unset LD_LIBRARY_PATH
```

CPU inference is not supported in v1.

## Model weights

Download the two checkpoints from [Google Drive](https://drive.google.com/drive/folders/1qIOZfk1cOgqKFd04wXXeLpuGF18qphBz?usp=sharing).
The commands below place them in the default cache location. The files are
checked against the published SHA-256 digests before inference; see the
[weight installation notes](docs/weights.md) for the expected paths.

```bash
WEIGHTS_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/rnacompass/weights/v1"
mkdir -p "$WEIGHTS_DIR/base" "$WEIGHTS_DIR/distilled"

curl -L --fail --retry 5 --continue-at - \
  -o "$WEIGHTS_DIR/base/Base_ernie_rna_giga.pt" \
  "https://drive.usercontent.google.com/download?id=1bBG45EQi1TZKxYiZS3jpt8ePprHFlzGz&export=download&confirm=t"

curl -L --fail --retry 5 --continue-at - \
  -o "$WEIGHTS_DIR/distilled/Distilled_ernie_rna_giga.pth" \
  "https://drive.usercontent.google.com/download?id=1rz-QT529q64kymgW4E__i0vGEoP2VLMj&export=download&confirm=t"

rnacompass weights status --check-hashes
```

Checkpoints you already have can be reused: keep the two filenames above, then
point `RNACOMPASS_WEIGHTS_DIR` (or `--weights-dir`) at that directory instead of
the default cache path.

## Quick start

Check the environment and run the included 120-nt example:

```bash
rnacompass preflight --device cuda:0
rnacompass run \
  --fasta examples/RF00001.fa \
  --output-dir output \
  --device cuda:0
```

## Usage

RNAcompass accepts one FASTA file or a directory of `.fa`, `.fasta`, and `.fna`
files. To process a directory:

```bash
rnacompass run \
  --fasta-dir /path/to/fastas \
  --output-dir output \
  --device cuda:0 \
  --batch-size 4
```

Use `--dry-run` to validate inputs without loading the model weights or running
inference:

```bash
rnacompass run --fasta examples/RF00001.fa --output-dir output --dry-run
```

Each sample produces a raw dependency map (`raw_dep_map.npy`), predicted base
pairs and helices, a JSON summary, and PDF/SVG/PNG heatmaps under
`output/results/<sample>/`. The model supports sequences up to 1,022 nt.
Run `rnacompass run --help` for all options.

## Multi-GPU inference

Use a single Python command to distribute FASTA records across the selected
GPUs. The dispatcher starts one independent inference process for each device
that receives records:

```bash
python -m rnacompass.run_multi_gpu \
  --fasta-dir /path/to/fastas \
  --output-dir output \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --batch-size 4
```

Input records are assigned automatically, including records in a multi-FASTA
file (`--fasta` is also accepted). `--batch-size` applies independently on
each GPU; a single sequence is processed on one GPU. Results and logs are
kept in `output/shard_0/`, `output/shard_1/`, and so on. Inspect
`output/multi_gpu_summary.json` for per-shard status; any failed shard makes
the command exit nonzero. The selected devices must be visible to PyTorch.

## Checks and test data

Check input handling without loading weights, then validate the environment
and checkpoints before GPU inference:

```bash
rnacompass run --fasta examples/RF00001.fa --output-dir dry-run-check --dry-run
rnacompass preflight --device cuda:0
```

The held-out 80-family [test sequences and reference labels](data/test/) are
also included.

## License

RNAcompass source code and trained model parameters are released under the
[MIT License](LICENSE), which also preserves the notice for the bundled
ERNIE-RNA-derived source code.
