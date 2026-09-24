# Third-party notices

RNAcompass includes the minimum ERNIE-RNA Python runtime needed to reconstruct
the published ERNIE-RNA-giga architecture and load the RNAcompass-trained base
checkpoint.

- Upstream: https://github.com/Bruce-ywj/ERNIE-RNA
- Upstream authors: Weijie Yin and contributors
- License: MIT; preserved in `licenses/ERNIE-RNA-LICENSE.txt`
- Reference: Yin et al., *ERNIE-RNA: an RNA language model with
  structure-enhanced representations*, Nature Communications 16, 10076
  (2025), https://doi.org/10.1038/s41467-025-64972-0

RNAcompass changes packaging and import paths and adds the frozen distilled
checkpoint loader and dependency-map inference workflow. The two checkpoint
files named in `MODEL_LICENSE.md` were trained by the RNAcompass project and
are not redistributed copies of an upstream pretrained checkpoint; their
parameters are licensed under `MODEL_LICENSE.md`. The ERNIE-RNA-derived code
remains under the preserved upstream MIT license. Python dependencies such as
PyTorch, Fairseq, NumPy, SciPy, scikit-learn and Matplotlib are installed as
dependencies and are not vendored here.
