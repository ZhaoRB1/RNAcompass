# Model provenance and licensing basis

This record distinguishes RNAcompass-trained parameters from the third-party
implementation on which the model architecture is based.

## Released parameter artifacts

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `base/checkpoint_12_960000.pt` | 3,653,738,558 | `e83ae351adf73ae2c619ff969d91b8545b1dc93745abaffd5eb2ce9cd380f61e` |
| `distilled/vernie_rna_giga_irf_epoch002.pth` | 1,218,010,975 | `02ae70fe93f3fa3b9b1b1020757b4e032df22becd225a055144cae17b032f9cd` |

The project training record for the base checkpoint reports both
`pretrained_file: None` and `finetune_from_model: None`. The base parameters
were therefore produced by the RNAcompass project's masked-language-model
training run rather than initialized by loading an upstream ERNIE-RNA
checkpoint. The distillation workflow subsequently saved the student model's
complete `state_dict` as the distilled artifact listed above.

On this basis, neither released parameter file is a copy of an upstream
pretrained checkpoint. Both parameter artifacts are released by their stated
rights holder, Ruobin Zhao, under `MODEL_LICENSE.md`.

## Third-party boundary

The network architecture and portions of the bundled runtime derive from the
ERNIE-RNA source-code project. That code is covered by its upstream MIT
license, preserved in `licenses/ERNIE-RNA-LICENSE.txt`, and is attributed in
`THIRD_PARTY_NOTICES.md`. Reusing an MIT-licensed implementation does not make
newly trained parameter values copies of the upstream checkpoint, but the
software notice must remain with redistributed copies of the derived code.

The distilled parameters may reflect predictions or representations produced
by teacher models during training even though no teacher checkpoint is bundled
in this release. Before public distribution, the project must retain a license
matrix confirming that each teacher model's terms permitted the actual
distillation use and do not impose incompatible conditions on the resulting
student parameters. This is independent of the resolved question of whether
the RNAcompass base file is a copy of the upstream ERNIE-RNA checkpoint.

## Internal records to retain

The public repository should not contain private infrastructure paths or
training data. The project should retain the immutable training configuration,
logs, input-data provenance, distillation configuration, artifact hashes and
the rights holder's release approval in its internal archive. The archive
should also contain the teacher-model license matrix and a snapshot of each
applicable license. These records support the provenance statement above if
questions arise after publication.
