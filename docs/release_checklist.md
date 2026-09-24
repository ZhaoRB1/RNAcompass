# Release checklist

1. Have Ruobin Zhao approve both `LICENSE` and `MODEL_LICENSE.md` as the stated
   software copyright and model-rights holder; retain that approval internally.
2. Confirm that every teacher model used in distillation permitted the actual
   use and does not impose incompatible conditions on the student parameters;
   retain a dated license snapshot and license matrix.
3. Confirm with Shandong University that no employment, funding, collaboration,
   dataset or patent obligation supersedes the stated individual ownership.
4. Retain the training records showing that the base run used no external
   pretrained or fine-tuning checkpoint, and that the distilled file was
   generated from the RNAcompass-trained student model.
5. Build the exact two-checkpoint archive described in `weights.md`.
6. Upload the archive to a durable host that accepts files of this size, and
   add its permanent URL to the README and release notes.
7. When the GitHub repository is public, add its canonical URL to
   `CITATION.cff` and `pyproject.toml`; add the paper DOI only after assignment.
8. Create a clean Conda environment from `environment.yml`.
9. Run `pytest` and the GPU validation in `reproducibility.md`.
10. Run `python scripts/check_release_tree.py --release`.
11. Build and inspect both wheel and source distribution.

Do not publish if any step fails.
