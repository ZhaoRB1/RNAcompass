"""ERNIE-RNA-giga raw dependency-map inference used by the public model."""

from __future__ import annotations

import math
from contextlib import nullcontext
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .distilled_checkpoint import load_distilled_checkpoint
from .fasta import SequenceRecord
from .weights import WEIGHT_SPECS, weight_path


RNA_BASES = ("A", "C", "G", "U")
NUC_TO_TOKEN = {"A": 5, "C": 7, "G": 4, "U": 6, "T": 6}
ERNIE_ACGU_IDXS = [5, 7, 4, 6]


def _paired_score(x: int, y: int, lamda: float = 0.8) -> float:
    if (x, y) in {(5, 6), (6, 5)}:
        return 2.0
    if (x, y) in {(4, 7), (7, 4)}:
        return 3.0
    if (x, y) in {(4, 6), (6, 4)}:
        return lamda
    return 0.0


PAIR_MAP = np.array(
    [[_paired_score(i, j) for i in range(30)] for j in range(30)],
    dtype=np.float32,
)


def patch_omegaconf_get_ref_type() -> None:
    """Apply only the compatibility shim required by Fairseq 0.12.2."""
    try:
        import omegaconf._utils as oc_utils
    except Exception:
        return
    if hasattr(oc_utils, "get_ref_type"):
        return

    def get_ref_type(obj, key=None):
        try:
            if key is not None and hasattr(obj, "_get_node"):
                node = obj._get_node(key)
                ref_type = getattr(getattr(node, "_metadata", None), "ref_type", None)
                if ref_type is not None:
                    return ref_type
            if hasattr(oc_utils, "get_type_of"):
                value = oc_utils.get_type_of(obj)
                if value is not None:
                    return value
        except Exception:
            pass
        return Any

    oc_utils.get_ref_type = get_ref_type


def _validate_device(device: str) -> tuple[str, int]:
    import torch

    if device.isdigit():
        device = f"cuda:{device}"
    if not device.startswith("cuda:"):
        raise ValueError("RNAcompass v1 inference requires an NVIDIA CUDA device such as cuda:0")
    try:
        index = int(device.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"invalid CUDA device: {device!r}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the active RNAcompass environment")
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {index} is unavailable; detected {torch.cuda.device_count()} device(s)")
    return device, index


def load_encoder(weights_dir: Path, device: str, log: Callable[[str], None] = print):
    import torch

    device, index = _validate_device(device)
    torch.cuda.set_device(index)
    patch_omegaconf_get_ref_type()
    from ._vendor.ernie_project.src.utils import load_pretrained_ernierna

    base_checkpoint = weight_path(weights_dir, WEIGHT_SPECS[0])
    distilled_checkpoint = weight_path(weights_dir, WEIGHT_SPECS[1])
    dictionary = resources.files("rnacompass._vendor.ernie_project.src").joinpath("dict")
    with resources.as_file(dictionary) as dict_dir:
        pretrained = load_pretrained_ernierna(str(base_checkpoint), {"data": str(dict_dir)})
    encoder = pretrained.encoder.to(device)
    load_distilled_checkpoint(encoder, distilled_checkpoint, logger=lambda message: log(f"[checkpoint] {message}"))
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def sequence_to_tokens(sequence: str) -> np.ndarray:
    tokens = np.ones(len(sequence) + 2, dtype=np.int64)
    tokens[0] = 0
    tokens[-1] = 2
    for index, base in enumerate(sequence):
        tokens[index + 1] = NUC_TO_TOKEN.get(base, 3)
    return tokens


def mutation_specs(sequence: str) -> Iterable[tuple[int, str]]:
    for position, reference in enumerate(sequence):
        if reference not in RNA_BASES:
            continue
        for alternative in RNA_BASES:
            if alternative != reference:
                yield position, alternative


def build_mutation_batch(reference: np.ndarray, specs: Sequence[tuple[int, str]]) -> np.ndarray:
    batch = np.tile(reference, (len(specs), 1))
    for row, (position, alternative) in enumerate(specs):
        batch[row, position + 1] = NUC_TO_TOKEN[alternative]
    return batch


def _autocast(device: str, enabled: bool):
    import torch

    if not enabled:
        return nullcontext()
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def _probabilities(model, token_batch: np.ndarray, sequence_length: int, device: str, use_amp: bool, epsilon: float):
    import torch

    one_d = torch.from_numpy(token_batch).long().to(device)
    batch_size, token_length = one_d.shape
    two_d = PAIR_MAP[token_batch[:, :, None], token_batch[:, None, :]]
    two_d = torch.from_numpy(np.ascontiguousarray(two_d[..., None], dtype=np.float32)).to(device)
    mask = torch.ones((batch_size, token_length), dtype=torch.bool, device=device)
    with _autocast(device, use_amp):
        logits, _, _ = model(
            src_tokens=one_d,
            twod_tokens=two_d,
            is_twod=True,
            masked_tokens=mask,
            extra_only=False,
            masked_only=True,
        )
    if logits is None:
        raise RuntimeError("ERNIE-RNA returned logits=None")
    if logits.dim() == 2:
        logits = logits.view(batch_size, token_length, -1)
    selected = logits[:, 1 : sequence_length + 1, ERNIE_ACGU_IDXS]
    probabilities = torch.softmax(selected, dim=-1).float() + epsilon
    return probabilities / probabilities.sum(dim=-1, keepdim=True)


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(value in text for value in ("out of memory", "memory allocation"))


def compute_raw_dependency_map(
    sequence: str,
    model,
    *,
    device: str,
    batch_size: int,
    use_amp: bool = True,
    epsilon: float = 1e-10,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, int]:
    import torch

    reference = sequence_to_tokens(sequence)
    specs = list(mutation_specs(sequence))
    current_batch_size = max(1, int(batch_size))
    with torch.inference_mode():
        reference_probs = _probabilities(model, reference.reshape(1, -1), len(sequence), device, use_amp, epsilon)[0]
        reference_odds = torch.log2(reference_probs + epsilon) - torch.log2(1.0 - reference_probs + epsilon)
        raw = np.zeros((len(sequence), len(sequence)), dtype=np.float32)
        processed = 0
        while processed < len(specs):
            selected_specs = specs[processed : processed + current_batch_size]
            token_batch = build_mutation_batch(reference, selected_specs)
            try:
                mutated_probs = _probabilities(model, token_batch, len(sequence), device, use_amp, epsilon)
            except RuntimeError as exc:
                if _is_oom(exc) and current_batch_size > 1:
                    old = current_batch_size
                    current_batch_size = max(1, current_batch_size // 2)
                    torch.cuda.empty_cache()
                    log(f"[inference] CUDA OOM at batch_size={old}; retrying with {current_batch_size}")
                    continue
                raise
            mutated_odds = torch.log2(mutated_probs + epsilon) - torch.log2(1.0 - mutated_probs + epsilon)
            scores = torch.amax(torch.abs(mutated_odds - reference_odds.unsqueeze(0)), dim=-1).cpu().numpy()
            for row, (position, _) in enumerate(selected_specs):
                np.maximum(raw[position], scores[row], out=raw[position])
            processed += len(selected_specs)
        np.fill_diagonal(raw, 0.0)
    return raw, current_batch_size


def generate_raw_dependency_maps(
    records: Sequence[SequenceRecord],
    *,
    weights_dir: Path,
    device: str,
    batch_size: int,
    log: Callable[[str], None] = print,
) -> Iterable[tuple[SequenceRecord, np.ndarray, int]]:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    model = load_encoder(weights_dir, device, log=log)
    for record in records:
        log(f"[inference] {record.sample_id}: L={len(record.sequence)}")
        matrix, final_batch_size = compute_raw_dependency_map(
            record.sequence,
            model,
            device=device,
            batch_size=batch_size,
            log=log,
        )
        yield record, matrix, final_batch_size

