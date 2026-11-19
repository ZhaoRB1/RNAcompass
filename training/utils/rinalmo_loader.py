from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


import os
from utils.local_paths import DATA_ROOT
DEFAULT_MODEL_ROOT = DATA_ROOT / "training/models/rinalmo_giga"
DEFAULT_SOURCE_ROOT = Path(os.environ.get("RNACOMPASS_RINALMO_SOURCE_ROOT", DEFAULT_MODEL_ROOT / "RiNALMo-1.0"))
DEFAULT_CHECKPOINT_PATH = DEFAULT_MODEL_ROOT / "rinalmo_giga_pretrained.pt"
DEFAULT_LM_CONFIG = "giga"
RNA_BASES = ("A", "C", "G", "U")


@dataclass
class LoadedRiNALMo:
    model: torch.nn.Module
    alphabet: object
    source_root: Path
    checkpoint_path: Path
    lm_config: str
    acgu_token_ids: list[int]
    torch_dtype: torch.dtype
    max_raw_sequence_length: int


def _prepend_sys_path(path: Path) -> None:
    path_str = str(Path(path).resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def ensure_source_root(source_root: Path) -> Path:
    source_root = Path(source_root).resolve()
    required = (
        "rinalmo/config.py",
        "rinalmo/model/model.py",
        "rinalmo/data/alphabet.py",
        "rinalmo/model/attention.py",
    )
    missing = [relative for relative in required if not (source_root / relative).exists()]
    if missing:
        raise FileNotFoundError(
            f"RiNALMo source root is incomplete: {source_root}. Missing: {', '.join(missing)}"
        )
    return source_root


def ensure_checkpoint_path(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RiNALMo checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _dependency_status() -> dict[str, bool]:
    return {
        "ml_collections": importlib.util.find_spec("ml_collections") is not None,
        "flash_attn": importlib.util.find_spec("flash_attn") is not None,
    }


def preflight_environment(device: str) -> None:
    missing = [name for name, ok in _dependency_status().items() if not ok]
    if missing:
        raise ImportError(
            "RiNALMo runtime dependencies are missing: "
            f"{', '.join(missing)}. Run with the configured rinalmo environment."
        )

    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        raise RuntimeError(
            "RiNALMo giga inference requires a CUDA device in this pipeline because the official "
            "checkpoint uses flash-attn modules. Use --device cuda:N with a CUDA-visible RiNALMo "
            "environment."
        )
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment, but a CUDA device was requested. "
            "Use the working RiNALMo Python via PYTHON_BIN=/path/to/python."
        )


def _resolve_torch_dtype(device: str, dtype: str = "auto") -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "float32":
        return torch.float32
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype != "auto":
        raise ValueError(f"Unsupported dtype: {dtype}")

    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def validate_checkpoint_state(state: Any, checkpoint_path: Path) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise ValueError(f"RiNALMo checkpoint is not a state_dict mapping: {checkpoint_path}")
    if "model" in state and "embedding.weight" not in state:
        raise ValueError(
            f"Expected a bare RiNALMo state_dict, but {checkpoint_path} looks like a wrapped checkpoint."
        )
    if "embedding.weight" not in state:
        raise ValueError(f"Expected a bare RiNALMo state_dict with embedding.weight: {checkpoint_path}")
    return state


def resolve_acgu_indices(alphabet: object) -> list[int]:
    get_idx = getattr(alphabet, "get_idx")
    return [int(get_idx("A")), int(get_idx("C")), int(get_idx("G")), int(get_idx("T"))]


def _run_forward_preflight(loaded: LoadedRiNALMo, device: str, use_amp: bool = True) -> None:
    tokens = torch.tensor(
        loaded.alphabet.batch_tokenize(["ACGU"]),
        dtype=torch.int64,
        device=device,
    )
    enabled = bool(use_amp and torch.device(device).type == "cuda" and loaded.torch_dtype != torch.float32)
    with torch.no_grad():
        if enabled:
            with torch.amp.autocast(device_type="cuda", dtype=loaded.torch_dtype, enabled=True):
                outputs = loaded.model(tokens)
        else:
            outputs = loaded.model(tokens)
    logits = outputs.get("logits")
    if logits is None or logits.shape[1] != 6:
        raise RuntimeError(f"RiNALMo preflight forward returned unexpected logits shape: {getattr(logits, 'shape', None)}")


def load_rinalmo_model(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    lm_config: str = DEFAULT_LM_CONFIG,
    device: str = "cuda:0",
    dtype: str = "auto",
    max_raw_sequence_length: int = 4096,
    run_forward_check: bool = True,
) -> LoadedRiNALMo:
    source_root = ensure_source_root(source_root)
    checkpoint_path = ensure_checkpoint_path(checkpoint_path)
    preflight_environment(device)
    _prepend_sys_path(source_root)

    from rinalmo.config import model_config
    from rinalmo.data.alphabet import Alphabet
    from rinalmo.model.model import RiNALMo

    config = model_config(lm_config)
    model = RiNALMo(config)
    alphabet = Alphabet(**config["alphabet"])

    state = validate_checkpoint_state(torch.load(str(checkpoint_path), map_location="cpu"), checkpoint_path)
    load_result = model.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"RiNALMo strict load failed: {load_result}")

    model.to(device)
    model.eval()
    torch_dtype = _resolve_torch_dtype(device, dtype)
    loaded = LoadedRiNALMo(
        model=model,
        alphabet=alphabet,
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        lm_config=lm_config,
        acgu_token_ids=resolve_acgu_indices(alphabet),
        torch_dtype=torch_dtype,
        max_raw_sequence_length=int(max_raw_sequence_length),
    )
    if run_forward_check:
        _run_forward_preflight(loaded, device=device)
    return loaded
