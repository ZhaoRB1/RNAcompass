from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import torch


from utils.local_paths import DATA_ROOT
DEFAULT_MODEL_ROOT = DATA_ROOT / "training/models/aidorna"
DEFAULT_SOURCE_ROOT = Path(os.environ.get("RNACOMPASS_AIDO_SOURCE_ROOT", DEFAULT_MODEL_ROOT / "ModelGenerator"))
DEFAULT_LOCAL_HF_DIR = Path(os.environ.get("RNACOMPASS_AIDO_CACHE", "/tmp/rnacompass-aidorna-hf"))
RNA_BASES = ("A", "C", "G", "U")
SHARD_NAMES = (
    "pytorch_model-00001-of-00002.safetensors",
    "pytorch_model-00002-of-00002.safetensors",
)
OFFICIAL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "pytorch_model.bin.index.json",
)
BIN_TO_SAFETENSORS_SHARDS = {
    "pytorch_model-00001-of-00002.bin": "pytorch_model-00001-of-00002.safetensors",
    "pytorch_model-00002-of-00002.bin": "pytorch_model-00002-of-00002.safetensors",
}


@dataclass
class LoadedAIDORNA:
    model: torch.nn.Module
    tokenizer: object
    model_root: Path
    source_root: Path
    local_hf_dir: Path
    max_length: int
    max_raw_sequence_length: int
    base_token_ids: Dict[str, int]
    torch_dtype: torch.dtype

    @property
    def acgu_token_ids(self) -> list[int]:
        return [self.base_token_ids[base] for base in RNA_BASES]


def _prepend_sys_path(path: Path) -> None:
    path_str = str(Path(path).resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _source_vocab_file(source_root: Path) -> Path:
    return Path(source_root) / "modelgenerator" / "huggingface_models" / "rnabert" / "vocab.txt"


def ensure_source_root(source_root: Path) -> Path:
    source_root = Path(source_root).resolve()
    required = (
        "modelgenerator/__init__.py",
        "modelgenerator/huggingface_models/rnabert/configuration_rnabert.py",
        "modelgenerator/huggingface_models/rnabert/modeling_rnabert.py",
        "modelgenerator/huggingface_models/rnabert/tokenization_rnabert.py",
        "modelgenerator/huggingface_models/rnabert/vocab.txt",
    )
    missing = [relative for relative in required if not (source_root / relative).exists()]
    if missing:
        raise FileNotFoundError(
            f"ModelGenerator source root is incomplete: {source_root}. Missing: {', '.join(missing)}"
        )
    return source_root


def ensure_model_root(model_root: Path) -> Path:
    model_root = Path(model_root).resolve()
    missing = [name for name in SHARD_NAMES if not (model_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"AIDO.RNA model root is incomplete: {model_root}. Missing: {', '.join(missing)}"
        )
    return model_root


def official_metadata_available(model_root: Path) -> bool:
    model_root = Path(model_root)
    return all((model_root / name).exists() for name in OFFICIAL_METADATA_FILES)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and Path(os.readlink(dst)) == src:
            return
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _read_safetensors_keys(shard_path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
        return list(handle.keys())


def build_weight_map(
    *,
    model_root: Path,
    weight_keys_by_shard: Optional[Mapping[str, Iterable[str]]] = None,
) -> Dict[str, str]:
    model_root = ensure_model_root(model_root)
    weight_map: Dict[str, str] = {}

    for shard_name in SHARD_NAMES:
        keys = (
            list(weight_keys_by_shard[shard_name])
            if weight_keys_by_shard is not None
            else _read_safetensors_keys(model_root / shard_name)
        )
        for key in keys:
            if key in weight_map:
                raise ValueError(f"Duplicate safetensors key across shards: {key}")
            weight_map[key] = shard_name

    if not weight_map:
        raise ValueError("No weights found in AIDO.RNA safetensors shards.")
    return weight_map


def _read_official_index(model_root: Path) -> Optional[dict]:
    index_file = Path(model_root) / "pytorch_model.bin.index.json"
    if not index_file.exists():
        return None
    with open(index_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_safetensors_weight_map(
    *,
    model_root: Path,
    weight_keys_by_shard: Optional[Mapping[str, Iterable[str]]] = None,
) -> Dict[str, str]:
    actual_weight_map = build_weight_map(
        model_root=model_root,
        weight_keys_by_shard=weight_keys_by_shard,
    )
    actual_keys = set(actual_weight_map)
    official_index = _read_official_index(model_root)
    if official_index is None:
        return actual_weight_map

    translated: Dict[str, str] = {}
    official_weight_map = official_index.get("weight_map", {})
    if not isinstance(official_weight_map, dict):
        raise ValueError(f"Official index has invalid weight_map: {Path(model_root) / 'pytorch_model.bin.index.json'}")

    for key, shard_name in official_weight_map.items():
        if key not in actual_keys:
            continue
        safe_shard_name = BIN_TO_SAFETENSORS_SHARDS.get(str(shard_name))
        if safe_shard_name is None:
            raise ValueError(f"Cannot map official shard {shard_name!r} to a local safetensors shard.")
        translated[key] = safe_shard_name

    for key, shard_name in actual_weight_map.items():
        translated.setdefault(key, shard_name)
    return translated


def _config_payload() -> dict:
    return {
        "architectures": ["RNABertForMaskedLM"],
        "model_type": "rnabert",
        "vocab_size": 16,
        "hidden_size": 2048,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "intermediate_size": 5440,
        "hidden_act": "swiglu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 1024,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 0,
        "add_linear_bias": True,
        "position_embedding_type": "rope",
        "normalization_type": "LayerNorm",
        "rotary_percent": 1.0,
        "use_cache": True,
        "torch_dtype": "float32",
        "tie_word_embeddings": True,
    }


def prepare_local_hf_dir(
    *,
    model_root: Path = DEFAULT_MODEL_ROOT,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    local_hf_dir: Path = DEFAULT_LOCAL_HF_DIR,
    weight_keys_by_shard: Optional[Mapping[str, Iterable[str]]] = None,
) -> Path:
    model_root = ensure_model_root(model_root)
    source_root = ensure_source_root(source_root)
    local_hf_dir = Path(local_hf_dir).resolve()
    local_hf_dir.mkdir(parents=True, exist_ok=True)

    for shard_name in SHARD_NAMES:
        _link_or_copy(model_root / shard_name, local_hf_dir / shard_name)
    _link_or_copy(_source_vocab_file(source_root), local_hf_dir / "vocab.txt")

    if official_metadata_available(model_root):
        for metadata_file in OFFICIAL_METADATA_FILES:
            _link_or_copy(model_root / metadata_file, local_hf_dir / metadata_file)
    else:
        config_file = local_hf_dir / "config.json"
        config_file.write_text(json.dumps(_config_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    weight_map = build_safetensors_weight_map(model_root=model_root, weight_keys_by_shard=weight_keys_by_shard)
    total_size = sum((model_root / shard_name).stat().st_size for shard_name in SHARD_NAMES)
    index_payload = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (local_hf_dir / "model.safetensors.index.json").write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tokenizer_config = {
        "model_max_length": 1024,
        "tokenizer_class": "RNABertTokenizer",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "sep_token": "[SEP]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }
    (local_hf_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    special_tokens = {
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "pad_token": "[PAD]",
        "mask_token": "[MASK]",
        "sep_token": "[SEP]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }
    (local_hf_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return local_hf_dir


def load_rnabert_tokenizer(source_root: Path = DEFAULT_SOURCE_ROOT):
    source_root = ensure_source_root(source_root)
    _prepend_sys_path(source_root)
    from modelgenerator.huggingface_models.rnabert import RNABertTokenizer

    return RNABertTokenizer(str(_source_vocab_file(source_root)), version="v2")


def validate_base_token_ids(tokenizer) -> Dict[str, int]:
    expected = {"A": 5, "C": 7, "G": 6, "U": 9}
    token_ids: Dict[str, int] = {}
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for base in RNA_BASES:
        token_id = tokenizer.convert_tokens_to_ids(base)
        if token_id is None or token_id == unk_id:
            raise ValueError(f"Tokenizer cannot represent RNA base {base!r}.")
        token_ids[base] = int(token_id)
    if token_ids != expected:
        raise ValueError(f"Unexpected AIDO.RNA token ids: {token_ids}; expected {expected}.")
    return token_ids


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


def repair_tied_mlm_head(model: torch.nn.Module) -> None:
    """Restore RNABert MLM decoder aliases after low-memory loading.

    The AIDO.RNA checkpoint stores ``cls.predictions.bias`` and relies on the
    decoder bias alias created by ``RNABertLMPredictionHead``. With
    ``low_cpu_mem_usage=True``, Transformers can leave the alias parameter on
    the meta device. Re-linking it before ``model.to(device)`` prevents the
    meta tensor copy failure while preserving the original checkpoint value.
    """
    predictions = getattr(getattr(model, "cls", None), "predictions", None)
    decoder = getattr(predictions, "decoder", None)
    bias = getattr(predictions, "bias", None)
    decoder_bias = getattr(decoder, "bias", None)
    if predictions is None or decoder is None or bias is None or decoder_bias is None:
        return
    if getattr(decoder_bias, "is_meta", False) or decoder_bias is not bias:
        decoder.bias = bias


def find_meta_parameters(model: torch.nn.Module) -> list[str]:
    return [name for name, param in model.named_parameters() if getattr(param, "is_meta", False)]


def load_aidorna_model(
    *,
    model_root: Path = DEFAULT_MODEL_ROOT,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    local_hf_dir: Path = DEFAULT_LOCAL_HF_DIR,
    device: str = "cuda:0",
    dtype: str = "auto",
) -> LoadedAIDORNA:
    source_root = ensure_source_root(source_root)
    model_root = ensure_model_root(model_root)
    local_hf_dir = prepare_local_hf_dir(
        model_root=model_root,
        source_root=source_root,
        local_hf_dir=local_hf_dir,
    )
    _prepend_sys_path(source_root)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:
        pass

    from modelgenerator.huggingface_models.rnabert import RNABertForMaskedLM

    tokenizer = load_rnabert_tokenizer(source_root)
    base_token_ids = validate_base_token_ids(tokenizer)
    torch_dtype = _resolve_torch_dtype(device, dtype)

    model = RNABertForMaskedLM.from_pretrained(
        str(local_hf_dir),
        local_files_only=True,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
    )
    repair_tied_mlm_head(model)
    meta_parameters = find_meta_parameters(model)
    if meta_parameters:
        preview = ", ".join(meta_parameters[:10])
        suffix = "" if len(meta_parameters) <= 10 else f", ... ({len(meta_parameters)} total)"
        raise RuntimeError(f"AIDO.RNA checkpoint left parameters on meta device: {preview}{suffix}")
    model.to(device)
    model.eval()

    max_length = int(model.config.max_position_embeddings)
    return LoadedAIDORNA(
        model=model,
        tokenizer=tokenizer,
        model_root=model_root,
        source_root=source_root,
        local_hf_dir=local_hf_dir,
        max_length=max_length,
        max_raw_sequence_length=max_length - 2,
        base_token_ids=base_token_ids,
        torch_dtype=torch_dtype,
    )
