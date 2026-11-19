import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RNA_BASES = ("A", "C", "G", "U")


class TensorBatch(dict):
    def to(self, device):
        return TensorBatch(
            {key: value.to(device) if torch.is_tensor(value) else value for key, value in self.items()}
        )



class SimpleRnaTokenizer:
    def __init__(
        self,
        base_to_id=None,
        output_base_ids=None,
        add_special_tokens=True,
        bos_token_id=4,
        eos_token_id=5,
        pad_token_id=6,
        unk_token_id=7,
    ):
        self.base_to_id = base_to_id or {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
        self.output_base_ids = output_base_ids or [self.base_to_id[base] for base in RNA_BASES]
        self.add_special_tokens_default = bool(add_special_tokens)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.unk_token_id = int(unk_token_id)
        self.id_to_base = {int(v): k for k, v in self.base_to_id.items() if k in RNA_BASES}
        self.id_to_base[int(self.base_to_id.get("T", self.base_to_id.get("U", 3)))] = "U"

    def __len__(self):
        ids = list(self.base_to_id.values()) + list(self.output_base_ids)
        ids += [self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id]
        return max(int(token_id) for token_id in ids) + 1

    def convert_tokens_to_ids(self, token):
        token = str(token).upper().replace("T", "U")
        if token in RNA_BASES:
            return int(self.output_base_ids[RNA_BASES.index(token)])
        return self.unk_token_id

    def sequence_from_ids(self, input_ids, attention_mask=None):
        ids = input_ids.detach().cpu().tolist()
        if attention_mask is not None:
            keep = attention_mask.detach().cpu().tolist()
            ids = [token_id for token_id, flag in zip(ids, keep) if int(flag)]
        bases = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in {self.bos_token_id, self.eos_token_id, self.pad_token_id}:
                continue
            bases.append(self.id_to_base.get(token_id, "N"))
        return "".join(bases)

    def __call__(
        self,
        sequences,
        truncation=True,
        padding="longest",
        max_length=None,
        return_tensors="pt",
        add_special_tokens=None,
        **_,
    ):
        add_special = self.add_special_tokens_default if add_special_tokens is None else bool(add_special_tokens)
        rows = []
        for sequence in sequences:
            seq = str(sequence).replace(" ", "").upper().replace("T", "U")
            token_ids = [int(self.base_to_id.get(base, self.unk_token_id)) for base in seq]
            if add_special:
                token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
            if truncation and max_length is not None:
                token_ids = token_ids[: int(max_length)]
            rows.append(token_ids)
        width = max(len(row) for row in rows) if rows else 0
        input_ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for idx, row in enumerate(rows):
            input_ids[idx, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[idx, : len(row)] = 1
        return TensorBatch({"input_ids": input_ids, "attention_mask": attention_mask})



class RiNALMoTokenizerAdapter:
    def __init__(self, alphabet, base_token_ids):
        self.alphabet = alphabet
        self.base_token_ids = list(base_token_ids)
        self.pad_token_id = int(getattr(alphabet, "padding_idx", getattr(alphabet, "pad_idx", 0)))
        self.unk_token_id = None

    def __len__(self):
        return int(len(self.alphabet))

    def convert_tokens_to_ids(self, token):
        token = str(token).upper()
        if token == "U":
            token = "T"
        if token in {"A", "C", "G", "T"}:
            base = "U" if token == "T" else token
            return int(self.base_token_ids[RNA_BASES.index(base)])
        return int(self.alphabet.get_idx(token))

    def __call__(self, sequences, truncation=True, padding="longest", max_length=None, return_tensors="pt", **_):
        seqs = [str(seq).replace(" ", "").upper().replace("U", "T") for seq in sequences]
        tokens = torch.as_tensor(self.alphabet.batch_tokenize(seqs), dtype=torch.long)
        if truncation and max_length is not None:
            tokens = tokens[:, : int(max_length)]
        attention_mask = tokens.ne(self.pad_token_id).long()
        return TensorBatch({"input_ids": tokens, "attention_mask": attention_mask})



class DeviceModule(nn.Module):
    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")



class DictLogitsAdapter(DeviceModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_input_embeddings(self):
        if hasattr(self.model, "get_input_embeddings"):
            return self.model.get_input_embeddings()
        for path in (
            "embedding",
            "embed_tokens",
            "encoder.embed_tokens",
            "encoder.sentence_encoder.embed_tokens",
            "decoder.embed_tokens",
            "decoder.sentence_encoder.embed_tokens",
            "encoder.backbone.embed_tokens",
        ):
            module = _resolve_attr(self.model, path)
            if isinstance(module, nn.Module) and hasattr(module, "weight"):
                return module
        return None



class RiNALMoModelAdapter(DictLogitsAdapter):
    def __init__(self, model, torch_dtype=torch.float32):
        super().__init__(model)
        self.torch_dtype = torch_dtype

    def forward(self, input_ids, attention_mask=None, **_):
        device_type = input_ids.device.type
        use_amp = device_type == "cuda" and self.torch_dtype != torch.float32
        with torch.amp.autocast(device_type=device_type, dtype=self.torch_dtype, enabled=use_amp):
            outputs = self.model(input_ids)
        return SimpleNamespace(logits=outputs["logits"])



class NamedParameterGroup(nn.Module):
    """Expose a non-contiguous set of existing parameters as one trainable head."""

    def __init__(self, named_params):
        super().__init__()
        self._named_param_refs = [(str(name), param) for name, param in named_params if param is not None]

    def parameters(self, recurse=True):
        seen = set()
        for _, param in self._named_param_refs:
            if id(param) in seen:
                continue
            seen.add(id(param))
            yield param

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        seen = set()
        for name, param in self._named_param_refs:
            if remove_duplicate and id(param) in seen:
                continue
            seen.add(id(param))
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param



class ErnieRnaModelAdapter(DeviceModule):
    def __init__(self, model_encoder, seq_to_index, prepare_input_for_ernierna, tokenizer):
        super().__init__()
        self.model = model_encoder
        self.encoder = model_encoder
        self.seq_to_index = seq_to_index
        self.prepare_input_for_ernierna = prepare_input_for_ernierna
        self.tokenizer = tokenizer
        self.lm_head = self._build_lm_head_group(model_encoder)

    @staticmethod
    def _build_lm_head_group(model_encoder):
        named_params = []
        head_modules = (
            ("lm_head_transform_weight", getattr(model_encoder, "lm_head_transform_weight", None)),
            ("layer_norm", getattr(model_encoder, "layer_norm", None)),
            ("embed_out", getattr(model_encoder, "embed_out", None)),
        )
        for prefix, module in head_modules:
            if isinstance(module, nn.Module):
                named_params.extend((f"{prefix}.{name}", param) for name, param in module.named_parameters())
        bias = getattr(model_encoder, "lm_output_learned_bias", None)
        if isinstance(bias, nn.Parameter):
            named_params.append(("lm_output_learned_bias", bias))
        return NamedParameterGroup(named_params)

    def get_input_embeddings(self):
        for path in (
            "sentence_encoder.embed_tokens",
            "encoder.sentence_encoder.embed_tokens",
            "embed_tokens",
        ):
            module = _resolve_attr(self.model, path)
            if isinstance(module, nn.Module) and hasattr(module, "weight"):
                return module
        return None

    def forward(self, input_ids, attention_mask=None, **_):
        max_len = int(input_ids.shape[1])
        sequences = []
        for row_idx in range(int(input_ids.shape[0])):
            seq = self.tokenizer.sequence_from_ids(
                input_ids[row_idx],
                attention_mask[row_idx] if attention_mask is not None else None,
            )
            sequences.append(seq)

        grouped = {}
        for row_idx, seq in enumerate(sequences):
            grouped.setdefault(len(seq), []).append(row_idx)

        output_rows = [None for _ in sequences]
        vocab_size = None
        for cur_len, row_indices in grouped.items():
            batch_seqs = [sequences[row_idx] for row_idx in row_indices]
            rna_index, rna_len_lst = self.seq_to_index(batch_seqs)
            one_d_tensors = []
            two_d_tensors = []
            for index, seq_len_item in zip(rna_index, rna_len_lst):
                if int(seq_len_item) != int(cur_len):
                    raise ValueError("ERNIE-RNA adapter received inconsistent same-length batch.")
                one_d, two_d = self.prepare_input_for_ernierna(index, cur_len)
                one_d_tensors.append(one_d)
                two_d_tensors.append(two_d)
            one_d = torch.cat(one_d_tensors, dim=0).to(input_ids.device)
            two_d = torch.cat(two_d_tensors, dim=0).to(device=input_ids.device, dtype=torch.float32)
            all_tokens_mask = torch.ones((one_d.shape[0], one_d.shape[1]), dtype=torch.bool, device=input_ids.device)
            logits, _, _ = self.model(
                src_tokens=one_d,
                twod_tokens=two_d,
                is_twod=True,
                masked_tokens=all_tokens_mask,
                extra_only=False,
                masked_only=False,
            )
            if logits.dim() == 2:
                logits = logits.view(one_d.shape[0], one_d.shape[1], -1)
            vocab_size = int(logits.shape[-1]) if vocab_size is None else vocab_size
            for local_idx, row_idx in enumerate(row_indices):
                padded = logits.new_zeros((max_len, vocab_size))
                copy_len = min(max_len, int(logits.shape[1]))
                padded[:copy_len] = logits[local_idx, :copy_len]
                output_rows[row_idx] = padded
        return SimpleNamespace(logits=torch.stack(output_rows, dim=0))



def _hidden_size(model):
    config = getattr(model, "config", None)
    if config is not None:
        for name in ("hidden_size", "d_model", "n_embd", "embed_dim", "embedding_dim"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return int(module.embedding_dim)
    return None



def _validate_base_token_ids(base_ids, model_key):
    if base_ids is None:
        raise ValueError(f"{model_key}: missing ci_params.base_token_ids for A/C/G/U.")
    base_ids = [int(token_id) for token_id in base_ids]
    if len(base_ids) != len(RNA_BASES):
        raise ValueError(f"{model_key}: base_token_ids must contain exactly A/C/G/U ids, got {base_ids}.")
    if any(token_id < 0 for token_id in base_ids):
        raise ValueError(f"{model_key}: base_token_ids must be non-negative, got {base_ids}.")
    if len(set(base_ids)) != len(base_ids):
        raise ValueError(f"{model_key}: A/C/G/U output token ids must be distinct, got {base_ids}.")
    return base_ids



def _tokenizer_base_id(tokenizer, base):
    if not hasattr(tokenizer, "convert_tokens_to_ids"):
        return None
    candidates = [base]
    if base == "U":
        candidates.append("T")
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for candidate in candidates:
        token_id = tokenizer.convert_tokens_to_ids(candidate)
        if isinstance(token_id, list):
            if len(token_id) != 1:
                continue
            token_id = token_id[0]
        if token_id is None:
            continue
        token_id = int(token_id)
        if unk_id is not None and token_id == int(unk_id):
            continue
        if token_id >= 0:
            return token_id
    return None



def _infer_vocab_size(model, tokenizer):
    for obj in (getattr(model, "config", None), getattr(getattr(model, "model", None), "config", None)):
        if obj is not None:
            vocab_size = getattr(obj, "vocab_size", None)
            if vocab_size is not None:
                return int(vocab_size)
    for attr in ("vocab_size",):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    try:
        return int(len(tokenizer))
    except Exception:
        return None



def validate_tokenizer_alignment(model_key, model, tokenizer, overrides, probe_sequence="ACGU"):
    model_cfg = overrides.get("model", {})
    ci_cfg = overrides.get("ci_params", {})
    base_ids = _validate_base_token_ids(ci_cfg.get("base_token_ids"), model_key)

    for base, expected_id in zip(RNA_BASES, base_ids):
        actual_id = _tokenizer_base_id(tokenizer, base)
        if actual_id is None:
            raise ValueError(f"{model_key}: tokenizer cannot resolve RNA base {base}.")
        if int(actual_id) != int(expected_id):
            raise ValueError(
                f"{model_key}: tokenizer id mismatch for {base}: "
                f"convert_tokens_to_ids={actual_id}, configured base_token_ids={expected_id}."
            )

    vocab_size = _infer_vocab_size(model, tokenizer)
    if vocab_size is not None and max(base_ids) >= vocab_size:
        raise ValueError(f"{model_key}: base_token_ids {base_ids} exceed vocab_size={vocab_size}.")

    tokenized = tokenizer(
        [probe_sequence],
        truncation=True,
        padding="longest",
        max_length=max(16, len(probe_sequence) + int(model_cfg.get("max_special_tokens", 2))),
        return_tensors="pt",
    )
    if "input_ids" not in tokenized:
        raise ValueError(f"{model_key}: tokenizer output does not contain input_ids.")
    token_len = int(tokenized["input_ids"].shape[1])
    for offset_name, offset_value in (
        ("token_offset", ci_cfg.get("token_offset", model_cfg.get("matrix_offset", 1))),
        ("matrix_offset", model_cfg.get("matrix_offset", 1)),
    ):
        offset = int(offset_value)
        if offset < 0:
            raise ValueError(f"{model_key}: {offset_name} must be non-negative, got {offset}.")
        if offset + len(probe_sequence) > token_len:
            raise ValueError(
                f"{model_key}: {offset_name}={offset} is incompatible with tokenized length {token_len} "
                f"for probe sequence {probe_sequence!r}."
            )

    return {
        "model_key": model_key,
        "base_token_ids": base_ids,
        "tokenized_probe_length": token_len,
        "token_offset": int(ci_cfg.get("token_offset", model_cfg.get("matrix_offset", 1))),
        "matrix_offset": int(model_cfg.get("matrix_offset", 1)),
        "vocab_size": vocab_size,
    }



def _attach_overrides(model, overrides):
    model.rcompass_config_overrides = overrides
    return model



def _resolve_attr(obj, dotted_path):
    cur = obj
    for part in dotted_path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur



def merge_overrides(configs, overrides):
    if not overrides:
        return configs
    model_overrides = overrides.get("model", {})
    ci_overrides = overrides.get("ci_params", {})
    lm_overrides = overrides.get("lm_params", {})
    if model_overrides:
        configs.setdefault("model", {}).update(model_overrides)
    if ci_overrides:
        configs.setdefault("model", {}).setdefault("ci_params", {}).update(ci_overrides)
    if lm_overrides:
        configs.setdefault("model", {}).setdefault("lm_params", {}).update(lm_overrides)
    return configs



def seq_to_index(sequences):
    '''
    input:
    sequences: list of string (difference length)

    return:
    rna_index: numpy matrix, shape like: [len(sequences), max_seq_len+2]
    rna_len_lst: list of length
    '''

    rna_len_lst = [len(ss) for ss in sequences]
    max_len = max(rna_len_lst)
    assert max_len <= 1022
    seq_nums = len(rna_len_lst)
    rna_index = np.ones((seq_nums,max_len+2))
    for i in range(seq_nums):
        for j in range(rna_len_lst[i]):
            if sequences[i][j] in set("Aa"):
                rna_index[i][j+1] = 5
            elif sequences[i][j] in set("Cc"):
                rna_index[i][j+1] = 7
            elif sequences[i][j] in set("Gg"):
                rna_index[i][j+1] = 4
            elif sequences[i][j] in set('TUtu'):
                rna_index[i][j+1] = 6
            else:
                rna_index[i][j+1] = 3
        rna_index[i][rna_len_lst[i]+1] = 2 # add 'eos' token
    rna_index[:,0] = 0 # add 'cls' token
    return rna_index, rna_len_lst


def _load_ernie(device, checkpoint):
    from importlib import resources
    from rnacompass.inference import patch_omegaconf_get_ref_type
    patch_omegaconf_get_ref_type()
    from rnacompass._vendor.ernie_project.src.utils import load_pretrained_ernierna, prepare_input_for_ernierna
    dictionary = resources.files('rnacompass._vendor.ernie_project.src').joinpath('dict')
    with resources.as_file(dictionary) as dict_dir:
        pretrained = load_pretrained_ernierna(str(checkpoint), {'data': str(dict_dir)})
    encoder = pretrained.encoder.to(device)
    encoder.eval()
    base_ids = [5, 7, 4, 6]
    tokenizer = SimpleRnaTokenizer(
        base_to_id={'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3},
        output_base_ids=base_ids, add_special_tokens=True,
        bos_token_id=4, eos_token_id=5, pad_token_id=6, unk_token_id=8,
    )
    model = ErnieRnaModelAdapter(encoder, seq_to_index, prepare_input_for_ernierna, tokenizer)
    overrides = {'model': {'matrix_offset': 1, 'max_special_tokens': 2},
                 'ci_params': {'token_offset': 1, 'base_token_ids': base_ids}}
    return model, tokenizer, _hidden_size(model), overrides


def _load_aidorna(device, checkpoint):
    from utils.aido_loader import load_aidorna_model
    loaded = load_aidorna_model(model_root=Path(checkpoint), device=str(device))
    overrides = {'model': {'matrix_offset': 1, 'max_special_tokens': 2},
                 'ci_params': {'token_offset': 1, 'base_token_ids': loaded.acgu_token_ids}}
    return loaded.model, loaded.tokenizer, _hidden_size(loaded.model), overrides


def _load_rinalmo(device, checkpoint):
    from utils.rinalmo_loader import load_rinalmo_model
    loaded = load_rinalmo_model(checkpoint_path=Path(checkpoint), device=str(device))
    model = RiNALMoModelAdapter(loaded.model, torch_dtype=loaded.torch_dtype)
    tokenizer = RiNALMoTokenizerAdapter(loaded.alphabet, loaded.acgu_token_ids)
    overrides = {'model': {'matrix_offset': 1, 'max_special_tokens': 2},
                 'ci_params': {'token_offset': 1, 'base_token_ids': loaded.acgu_token_ids}}
    return model, tokenizer, _hidden_size(model), overrides


LOCAL_LOADERS = {
    'ernie_rna_giga_irf': _load_ernie, 'ernie_rna_giga': _load_ernie,
    'ernie_rna': _load_ernie, 'aidorna': _load_aidorna,
    'rinalmo_giga': _load_rinalmo,
}


def available_model_keys():
    return sorted(LOCAL_LOADERS)


def load_local_collective_model(model_key, checkpoint=None, device='cpu', **kwargs):
    if model_key not in LOCAL_LOADERS:
        raise ValueError(f'Unknown model key {model_key!r}; choose {available_model_keys()}')
    if not checkpoint or not Path(checkpoint).exists():
        raise FileNotFoundError(f'Local base checkpoint not found: {checkpoint}')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    model, tokenizer, hidden_size, overrides = LOCAL_LOADERS[model_key](device, checkpoint)
    overrides['validation'] = validate_tokenizer_alignment(model_key, model, tokenizer, overrides)
    _attach_overrides(model, overrides)
    return model, tokenizer, hidden_size
