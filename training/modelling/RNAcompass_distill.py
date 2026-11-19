import logging
import math
import os
import time
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

from utils.helpers import AverageMeterList, dict2json


RNA_BASES = ("A", "C", "G", "U")


def split_decay(named_params):
    decay, no_decay = [], []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if name.endswith("bias") or "layernorm.weight" in lowered or "layer_norm.weight" in lowered:
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def masked_ci_loss(pred, target, mask, mu_base=0.0, mu_target=0.0, sig=1.0):
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    sig = float(sig)
    if sig == 0.0:
        raise ValueError("Score-shifting sig must be non-zero.")
    shifted_pred = (pred - float(mu_base)) / sig
    shifted_target = (target.float() - float(mu_target)) / sig
    loss = F.mse_loss(shifted_pred, shifted_target, reduction="none")
    return (loss * mask).sum() / denom


def _normalize_base_probs(probs, epsilon):
    probs = probs.float().clamp_min(epsilon)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    return probs.clamp(max=1.0 - epsilon)


def _diag_keep_mask(size, mask_width, device, dtype):
    keep = torch.ones((size, size), device=device, dtype=dtype)
    if mask_width <= 0:
        return keep
    for k in range(-mask_width // 2 + 1, mask_width // 2 + 1):
        diag_len = size - abs(k)
        if diag_len <= 0:
            continue
        idx = torch.arange(diag_len, device=device)
        if k >= 0:
            keep[idx, idx + k] = 0.0
        else:
            keep[idx - k, idx] = 0.0
    return keep


def log_odds_ci_from_language_probs(
    ref_probs,
    mut_probs,
    mutation_positions,
    seq_len=None,
    mask_diag=1,
    symmetrize=False,
    epsilon=1e-10,
):
    """Match the ablation pipeline CI formula using A/C/G/U language-head probabilities."""
    ref_probs = _normalize_base_probs(ref_probs, epsilon)
    mut_probs = _normalize_base_probs(mut_probs, epsilon)
    if seq_len is None:
        seq_len = int(ref_probs.shape[0])
    seq_len = int(seq_len)
    mutation_positions = mutation_positions.to(ref_probs.device).long()

    if mut_probs.numel() == 0:
        ci = ref_probs.new_zeros((seq_len, seq_len))
    else:
        log_odds_ref = torch.log2(ref_probs + epsilon) - torch.log2(1.0 - ref_probs + epsilon)
        log_odds_mut = torch.log2(mut_probs + epsilon) - torch.log2(1.0 - mut_probs + epsilon)
        effects = log_odds_mut - log_odds_ref.unsqueeze(0)
        rows = []
        for pos in range(seq_len):
            idxs = torch.nonzero(mutation_positions == pos, as_tuple=False).flatten()
            if idxs.numel() == 0:
                rows.append(ref_probs.new_zeros(seq_len))
            else:
                pos_effects = effects.index_select(0, idxs)
                rows.append(pos_effects.abs().amax(dim=(0, 2))[:seq_len])
        ci = torch.stack(rows, dim=0)

    if seq_len > 0:
        ci = ci.clone()
        ci.fill_diagonal_(0.0)
    ci = ci * _diag_keep_mask(seq_len, int(mask_diag), ci.device, ci.dtype)
    if symmetrize:
        ci = torch.maximum(ci, ci.transpose(-1, -2))
    return ci


def _resolve_attr(obj, dotted_path):
    cur = obj
    for part in dotted_path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _model_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _as_layer_list(module):
    if isinstance(module, (nn.ModuleList, list, tuple)):
        return module
    return None


def find_transformer_layers(model):
    candidates = (
        "bert.encoder.layer",
        "roberta.encoder.layer",
        "encoder.layer",
        "model.encoder.layer",
        "transformer.h",
        "transformer.blocks",
        "model.transformer.blocks",
        "model.layers",
        "model.model.layers",
        "encoder.sentence_encoder.layers",
        "model.encoder.sentence_encoder.layers",
        "encoder.backbone.layers",
        "model.encoder.backbone.layers",
        "layers",
        "evmodel.evos",
        "model.evmodel.evos",
        "backbone.encoder.layer",
    )
    for path in candidates:
        layers = _as_layer_list(_resolve_attr(model, path))
        if layers is not None and len(layers) > 0:
            return layers
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            if all(isinstance(layer, nn.Module) for layer in module):
                return module
    return None


def find_embeddings(model):
    candidates = (
        "bert.embeddings",
        "roberta.embeddings",
        "embeddings",
        "model.embeddings",
        "transformer.wte",
        "embedding",
        "model.embedding",
        "embed_tokens",
        "model.embed_tokens",
        "encoder.embed_tokens",
        "model.encoder.embed_tokens",
        "emb",
        "model.emb",
    )
    for path in candidates:
        module = _resolve_attr(model, path)
        if isinstance(module, nn.Module):
            return module
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    return None


def find_input_embeddings(model):
    if hasattr(model, "get_input_embeddings"):
        module = model.get_input_embeddings()
        if isinstance(module, nn.Module) and hasattr(module, "weight"):
            return module
    candidates = (
        "bert.embeddings.word_embeddings",
        "roberta.embeddings.word_embeddings",
        "embeddings.word_embeddings",
        "model.embeddings.word_embeddings",
        "model.bert.embeddings.word_embeddings",
        "model.roberta.embeddings.word_embeddings",
        "embedding",
        "model.embedding",
        "embed_tokens",
        "model.embed_tokens",
        "encoder.embed_tokens",
        "model.encoder.embed_tokens",
        "decoder.embed_tokens",
        "model.decoder.embed_tokens",
        "sentence_encoder.embed_tokens",
        "model.sentence_encoder.embed_tokens",
        "encoder.sentence_encoder.embed_tokens",
        "model.encoder.sentence_encoder.embed_tokens",
        "decoder.sentence_encoder.embed_tokens",
        "model.decoder.sentence_encoder.embed_tokens",
        "encoder.backbone.embed_tokens",
        "model.encoder.backbone.embed_tokens",
    )
    for path in candidates:
        module = _resolve_attr(model, path)
        if isinstance(module, nn.Module) and hasattr(module, "weight"):
            return module
    return None


def find_norm(model):
    candidates = (
        "bert.encoder.layer_norm",
        "roberta.encoder.layer_norm",
        "encoder.layer_norm",
        "transformer.ln_f",
        "transformer.final_layer_norm",
        "model.transformer.final_layer_norm",
        "model.norm",
        "encoder.backbone.norm_f",
        "model.encoder.backbone.norm_f",
        "norm",
    )
    for path in candidates:
        module = _resolve_attr(model, path)
        if isinstance(module, nn.Module):
            return module
    return None


def _resolve_parent_attr(obj, dotted_path):
    parts = dotted_path.split(".")
    if not parts:
        return None, None
    parent = obj
    for part in parts[:-1]:
        if not hasattr(parent, part):
            return None, None
        parent = getattr(parent, part)
    if not hasattr(parent, parts[-1]):
        return None, None
    return parent, parts[-1]


def find_language_head_output_target(model):
    candidates = (
        "lm_head.decoder",
        "model.lm_head.decoder",
        "cls.predictions.decoder",
        "model.cls.predictions.decoder",
        "lm_mask_head.linear2",
        "model.lm_mask_head.linear2",
        "embed_out",
        "model.embed_out",
        "encoder.embed_out",
        "model.encoder.embed_out",
        "decoder.output_projection",
        "model.decoder.output_projection",
        "encoder.output_projection",
        "model.encoder.output_projection",
        "decoder.lm_head",
        "model.decoder.lm_head",
        "encoder.lm_head",
        "model.encoder.lm_head",
        "lm_head",
        "model.lm_head",
        "seq_head",
        "model.seq_head",
    )
    for path in candidates:
        target = _resolve_attr(model, path)
        if isinstance(target, nn.Module) and hasattr(target, "weight"):
            return path, target, "weight"
        if isinstance(target, nn.Parameter):
            parent, attr = _resolve_parent_attr(model, path)
            if parent is not None:
                return path, parent, attr
    return None, None, None


def _target_weight(target, attr):
    if target is None:
        return None
    return getattr(target, attr, None)


def _functionally_tied_to_input_embeddings(model):
    for path in (
        "share_input_output_embed",
        "model.share_input_output_embed",
        "encoder.share_input_output_embed",
        "model.encoder.share_input_output_embed",
    ):
        if bool(_resolve_attr(model, path)):
            return path
    return None


def tie_language_head_to_input_embeddings(model, force=False):
    input_embeddings = find_input_embeddings(model)
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        return {"status": "missing_input_embedding", "train_tied_embeddings": False}

    input_weight = input_embeddings.weight
    target_path, target, attr = find_language_head_output_target(model)
    if target is None:
        functional_path = _functionally_tied_to_input_embeddings(model)
        if functional_path is not None:
            return {
                "status": "already_functionally_tied",
                "output_path": functional_path,
                "train_tied_embeddings": True,
            }
        return {"status": "missing_output_projection", "train_tied_embeddings": False}

    output_weight = _target_weight(target, attr)
    if output_weight is None:
        return {"status": "missing_output_weight", "output_path": target_path, "train_tied_embeddings": False}
    if output_weight.data_ptr() == input_weight.data_ptr():
        return {"status": "already_tied", "output_path": target_path, "train_tied_embeddings": True}
    if not force:
        return {"status": "independent_head", "output_path": target_path, "train_tied_embeddings": False}
    if tuple(output_weight.shape) != tuple(input_weight.shape):
        raise ValueError(
            "Cannot tie language head to input embeddings: "
            f"{target_path}.weight shape={tuple(output_weight.shape)} "
            f"does not match input embedding shape={tuple(input_weight.shape)}."
        )

    setattr(target, attr, input_weight)
    return {"status": "force_tied", "output_path": target_path, "train_tied_embeddings": True}


def find_language_head(model):
    candidates = (
        "lm_head",
        "model.lm_head",
        "lm_mask_head",
        "model.lm_mask_head",
        "seq_head",
        "model.seq_head",
        "encoder.lm_head",
        "model.encoder.lm_head",
        "cls.predictions",
        "predictions",
        "output_heads.sequence_head",
        "sequence_head",
        "heads.lm_head",
    )
    for path in candidates:
        module = _resolve_attr(model, path)
        if isinstance(module, nn.Module):
            return module
    return None


def infer_base_token_ids(tokenizer, configured_ids=None):
    if configured_ids is not None:
        if isinstance(configured_ids, dict):
            return [int(configured_ids[base]) for base in RNA_BASES]
        return [int(token_id) for token_id in configured_ids]

    token_ids = []
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for base in RNA_BASES:
        found = None
        candidates = (base, base.lower())
        if base == "U":
            candidates = ("U", "u", "T", "t")
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            for candidate in candidates:
                token_id = tokenizer.convert_tokens_to_ids(candidate)
                if token_id is not None and token_id != unk_id and int(token_id) >= 0:
                    found = int(token_id)
                    break
        if found is None:
            encoded = tokenizer(base, add_special_tokens=False).get("input_ids", [])
            if isinstance(encoded, torch.Tensor):
                encoded = encoded.tolist()
            if len(encoded) == 1:
                found = int(encoded[0])
        if found is None:
            raise ValueError(
                f"Could not infer tokenizer id for RNA base {base}; "
                "set model.ci_params.base_token_ids as [A,C,G,U] ids in config."
            )
        token_ids.append(found)
    return token_ids


class RNAcompass_distill:
    def __init__(self, model, tokenizer, configs, hidden_size=None):
        self.model = model
        self.raw_model = _unwrap_model(model)
        self.tokenizer = tokenizer
        self.device = _model_device(self.model)
        self.configs = configs
        self.distributed_cfgs = configs.get("distributed", {})
        self.rank = int(self.distributed_cfgs.get("rank", 0))
        self.world_size = int(self.distributed_cfgs.get("world_size", 1))
        self.is_main_process = self.rank == 0
        self.model_cfgs = configs["model"]
        self.train_cfgs = configs["training"]
        self.model_name = configs["model_name"]
        self.early_stops = self.train_cfgs["early_stopping"]
        self.measures = [m for m, t in self.model_cfgs["measure_dct"].items() if t == "avg"]
        ci_cfg = self.model_cfgs.get("ci_params", {})
        self.base_token_ids = infer_base_token_ids(self.tokenizer, ci_cfg.get("base_token_ids"))
        self.model_path = f'{configs["model_dir"]}/v{self.model_name}.pth'
        log_fpath = f'{configs["model_dir"]}/log_{self.model_name}.out'
        score_fpath = f'{configs["model_dir"]}/score_{self.model_name}.json'

        if self.is_main_process:
            logging.basicConfig(
                filename=log_fpath,
                level=logging.INFO,
                format="%(asctime)s - %(message)s",
                force=True,
            )
            logging.getLogger().addHandler(logging.StreamHandler())
            self.log_fn = lambda message: logging.info(message)
            self.log_score_fn = lambda d: dict2json(score_fpath, d)
            self.save_score_fn = lambda d, suffix: dict2json(
                f'{configs["model_dir"]}/score_{self.model_name}_{suffix}.json', d
            )
        else:
            self.log_fn = lambda message: None
            self.log_score_fn = lambda d: None
            self.save_score_fn = lambda d, suffix: None

        self.train_tied_embeddings = False
        params_cfg = self._lm_params()
        if params_cfg.get("tie_head_to_embeddings", 0) or params_cfg.get("force_tie_head_to_embeddings", 0):
            force_tie = bool(params_cfg.get("force_tie_head_to_embeddings", 0))
            tie_status = tie_language_head_to_input_embeddings(self.raw_model, force=force_tie)
            self.train_tied_embeddings = bool(tie_status.get("train_tied_embeddings", False))
            output_path = tie_status.get("output_path")
            if output_path:
                self.log_fn(f"Input embedding/head tie status: {tie_status['status']} ({output_path})")
            else:
                self.log_fn(f"Input embedding/head tie status: {tie_status['status']}")
            if force_tie and tie_status["status"].startswith("missing_"):
                raise ValueError(
                    "model.lm_params.force_tie_head_to_embeddings=1, but Rcompass could not tie "
                    f"the language head to the input embedding: {tie_status['status']}."
                )
        self.language_head = find_language_head(self.raw_model)
        self.input_embedding_weight_id = None
        self.input_embedding_weight_ptr = None
        input_embeddings = find_input_embeddings(self.raw_model)
        if input_embeddings is not None and hasattr(input_embeddings, "weight"):
            weight = input_embeddings.weight
            self.input_embedding_weight_id = id(weight)
            self.input_embedding_weight_ptr = weight.data_ptr()

    def set_loaders(self, loaders):
        self.loaders = loaders

    def _progress(self, iterable, **kwargs):
        kwargs.setdefault("disable", not self.is_main_process)
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("mininterval", 2.0)
        return tqdm(iterable, **kwargs)

    @staticmethod
    def _metric_value(value):
        if torch.is_tensor(value):
            return float(value.detach().float().cpu())
        return float(value)

    def _set_progress_postfix(self, bar, metrics):
        if not self.is_main_process or not hasattr(bar, "set_postfix"):
            return
        postfix = {}
        for key, value in metrics.items():
            if value is None:
                continue
            postfix[key] = f"{value:.4g}" if isinstance(value, float) else value
        bar.set_postfix(postfix, refresh=False)

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
        return f"{minutes:d}m{seconds:02d}s"

    def _overall_progress(self, epoch, completed_batches, num_train_batches):
        total_batches = int(getattr(self, "train_total_batches", 0) or 0)
        if total_batches <= 0:
            total_batches = int(num_train_batches * self.train_cfgs["nepochs"])
        completed_total = int(epoch * num_train_batches + completed_batches)
        started_at = float(getattr(self, "train_started_at", time.time()))
        elapsed = max(time.time() - started_at, 1e-6)
        rate = completed_total / elapsed if completed_total > 0 else 0.0
        remaining = (total_batches - completed_total) / rate if rate > 0 else 0.0
        pct = 100.0 * completed_total / max(1, total_batches)
        return {
            "completed_total": completed_total,
            "total_batches": total_batches,
            "pct": pct,
            "elapsed": elapsed,
            "eta": remaining,
        }

    def _current_lr(self):
        if not hasattr(self, "optimizer") or not self.optimizer.param_groups:
            return None
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def _lm_params(self):
        return self.model_cfgs.get("lm_params", {})

    def _param_enabled(self, param_name, params_cfg):
        if param_name == "tied_embeddings":
            requested = params_cfg.get("tied_embeddings", params_cfg.get("tie_head_to_embeddings", 0))
            return requested and bool(getattr(self, "train_tied_embeddings", False))
        return params_cfg.get(param_name, 0)

    def _is_input_embedding_weight(self, param):
        if param is None:
            return False
        if self.input_embedding_weight_id is not None and id(param) == self.input_embedding_weight_id:
            return True
        if self.input_embedding_weight_ptr is not None and param.data_ptr() == self.input_embedding_weight_ptr:
            return True
        return False

    def _head_param_allowed(self, param, param_name, params_cfg):
        if param_name != "head":
            return True
        if self._param_enabled("tied_embeddings", params_cfg):
            return True
        return not self._is_input_embedding_weight(param)

    @contextmanager
    def _ci_forward_context(self):
        ci_cfg = self.model_cfgs.get("ci_params", {})
        disable_dropout = bool(ci_cfg.get("disable_dropout_for_ci", True))
        if not disable_dropout:
            yield
            return

        was_training = self.model.training
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(was_training)

    def get_param_dct(self):
        for param in self.raw_model.parameters():
            param.requires_grad = False
        return {
            "head": self.language_head,
            "nlayers": find_transformer_layers(self.raw_model),
            "tied_embeddings": find_input_embeddings(self.raw_model),
            "embeddings": find_embeddings(self.raw_model),
            "norm": find_norm(self.raw_model),
        }

    def get_trainable_parameters(self, param_dct, base_lr=2e-5, decay_factor=0.8, weight_decay=1e-2):
        params_cfg = self._lm_params()
        groups = []
        seen = set()

        def add_group(params, lr, param_name):
            unique = []
            for param in params:
                if not self._head_param_allowed(param, param_name, params_cfg):
                    continue
                param.requires_grad = True
                if id(param) in seen:
                    continue
                seen.add(id(param))
                unique.append(param)
            if unique:
                groups.append({"params": unique, "lr": lr, "weight_decay": weight_decay})

        for param_name, params in param_dct.items():
            if params is None or not self._param_enabled(param_name, params_cfg):
                continue
            if param_name == "nlayers":
                layers = params
                total_layers = len(layers)
                k = min(int(params_cfg.get("nlayers", 0)), total_layers)
                self.log_fn(f"Number of RNA LM layers: {total_layers}; train last {k}")
                for i in range(total_layers - k, total_layers):
                    lr_i = base_lr * (decay_factor ** (total_layers - 1 - i))
                    add_group(layers[i].parameters(), lr_i, param_name)
            else:
                add_group(params.parameters(), base_lr, param_name)
        return groups

    def get_trainable_parameters_with_decay(
        self, param_dct, base_lr=2e-5, head_lr=5e-5, decay_factor=0.8, weight_decay=1e-2
    ):
        params_cfg = self._lm_params()
        groups = []
        seen = set()

        def add_split_group(named_params, lr, param_name):
            materialized = []
            for name, param in named_params:
                if not self._head_param_allowed(param, param_name, params_cfg):
                    continue
                param.requires_grad = True
                if id(param) in seen:
                    continue
                seen.add(id(param))
                materialized.append((name, param))
            dec, nd = split_decay(materialized)
            if dec:
                groups.append({"params": dec, "lr": lr, "weight_decay": weight_decay})
            if nd:
                groups.append({"params": nd, "lr": lr, "weight_decay": 0.0})

        for param_name, params in param_dct.items():
            if params is None or not self._param_enabled(param_name, params_cfg):
                continue
            if param_name == "nlayers":
                layers = params
                total_layers = len(layers)
                k = min(int(params_cfg.get("nlayers", 0)), total_layers)
                self.log_fn(f"Number of RNA LM layers: {total_layers}; train last {k}")
                for i in range(total_layers - k, total_layers):
                    lr_i = base_lr * (decay_factor ** (total_layers - 1 - i))
                    add_split_group(layers[i].named_parameters(), lr_i, param_name)
            else:
                lr = head_lr if param_name in {"head", "tied_embeddings"} else base_lr
                add_split_group(params.named_parameters(), lr, param_name)
        return groups

    def get_llrd_groups(self, base_lr=2e-5, decay_rate=0.9, head_lr_mult=2.0, weight_decay=0.01):
        groups = []
        seen = set()

        def add_group(named_params, lr, wd):
            dec, nd = split_decay(named_params)
            dec = [p for p in dec if id(p) not in seen]
            nd = [p for p in nd if id(p) not in seen]
            seen.update(id(p) for p in dec)
            seen.update(id(p) for p in nd)
            if dec:
                groups.append({"params": dec, "lr": lr, "weight_decay": wd})
            if nd:
                groups.append({"params": nd, "lr": lr, "weight_decay": 0.0})

        for param in self.raw_model.parameters():
            param.requires_grad = True

        layers = find_transformer_layers(self.raw_model)
        embeddings = find_embeddings(self.raw_model)
        language_head = find_language_head(self.raw_model)
        if layers is None:
            add_group(self.raw_model.named_parameters(), base_lr, weight_decay)
        else:
            total_layers = len(layers)
            if embeddings is not None:
                add_group(embeddings.named_parameters(), base_lr * (decay_rate ** total_layers), weight_decay)
            for i in range(total_layers):
                lr_i = base_lr * (decay_rate ** (total_layers - 1 - i))
                add_group(layers[i].named_parameters(), lr_i, weight_decay)
            if language_head is not None:
                add_group(language_head.named_parameters(), base_lr * head_lr_mult, weight_decay)
        return groups

    def init_training(self, niters, decay_rate=0.9):
        base_lr = self.train_cfgs["optim"]["lr"]
        weight_decay = self.train_cfgs["optim"]["weight_decay"]
        head_lr_mult = self.train_cfgs["optim"]["head_lr_mult"]
        params_cfg = self._lm_params()
        if params_cfg.get("full", 0):
            groups = self.get_llrd_groups(base_lr, decay_rate, head_lr_mult, weight_decay)
        else:
            param_dct = self.get_param_dct()
            if self.train_cfgs["optim"].get("split_decay", 0):
                groups = self.get_trainable_parameters_with_decay(
                    param_dct, base_lr, base_lr * head_lr_mult, decay_rate, weight_decay
                )
            else:
                groups = self.get_trainable_parameters(param_dct, base_lr, decay_rate, weight_decay)
        if not groups:
            raise ValueError("No trainable parameter groups were created. Check model.lm_params in config.")
        self.optimizer = AdamW(groups, betas=(0.9, 0.999), eps=1e-8)

        total_steps = int(niters * self.train_cfgs["nepochs"] * self.train_cfgs["optim"]["scheduler_factor"])
        warmup_steps = int(self.train_cfgs["optim"]["warmup_rate"] * total_steps)
        self.train_started_at = time.time()
        self.train_total_batches = int(niters * self.train_cfgs["nepochs"])

        def cosine(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=cosine)

    def _normalize_sequence(self, sequence):
        tok_cfg = self.model_cfgs.get("tokenizer", {})
        seq = str(sequence).upper()
        if tok_cfg.get("replace_t_with_u", True):
            seq = seq.replace("T", "U")
        return seq

    def _prepare_sequences(self, sequences):
        tok_cfg = self.model_cfgs.get("tokenizer", {})
        prepared = []
        for sequence in sequences:
            seq = self._normalize_sequence(sequence)
            if tok_cfg.get("add_spaces", False):
                seq = " ".join(seq)
            prepared.append(seq)
        return prepared

    def _max_residue_length(self):
        max_length = int(self.model_cfgs.get("max_length", 1024))
        max_special_tokens = int(self.model_cfgs.get("max_special_tokens", 2))
        return max(1, max_length - max_special_tokens)

    def _mutable_query_rows(self, sequences, matrix_length):
        """Use only A/C/G/U mutation rows in both loss computation paths.

        N remains part of the input context and may be a target column, but
        cannot supply a reference base for the three-alternative mutation rule.
        """
        matrix_offset = int(self.model_cfgs.get("matrix_offset", 1))
        max_residues = self._max_residue_length()
        rows = torch.zeros(
            (len(sequences), matrix_length), device=self.device, dtype=torch.float32
        )
        for batch_idx, sequence in enumerate(sequences):
            sequence = self._normalize_sequence(sequence)[:max_residues]
            residue_len = min(len(sequence), max(0, matrix_length - matrix_offset))
            if residue_len:
                rows[batch_idx, matrix_offset : matrix_offset + residue_len] = rows.new_tensor(
                    [base in RNA_BASES for base in sequence[:residue_len]]
                )
        return rows

    def _generate_single_mutants(self, sequence, mutation_positions=None):
        mutants = [sequence]
        positions = []
        if mutation_positions is None:
            mutation_positions = range(len(sequence))
        for pos in mutation_positions:
            pos = int(pos)
            if pos < 0 or pos >= len(sequence):
                continue
            ref_base = sequence[pos]
            if ref_base not in RNA_BASES:
                continue
            for alt_base in RNA_BASES:
                if alt_base == ref_base:
                    continue
                mutants.append(sequence[:pos] + alt_base + sequence[pos + 1 :])
                positions.append(pos)
        return mutants, positions

    def _language_base_probs(self, sequences, seq_len):
        ci_cfg = self.model_cfgs.get("ci_params", {})
        batch_size = int(ci_cfg.get("mutation_batch_size", 16))
        token_offset = int(ci_cfg.get("token_offset", self.model_cfgs.get("matrix_offset", 1)))
        max_length = self.model_cfgs.get("max_length", 1024)
        base_ids = torch.tensor(self.base_token_ids, device=self.device, dtype=torch.long)
        probs = []
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            tokens = self.tokenizer(
                self._prepare_sequences(chunk),
                truncation=True,
                padding="longest",
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**tokens, output_hidden_states=False, return_dict=True)
            if not hasattr(outputs, "logits") or outputs.logits is None:
                raise ValueError("RNA LM did not return language-head logits; cannot build CI predictions.")
            logits = outputs.logits.float()
            if token_offset + seq_len > logits.shape[1]:
                raise ValueError(
                    f"Tokenized sequence length is too short for CI extraction: "
                    f"need offset {token_offset}+{seq_len}, got {logits.shape[1]}."
                )
            acgu_logits = logits[:, token_offset : token_offset + seq_len, :].index_select(-1, base_ids)
            chunk_probs = F.softmax(acgu_logits, dim=-1)
            probs.append(_normalize_base_probs(chunk_probs, float(ci_cfg.get("epsilon", 1e-10))))
        return torch.cat(probs, dim=0)

    def _log_odds_ci_for_sequence(self, sequence, mutation_positions=None):
        ci_cfg = self.model_cfgs.get("ci_params", {})
        seq_len = len(sequence)
        mutants, positions = self._generate_single_mutants(sequence, mutation_positions=mutation_positions)
        if len(positions) == 0:
            return torch.zeros((seq_len, seq_len), device=self.device, dtype=torch.float32)
        all_probs = self._language_base_probs(mutants, seq_len)
        mutation_positions = torch.tensor(positions, device=self.device, dtype=torch.long)
        return log_odds_ci_from_language_probs(
            all_probs[0],
            all_probs[1:],
            mutation_positions,
            seq_len=seq_len,
            mask_diag=int(ci_cfg.get("mask_diag", 1)),
            symmetrize=bool(ci_cfg.get("symmetrize", False)),
            epsilon=float(ci_cfg.get("epsilon", 1e-10)),
        )

    def _log_odds_logits(self, batch):
        max_residues = self._max_residue_length()
        sequences = [self._normalize_sequence(seq)[:max_residues] for seq in batch["sequence"]]
        max_length = self.model_cfgs.get("max_length", 1024)
        matrix_offset = int(self.model_cfgs.get("matrix_offset", 1))
        tokens = self.tokenizer(
            self._prepare_sequences(sequences),
            truncation=True,
            padding="longest",
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        token_len = int(tokens["attention_mask"].shape[1])
        ci_matrices = []
        for sequence in sequences:
            if matrix_offset + len(sequence) > token_len:
                raise ValueError(
                    f"Tokenized sequence length is too short for CI extraction: "
                    f"need offset {matrix_offset}+{len(sequence)}, got {token_len}."
                )
            ci_residue = self._log_odds_ci_for_sequence(sequence)
            padded = ci_residue.new_zeros((token_len, token_len))
            length = ci_residue.shape[0]
            stop = matrix_offset + length
            padded[matrix_offset:stop, matrix_offset:stop] = ci_residue[:length, :length]
            ci_matrices.append(padded)
        return {
            "attention_masks": tokens["attention_mask"],
            "input_ids": tokens["input_ids"],
            "sequence_logits": None,
            "ci_logits": torch.stack(ci_matrices, dim=0),
        }

    def logits(self, batch):
        return self._log_odds_logits(batch)

    def loss_fn(self, batch, return_stem_metrics=False):
        with self._ci_forward_context():
            return self._loss_fn(batch, return_stem_metrics=return_stem_metrics)

    def _loss_fn(self, batch, return_stem_metrics=False):
        outputs = self.logits(batch)
        pred = outputs["ci_logits"]
        target = batch["ci_matrix"].to(self.device).float()
        ci_mask = batch["ci_mask"].to(self.device).float()

        length = min(pred.shape[-1], target.shape[-1])
        pred = pred[:, :length, :length]
        target = target[:, :length, :length]
        ci_mask = ci_mask[:, :length, :length]

        finite_mask = torch.isfinite(target).float()
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        attention = outputs["attention_masks"][:, :length].float()
        pair_attention = attention.unsqueeze(1) * attention.unsqueeze(2)
        mask = ci_mask * finite_mask * pair_attention
        if self.train_cfgs.get("ignore_diagonal", True):
            eye = torch.eye(length, device=self.device).unsqueeze(0)
            mask = mask * (1.0 - eye)
        mask = mask * self._mutable_query_rows(batch["sequence"], length).unsqueeze(2)

        loss_sum = self._masked_ci_loss_numerator(pred, target, mask)
        loss_weight = mask.sum()
        global_loss = loss_sum / loss_weight.clamp_min(1.0)
        stem_cfg = self.model_cfgs.get("stem_loss", {})
        lambda_global = float(stem_cfg.get("lambda_global", 1.0))
        lambda_pair = float(stem_cfg.get("lambda_pair", 0.0) or 0.0)
        lambda_stem_window = float(stem_cfg.get("lambda_stem_window", 0.0) or 0.0)
        use_stem_loss = bool(stem_cfg.get("enabled", False)) and (
            lambda_pair != 0.0 or lambda_stem_window != 0.0 or lambda_global != 1.0
        )
        pair_mask = self._filtered_pair_mask(batch, length, mask)
        if not use_stem_loss:
            result = {
                "loss": global_loss,
                "loss_sum": loss_sum.detach(),
                "loss_weight": loss_weight.detach(),
            }
            if return_stem_metrics:
                result.update(self._stem_recall_topl_metrics(pred.detach(), pair_mask, mask, batch))
            return result

        pair_loss_sum = self._masked_ci_loss_numerator(pred, target, pair_mask)
        pair_weight = pair_mask.sum()
        pair_loss = pair_loss_sum / pair_weight.clamp_min(1.0)
        loss = lambda_global * global_loss + lambda_pair * pair_loss
        result = {
            "loss": loss,
            "loss_global": global_loss.detach(),
            "loss_pair": pair_loss.detach(),
            "loss_sum": (loss.detach() * loss_weight.detach()),
            "loss_weight": loss_weight.detach(),
            "loss_global_weight": loss_weight.detach(),
            "loss_pair_weight": pair_weight.detach(),
            "filtered_pair_weight": pair_weight.detach(),
        }
        if lambda_stem_window != 0.0:
            stem_window_mask = self._sampled_stem_window_mask(pred.detach(), pair_mask, mask, stem_cfg)
            stem_window_loss_sum = self._stem_window_loss_numerator(pred, target, stem_window_mask, stem_cfg)
            stem_window_weight = stem_window_mask.sum()
            stem_window_loss = stem_window_loss_sum / stem_window_weight.clamp_min(1.0)
            loss = loss + lambda_stem_window * stem_window_loss
            result.update(
                {
                    "loss": loss,
                    "loss_stem_window": stem_window_loss.detach(),
                    "loss_stem_window_weight": stem_window_weight.detach(),
                    "stem_window_weight": stem_window_weight.detach(),
                }
            )
        if return_stem_metrics:
            result.update(self._stem_recall_topl_metrics(pred.detach(), pair_mask, mask, batch))
        return result

    def _masked_ci_loss_numerator(self, pred, target, mask):
        setting_cfg = self.configs.get("setting", {})
        sig = float(setting_cfg.get("sig", 1.0))
        if sig == 0.0:
            raise ValueError("Score-shifting sig must be non-zero.")
        shifted_pred = (pred - float(setting_cfg.get("mu_base", 0.0))) / sig
        shifted_target = (target.float() - float(setting_cfg.get("mu_target", 0.0))) / sig
        loss = F.mse_loss(shifted_pred, shifted_target, reduction="none")
        return (loss * mask.float()).sum()

    def _filtered_pair_mask(self, batch, length, base_mask):
        pair_mask = batch.get("filtered_pair_mask")
        if pair_mask is None:
            return base_mask.new_zeros(base_mask.shape)
        pair_mask = pair_mask.to(self.device).float()
        pair_mask = pair_mask[:, :length, :length]
        return pair_mask * base_mask

    def _use_static_stem_window(self, stem_cfg):
        mode = str(stem_cfg.get("stem_window_mode", "sampled")).strip().lower()
        return mode in {"local_static", "static_local"}

    def _use_teacher_stem_window_target(self, stem_cfg):
        target = str(stem_cfg.get("stem_window_target", "floor")).strip().lower()
        return target in {"teacher", "teacher_mse", "target", "target_mse"}

    def _include_pair_in_stem_window(self, stem_cfg):
        return bool(
            stem_cfg.get(
                "include_pair_in_stem_window",
                stem_cfg.get("local_window_include_pairs", False),
            )
        )

    def _stem_window_mask(self, pair_mask, base_mask, stem_cfg):
        radius = int(stem_cfg.get("stem_window_radius", 0) or 0)
        if radius <= 0 or pair_mask.sum().item() <= 0:
            return base_mask.new_zeros(base_mask.shape)
        kernel_size = 2 * radius + 1
        kernel = pair_mask.new_ones((1, 1, kernel_size, kernel_size))
        nearby = F.conv2d(pair_mask.float().unsqueeze(1), kernel, padding=radius).squeeze(1) > 0
        local_mask = nearby.float() * base_mask.float()
        if self._include_pair_in_stem_window(stem_cfg):
            return local_mask
        return local_mask * (1.0 - pair_mask.float().clamp(max=1.0))

    def _row_hard_stem_window_mask(self, pred, pair_mask, base_mask, stem_cfg):
        hard_per_positive = float(stem_cfg.get("stem_window_hard_per_positive", 0.0) or 0.0)
        if hard_per_positive <= 0.0 or pair_mask.sum().item() <= 0:
            return base_mask.new_zeros(base_mask.shape)
        candidates = base_mask.bool() & ~pair_mask.bool()
        hard_mask = torch.zeros_like(base_mask, dtype=torch.bool)
        batch_size, n_rows, _ = candidates.shape
        for batch_idx in range(batch_size):
            for row_idx in range(n_rows):
                row_candidates = candidates[batch_idx, row_idx]
                candidate_count = int(row_candidates.sum().item())
                row_positive_count = int(pair_mask[batch_idx, row_idx].bool().sum().item())
                if candidate_count <= 0 or row_positive_count <= 0:
                    continue
                k = min(candidate_count, max(1, int(math.ceil(row_positive_count * hard_per_positive))))
                row_scores = pred[batch_idx, row_idx].float().masked_fill(~row_candidates, -float("inf"))
                selected = torch.topk(row_scores, k=k).indices
                hard_mask[batch_idx, row_idx, selected] = True
        return hard_mask.float() * base_mask.float()

    def _cap_stem_window_mask_by_row(self, pred, stem_window_mask, pair_mask, stem_cfg):
        max_per_positive = float(stem_cfg.get("stem_window_max_per_positive", 0.0) or 0.0)
        if max_per_positive <= 0.0 or stem_window_mask.sum().item() <= 0:
            return stem_window_mask.float()
        capped = torch.zeros_like(stem_window_mask, dtype=torch.bool)
        candidates = stem_window_mask.bool()
        batch_size, n_rows, _ = candidates.shape
        for batch_idx in range(batch_size):
            for row_idx in range(n_rows):
                row_candidates = candidates[batch_idx, row_idx]
                candidate_count = int(row_candidates.sum().item())
                if candidate_count <= 0:
                    continue
                row_positive_count = int(pair_mask[batch_idx, row_idx].bool().sum().item())
                if row_positive_count <= 0:
                    continue
                k = min(candidate_count, max(1, int(math.ceil(row_positive_count * max_per_positive))))
                row_scores = pred[batch_idx, row_idx].float().masked_fill(~row_candidates, -float("inf"))
                selected = torch.topk(row_scores, k=k).indices
                capped[batch_idx, row_idx, selected] = True
        return capped.float() * stem_window_mask.float()

    def _sampled_stem_window_mask(self, pred, pair_mask, base_mask, stem_cfg):
        local_mask = self._stem_window_mask(pair_mask, base_mask, stem_cfg)
        if self._use_static_stem_window(stem_cfg):
            return local_mask
        hard_mask = self._row_hard_stem_window_mask(pred, pair_mask, base_mask, stem_cfg)
        stem_window_mask = ((local_mask + hard_mask) > 0).float() * base_mask.float()
        stem_window_mask = stem_window_mask * (1.0 - pair_mask.float().clamp(max=1.0))
        return self._cap_stem_window_mask_by_row(pred, stem_window_mask, pair_mask, stem_cfg)

    def _sampled_stem_window_rows(self, pred_rows, pair_mask_rows, base_mask_rows, local_mask_rows, stem_cfg):
        if self._use_static_stem_window(stem_cfg):
            return local_mask_rows.float()
        hard_rows = self._row_hard_stem_window_mask(
            pred_rows.detach().unsqueeze(0),
            pair_mask_rows.unsqueeze(0),
            base_mask_rows.unsqueeze(0),
            stem_cfg,
        )[0]
        stem_window_rows = ((local_mask_rows + hard_rows) > 0).float() * base_mask_rows.float()
        stem_window_rows = stem_window_rows * (1.0 - pair_mask_rows.float().clamp(max=1.0))
        return self._cap_stem_window_mask_by_row(
            pred_rows.detach().unsqueeze(0),
            stem_window_rows.unsqueeze(0),
            pair_mask_rows.unsqueeze(0),
            stem_cfg,
        )[0]

    def _stem_window_floor_value(self, stem_cfg):
        setting_cfg = self.configs.get("setting", {})
        floor = stem_cfg.get("stem_window_floor", "mu_target")
        if isinstance(floor, str):
            key = floor.strip()
            if key == "mu_target":
                return float(setting_cfg.get("mu_target", 0.0))
            if key == "mu_base":
                return float(setting_cfg.get("mu_base", 0.0))
            return float(key)
        return float(floor)

    def _stem_window_floor_loss_numerator(self, pred, mask, stem_cfg):
        setting_cfg = self.configs.get("setting", {})
        sig = float(setting_cfg.get("sig", 1.0))
        if sig == 0.0:
            raise ValueError("Score-shifting sig must be non-zero.")
        shifted_pred = (pred - float(setting_cfg.get("mu_base", 0.0))) / sig
        shifted_floor = (self._stem_window_floor_value(stem_cfg) - float(setting_cfg.get("mu_target", 0.0))) / sig
        loss = F.relu(shifted_pred - shifted_floor).pow(2)
        return (loss * mask.float()).sum()

    def _stem_window_loss_numerator(self, pred, target, mask, stem_cfg):
        if self._use_teacher_stem_window_target(stem_cfg):
            return self._masked_ci_loss_numerator(pred, target, mask)
        return self._stem_window_floor_loss_numerator(pred, mask, stem_cfg)

    def _stem_eval_top_k(self, residue_len, candidate_count):
        stem_eval_cfg = self.model_cfgs.get("stem_eval", {})
        top_k = stem_eval_cfg.get("top_k", "L")
        if isinstance(top_k, str):
            value = top_k.strip().upper()
            if value == "L":
                k = int(residue_len)
            elif value.endswith("L"):
                factor = float(value[:-1] or 1.0)
                k = int(round(factor * int(residue_len)))
            else:
                k = int(float(value))
        else:
            k = int(top_k)
        return max(1, min(int(candidate_count), k))

    def _cluster_label_pairs_to_stems(self, label_pairs):
        stems = []
        current = []
        for i, j in sorted(label_pairs, key=lambda pair: (pair[0], -pair[1])):
            if current:
                prev_i, prev_j = current[-1]
                if int(i) == int(prev_i) + 1 and int(j) == int(prev_j) - 1:
                    current.append((int(i), int(j)))
                    continue
                stems.append(current)
            current = [(int(i), int(j))]
        if current:
            stems.append(current)
        return stems

    def _stem_recall_topl_metrics(self, pred, pair_mask, base_mask, batch):
        matrix_offset = int(self.model_cfgs.get("matrix_offset", 1))
        stem_eval_cfg = self.model_cfgs.get("stem_eval", {})
        stem_hit_fraction = float(stem_eval_cfg.get("stem_hit_fraction", 0.5))
        seq_lengths = batch.get("seq_length")
        stem_hits = 0
        stem_total = 0
        pair_hits = 0
        pair_total = 0
        batch_size = int(pred.shape[0])
        for batch_idx in range(batch_size):
            if seq_lengths is None:
                residue_len = max(0, int(pred.shape[-1]) - matrix_offset)
            else:
                residue_len = int(seq_lengths[batch_idx].item() if torch.is_tensor(seq_lengths) else seq_lengths[batch_idx])
            start = matrix_offset
            stop = min(int(pred.shape[-1]), start + residue_len)
            residue_len = max(0, stop - start)
            if residue_len <= 1:
                continue

            scores = pred[batch_idx, start:stop, start:stop].float()
            candidates = base_mask[batch_idx, start:stop, start:stop].bool()
            upper = torch.triu(torch.ones((residue_len, residue_len), device=scores.device, dtype=torch.bool), diagonal=1)
            candidates = candidates & upper
            labels = pair_mask[batch_idx, start:stop, start:stop].bool() & upper
            label_pairs = torch.nonzero(labels, as_tuple=False).tolist()
            if not label_pairs:
                continue
            candidate_count = int(candidates.sum().item())
            if candidate_count <= 0:
                continue

            k = self._stem_eval_top_k(residue_len, candidate_count)
            masked_scores = scores.masked_fill(~candidates, -float("inf")).flatten()
            top_indices = torch.topk(masked_scores, k=k).indices
            hit_mask = torch.zeros_like(candidates)
            hit_mask.flatten()[top_indices] = True
            pair_hits += int((hit_mask & labels).sum().item())
            pair_total += int(labels.sum().item())

            stems = self._cluster_label_pairs_to_stems(label_pairs)
            stem_total += len(stems)
            for stem in stems:
                hits = sum(1 for i, j in stem if bool(hit_mask[int(i), int(j)].item()))
                if hits / max(1, len(stem)) >= stem_hit_fraction:
                    stem_hits += 1

        return {
            "stem_recall_topL": stem_hits / max(1, stem_total),
            "stem_recall_topL_hit": stem_hits,
            "stem_recall_topL_weight": stem_total,
            "filtered_pair_recall_topL": pair_hits / max(1, pair_total),
            "filtered_pair_recall_topL_hit": pair_hits,
            "filtered_pair_recall_topL_weight": pair_total,
        }

    def _sequence_loss_plan(self, sequence, target, ci_mask, filtered_pair_mask=None):
        sequence = self._normalize_sequence(sequence)[: self._max_residue_length()]
        max_length = self.model_cfgs.get("max_length", 1024)
        matrix_offset = int(self.model_cfgs.get("matrix_offset", 1))
        tokens = self.tokenizer(
            self._prepare_sequences([sequence]),
            truncation=True,
            padding="longest",
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        token_len = int(tokens["attention_mask"].shape[1])
        length = min(token_len, target.shape[-1])

        target = target.to(self.device).float()[:length, :length]
        ci_mask = ci_mask.to(self.device).float()[:length, :length]
        finite_mask = torch.isfinite(target).float()
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        attention = tokens["attention_mask"][0, :length].float()
        pair_attention = attention.unsqueeze(0) * attention.unsqueeze(1)
        mask = ci_mask * finite_mask * pair_attention
        if self.train_cfgs.get("ignore_diagonal", True):
            mask = mask * (1.0 - torch.eye(length, device=self.device))
        residue_len = min(len(sequence), max(0, length - matrix_offset))
        mutable_rows = self._mutable_query_rows([sequence], length)[0]
        mask = mask * mutable_rows.unsqueeze(1)
        if filtered_pair_mask is None:
            pair_mask = torch.zeros_like(mask)
        else:
            pair_mask = filtered_pair_mask.to(self.device).float()[:length, :length] * mask
        stem_cfg = self.model_cfgs.get("stem_loss", {})
        stem_window_mask = self._stem_window_mask(
            pair_mask.unsqueeze(0),
            mask.unsqueeze(0),
            stem_cfg,
        )[0]

        positions = []
        for pos in range(residue_len):
            row_idx = matrix_offset + pos
            if row_idx < length and mask[row_idx].sum().item() > 0:
                positions.append(pos)
        return {
            "sequence": sequence,
            "target": target,
            "mask": mask,
            "pair_mask": pair_mask,
            "stem_window_mask": stem_window_mask,
            "length": length,
            "residue_len": residue_len,
            "positions": positions,
        }

    def _padded_ci_rows(self, ci_residue, positions, length, residue_len):
        matrix_offset = int(self.model_cfgs.get("matrix_offset", 1))
        pos_tensor = torch.tensor(positions, device=self.device, dtype=torch.long)
        rows = ci_residue.index_select(0, pos_tensor)[:, :residue_len]
        pred_rows = rows.new_zeros((len(positions), length))
        if residue_len > 0:
            pred_rows[:, matrix_offset : matrix_offset + residue_len] = rows
        return pred_rows

    def backward_chunked_ci_loss(self, batch, row_chunk_size):
        ci_cfg = self.model_cfgs.get("ci_params", {})
        if bool(ci_cfg.get("symmetrize", False)):
            raise ValueError("Chunked CI backward does not support symmetrize=true.")
        row_chunk_size = max(1, int(row_chunk_size))
        with self._ci_forward_context():
            stem_cfg = self.model_cfgs.get("stem_loss", {})
            lambda_global = float(stem_cfg.get("lambda_global", 1.0))
            lambda_pair = float(stem_cfg.get("lambda_pair", 0.0) or 0.0)
            lambda_stem_window = float(stem_cfg.get("lambda_stem_window", 0.0) or 0.0)
            static_stem_window = self._use_static_stem_window(stem_cfg)
            use_stem_loss = bool(stem_cfg.get("enabled", False)) and (
                lambda_pair != 0.0 or lambda_stem_window != 0.0 or lambda_global != 1.0
            )
            if not use_stem_loss:
                lambda_global = 1.0
                lambda_pair = 0.0
                lambda_stem_window = 0.0
                static_stem_window = False
            plans = []
            loss_weight = torch.zeros((), device=self.device, dtype=torch.float32)
            pair_weight = torch.zeros((), device=self.device, dtype=torch.float32)
            stem_window_weight = torch.zeros((), device=self.device, dtype=torch.float32)
            targets = batch["ci_matrix"]
            masks = batch["ci_mask"]
            filtered_pair_masks = batch.get("filtered_pair_mask")
            for idx, sequence in enumerate(batch["sequence"]):
                filtered_pair_mask = filtered_pair_masks[idx] if filtered_pair_masks is not None else None
                plan = self._sequence_loss_plan(sequence, targets[idx], masks[idx], filtered_pair_mask)
                loss_weight = loss_weight + plan["mask"].sum()
                pair_weight = pair_weight + plan["pair_mask"].sum()
                if plan["positions"]:
                    plans.append(plan)

            denom = loss_weight.clamp_min(1.0)
            pair_denom = pair_weight.clamp_min(1.0)
            chunks = []
            for plan in plans:
                positions = plan["positions"]
                for start in range(0, len(positions), row_chunk_size):
                    chunks.append((plan, positions[start : start + row_chunk_size]))

            if lambda_stem_window != 0.0 and chunks:
                if static_stem_window:
                    for plan, positions in chunks:
                        row_indices = torch.tensor(
                            [int(self.model_cfgs.get("matrix_offset", 1)) + pos for pos in positions],
                            device=self.device,
                            dtype=torch.long,
                        )
                        stem_window_weight = stem_window_weight + plan["stem_window_mask"].index_select(0, row_indices).sum()
                else:
                    with torch.no_grad():
                        for plan, positions in chunks:
                            ci_residue = self._log_odds_ci_for_sequence(plan["sequence"], mutation_positions=positions)
                            pred_rows = self._padded_ci_rows(
                                ci_residue,
                                positions,
                                plan["length"],
                                plan["residue_len"],
                            )
                            row_indices = torch.tensor(
                                [int(self.model_cfgs.get("matrix_offset", 1)) + pos for pos in positions],
                                device=self.device,
                                dtype=torch.long,
                            )
                            mask_rows = plan["mask"].index_select(0, row_indices)
                            pair_mask_rows = plan["pair_mask"].index_select(0, row_indices)
                            stem_window_rows = plan["stem_window_mask"].index_select(0, row_indices)
                            stem_window_rows = self._sampled_stem_window_rows(
                                pred_rows,
                                pair_mask_rows,
                                mask_rows,
                                stem_window_rows,
                                stem_cfg,
                            )
                            stem_window_weight = stem_window_weight + stem_window_rows.sum()

            total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            total_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            total_pair_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            total_stem_window_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
            if not chunks:
                return {
                    "loss": total_loss,
                    "loss_sum": total_loss_sum,
                    "loss_weight": loss_weight.detach(),
                }

            chunk_iter = self._progress(
                list(enumerate(chunks)),
                desc="backward chunks",
                leave=False,
                total=len(chunks),
            )
            for chunk_idx, (plan, positions) in chunk_iter:
                sync_context = nullcontext()
                if hasattr(self.model, "no_sync") and chunk_idx < len(chunks) - 1:
                    sync_context = self.model.no_sync()
                with sync_context:
                    ci_residue = self._log_odds_ci_for_sequence(plan["sequence"], mutation_positions=positions)
                    pred_rows = self._padded_ci_rows(
                        ci_residue,
                        positions,
                        plan["length"],
                        plan["residue_len"],
                    )
                    row_indices = torch.tensor(
                        [int(self.model_cfgs.get("matrix_offset", 1)) + pos for pos in positions],
                        device=self.device,
                        dtype=torch.long,
                    )
                    target_rows = plan["target"].index_select(0, row_indices)
                    mask_rows = plan["mask"].index_select(0, row_indices)
                    pair_mask_rows = plan["pair_mask"].index_select(0, row_indices)
                    loss_sum = self._masked_ci_loss_numerator(pred_rows, target_rows, mask_rows)
                    pair_loss_sum = self._masked_ci_loss_numerator(pred_rows, target_rows, pair_mask_rows)
                    stem_window_loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
                    if lambda_stem_window != 0.0:
                        stem_window_rows = plan["stem_window_mask"].index_select(0, row_indices)
                        stem_window_rows = self._sampled_stem_window_rows(
                            pred_rows,
                            pair_mask_rows,
                            mask_rows,
                            stem_window_rows,
                            stem_cfg,
                        )
                        stem_window_loss_sum = self._stem_window_loss_numerator(
                            pred_rows,
                            target_rows,
                            stem_window_rows,
                            stem_cfg,
                        )
                    loss = lambda_global * loss_sum / denom
                    if use_stem_loss:
                        loss = loss + lambda_pair * pair_loss_sum / pair_denom
                        if lambda_stem_window != 0.0:
                            loss = loss + lambda_stem_window * stem_window_loss_sum / stem_window_weight.clamp_min(1.0)
                    loss.backward()
                    total_loss = total_loss + loss.detach()
                    total_loss_sum = total_loss_sum + loss_sum.detach()
                    total_pair_loss_sum = total_pair_loss_sum + pair_loss_sum.detach()
                    total_stem_window_loss_sum = total_stem_window_loss_sum + stem_window_loss_sum.detach()
                    self._set_progress_postfix(
                        chunk_iter,
                        {
                            "chunk_loss": self._metric_value(loss),
                            "rows": len(positions),
                        },
                    )
            result = {
                "loss": total_loss,
                "loss_sum": total_loss_sum,
                "loss_weight": loss_weight.detach(),
            }
            if use_stem_loss:
                result.update(
                    {
                        "loss_global": (total_loss_sum / denom).detach(),
                        "loss_pair": (total_pair_loss_sum / pair_denom).detach(),
                        "loss_global_weight": loss_weight.detach(),
                        "loss_pair_weight": pair_weight.detach(),
                        "filtered_pair_weight": pair_weight.detach(),
                    }
                )
                if lambda_stem_window != 0.0:
                    result.update(
                        {
                            "loss_stem_window": (total_stem_window_loss_sum / stem_window_weight.clamp_min(1.0)).detach(),
                            "loss_stem_window_weight": stem_window_weight.detach(),
                            "stem_window_weight": stem_window_weight.detach(),
                        }
                    )
            return result

    def optimize(self, batch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        row_chunk_size = int(self.model_cfgs.get("ci_params", {}).get("backward_row_chunk_size", 0) or 0)
        if row_chunk_size > 0:
            loss_dct = self.backward_chunked_ci_loss(batch, row_chunk_size)
        else:
            loss_dct = self.loss_fn(batch)
            loss_dct["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.train_cfgs.get("grad_clip", 1.0))
        self.optimizer.step()
        self.scheduler.step()
        return loss_dct

    def _distributed_avg(self, losses_dct):
        averages = losses_dct.get_avg()
        if self.world_size <= 1 or not dist.is_available() or not dist.is_initialized():
            return averages
        reduce_device = self.device if torch.device(self.device).type == "cuda" else torch.device("cpu")
        for metric, metric_type in self.model_cfgs["measure_dct"].items():
            if metric_type != "avg":
                continue
            meter = losses_dct.meter_lst[metric]
            stats = torch.tensor([meter.sum, meter.count], device=reduce_device, dtype=torch.float64)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            averages[metric] = (stats[0] / stats[1].clamp_min(1.0)).item()
        return averages

    def evaluate(self, val_loader):
        self.model.eval()
        losses_dct = AverageMeterList(self.model_cfgs["measure_dct"])
        stem_hits = 0.0
        stem_total = 0.0
        pair_hits = 0.0
        pair_total = 0.0
        iterator = self._progress(val_loader, desc="valid", leave=False, total=len(val_loader))
        with torch.no_grad():
            for batch in iterator:
                loss_dct = self.loss_fn(batch, return_stem_metrics=True)
                losses_dct.update(loss_dct, len(batch["row_id"]))
                stem_hits += float(loss_dct.get("stem_recall_topL_hit", 0.0))
                stem_total += float(loss_dct.get("stem_recall_topL_weight", 0.0))
                pair_hits += float(loss_dct.get("filtered_pair_recall_topL_hit", 0.0))
                pair_total += float(loss_dct.get("filtered_pair_recall_topL_weight", 0.0))
                averages = losses_dct.get_avg()
                self._set_progress_postfix(
                    iterator,
                    {metric: float(averages[metric]) for metric in self.measures if metric in averages},
                )
        averages = self._distributed_avg(losses_dct)
        stats = torch.tensor([stem_hits, stem_total, pair_hits, pair_total], device=self.device, dtype=torch.float64)
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        stem_hits, stem_total, pair_hits, pair_total = [float(value) for value in stats.tolist()]
        if stem_total > 0:
            averages["stem_recall_topL"] = stem_hits / stem_total
            averages["stem_recall_topL_hit"] = stem_hits
            averages["stem_recall_topL_weight"] = stem_total
        if pair_total > 0:
            averages["filtered_pair_recall_topL"] = pair_hits / pair_total
            averages["filtered_pair_recall_topL_hit"] = pair_hits
            averages["filtered_pair_recall_topL_weight"] = pair_total
        return averages

    def eval_epoch(self, loss_dct, return_score=False):
        score = -float(loss_dct["loss"])
        post_fix = {"loss": loss_dct["loss"], "score": score, "losses": loss_dct}
        self.log_fn(f"\t ci-val-score {score:.4f}")
        return post_fix

    def _maybe_save_best_stem_checkpoint(self, valid_metrics, best_stem_score, post_fix):
        if "stem_recall_topL" not in valid_metrics:
            return best_stem_score
        stem_score = float(valid_metrics["stem_recall_topL"])
        if stem_score > best_stem_score:
            best_stem_score = stem_score
            self.log_fn("== best stem recall")
            self.save_model("stem")
            self.save_score_fn(post_fix, "stem")
        return best_stem_score

    def _validation_interval(self, num_train_batches):
        validations_per_epoch = int(self.train_cfgs.get("validation_per_epoch", 1) or 1)
        validations_per_epoch = max(1, validations_per_epoch)
        return max(1, math.ceil(max(1, int(num_train_batches)) / validations_per_epoch))

    def _should_validate_batch(self, completed_batches, num_train_batches):
        completed_batches = int(completed_batches)
        num_train_batches = int(num_train_batches)
        if completed_batches >= num_train_batches:
            return True
        return completed_batches % self._validation_interval(num_train_batches) == 0

    def train_epoch(self, epoch, best_scores, log_freq):
        min_loss, best_score = best_scores[:2]
        best_stem_score = best_scores[2] if len(best_scores) >= 3 else -float("inf")
        loss_train_dct = AverageMeterList(self.model_cfgs["measure_dct"])
        high_loss_count = 0
        num_train_batches = len(self.loaders["train"])
        log_every = max(1, int(log_freq))
        train_iter = self._progress(
            self.loaders["train"],
            desc=f"epoch {epoch + 1}/{self.train_cfgs['nepochs']}",
            leave=True,
            total=num_train_batches,
        )
        for batch_idx, batch in enumerate(train_iter):
            loss_batch_dct = self.optimize(batch)
            loss_train_dct.update(loss_batch_dct, len(batch["row_id"]))
            train_avg = loss_train_dct.get_avg()

            completed_batches = batch_idx + 1
            overall = self._overall_progress(epoch, completed_batches, num_train_batches)
            self._set_progress_postfix(
                train_iter,
                {
                    "loss": self._metric_value(loss_batch_dct.get("loss", 0.0)),
                    "avg_loss": float(train_avg.get("loss", 0.0)),
                    "lr": self._current_lr(),
                    "total": f"{overall['pct']:.1f}%",
                    "elapsed": self._format_duration(overall["elapsed"]),
                    "eta": self._format_duration(overall["eta"]),
                },
            )
            is_epoch_end = completed_batches == num_train_batches
            if completed_batches % log_every == 0 or is_epoch_end:
                self.log_fn(
                    "[{:02}/{}] total {}/{} ({:.1f}%) elapsed {} eta {}".format(
                        epoch,
                        completed_batches,
                        overall["completed_total"],
                        overall["total_batches"],
                        overall["pct"],
                        self._format_duration(overall["elapsed"]),
                        self._format_duration(overall["eta"]),
                    )
                )

            if self._should_validate_batch(completed_batches, num_train_batches):
                self.model.eval()
                loss_dct = {
                    "train": self._distributed_avg(loss_train_dct),
                    "valid": self.evaluate(self.loaders["valid"]),
                }
                for split, dct in loss_dct.items():
                    metric_names = list(self.measures)
                    if split == "valid":
                        metric_names += [
                            "stem_recall_topL",
                            "filtered_pair_recall_topL",
                        ]
                    self.log_fn(
                        f"\t {split}: "
                        + " ".join(["{} {:.3f}".format(m, dct[m]) for m in metric_names if m in dct])
                    )

                val_loss = loss_dct["valid"]["loss"]
                post_fix = self.eval_epoch(loss_dct["valid"], return_score=False)
                score = post_fix["score"]
                post_fix.update(
                    {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "completed_batches": completed_batches,
                        "num_batches": num_train_batches,
                        "epoch_end": is_epoch_end,
                    }
                )

                if is_epoch_end:
                    epoch_suffix = f"epoch{epoch + 1:03d}"
                    self.log_fn(f"== save {epoch_suffix}")
                    self.save_model(epoch_suffix)
                    self.save_score_fn(post_fix, epoch_suffix)

                if val_loss < min_loss:
                    min_loss = val_loss
                    self.log_fn("== best validation loss")
                    self.save_model("val")
                    self.save_score_fn(post_fix, "val")
                    high_loss_count = 0
                elif val_loss > loss_dct["train"]["loss"]:
                    gap = (val_loss - loss_dct["train"]["loss"]) / max(loss_dct["train"]["loss"], 1e-8)
                    if gap >= 0.5:
                        self.log_fn("high val/train loss ratio")
                        high_loss_count += 1

                if score >= best_score + self.early_stops["max_delta"]:
                    best_score = score
                    self.log_fn("== best score")
                    self.save_model()
                    self.log_score_fn(post_fix)
                    self.early_stops["counter"] = 0
                elif score < best_score - self.early_stops["min_delta"]:
                    self.early_stops["counter"] += 1
                best_stem_score = self._maybe_save_best_stem_checkpoint(
                    loss_dct["valid"],
                    best_stem_score,
                    post_fix,
                )

            if self.early_stops["counter"] >= self.early_stops["patience"] and epoch >= self.train_cfgs["minimum_epochs"]:
                self.log_fn("Early stopping triggered.")
                break
            if high_loss_count > self.early_stops["patience"]:
                self.log_fn("Early stopping triggered with high validation loss.")
                break

        return min_loss, best_score, best_stem_score

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Training checkpoint not found: {model_path}")
        state = torch.load(model_path, map_location=self.device)
        if "model" in state:
            self.raw_model.load_state_dict(state["model"], strict=True)
        else:
            self.raw_model.load_state_dict(state, strict=True)
        self.log_fn("===> Load model")

    def save_model(self, suffix=""):
        if not self.is_main_process:
            return
        saved_dct = self.raw_model.state_dict()
        if len(suffix) > 0:
            model_path = self.model_path.replace(".pth", f"_{suffix}.pth")
        else:
            model_path = self.model_path
        torch.save(saved_dct, model_path)
