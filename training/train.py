import argparse
import logging
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data._utils.collate import default_collate

from modelling.RNAcompass_distill import RNAcompass_distill
from utils.helpers import load_rna_model, read_json, set_all_seeds, try_makedir
from utils.rna_ci_data_utils import get_ci_loaders, get_ci_loaders_from_splits, load_ci_data
from utils.rna_model_registry import merge_overrides
from utils.local_paths import DATA_ROOT, RESULTS_ROOT, resolve_config


def parse_args():
    parser = argparse.ArgumentParser(description="Rcompass RNA CI distillation")
    parser.add_argument("-g", "--gpu_id", default=0, type=int)
    parser.add_argument("-m", "--model_name", default="rna_lm", type=str)
    parser.add_argument("--model_checkpoint", default="", type=str)
    parser.add_argument("--model_key", default="", type=str)
    parser.add_argument("-c", "--config", default="r1/ernie_rna_giga", type=str)
    parser.add_argument("-b", "--batch_size", default=4, type=int)
    parser.add_argument("-v", "--val_batch_size_factor", default=2, type=int)
    parser.add_argument("--data_dir", default=str(DATA_ROOT / "training"), type=str)
    parser.add_argument("--train_file", default="", type=str)
    parser.add_argument("--valid_file", default="", type=str)
    parser.add_argument("--max_length", default=0, type=int)
    parser.add_argument("--initial_checkpoint", default="", help="Initialize from a distilled checkpoint.")
    parser.add_argument("--memory_preflight", action="store_true")
    parser.add_argument("--memory_preflight_only", action="store_true")
    parser.add_argument(
        "--seed_override",
        default=None,
        type=int,
        help="Override setting.seed without creating a duplicate config file.",
    )
    parser.add_argument(
        "--model_dir",
        default="",
        type=str,
        help="Override the checkpoint output directory.",
    )
    parser.add_argument(
        "--stop_after_epochs",
        default=0,
        type=int,
        help="Runtime-only cap on completed epochs; keeps config nepochs unchanged for scheduler setup.",
    )
    return parser.parse_args()


def setup_distributed(gpu_id):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(gpu_id)))
    distributed = world_size > 1

    if torch.cuda.is_available():
        visible_rank = local_rank if distributed else gpu_id
        torch.cuda.set_device(visible_rank)
        device = torch.device(f"cuda:{visible_rank}")
    else:
        device = torch.device("cpu")

    if distributed and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        timeout_minutes = float(os.environ.get("DDP_TIMEOUT_MINUTES", "60"))
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))

    return {
        "enabled": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "is_main": rank == 0,
    }


def cleanup_distributed(context):
    if context["enabled"] and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def maybe_barrier(context):
    if context["enabled"] and dist.is_available() and dist.is_initialized():
        dist.barrier()


def distributed_any(context, value):
    if not (context["enabled"] and dist.is_available() and dist.is_initialized()):
        return bool(value)
    flag = torch.tensor([1 if value else 0], device=context["device"], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _uses_chunked_ci_backward(configs):
    if not configs:
        return False
    ci_cfg = configs.get("model", {}).get("ci_params", {})
    try:
        return int(ci_cfg.get("backward_row_chunk_size", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _ddp_static_graph_enabled(configs=None):
    if _uses_chunked_ci_backward(configs):
        return False
    return os.environ.get("DDP_STATIC_GRAPH", "1") != "0"


def wrap_distributed_model(model, context, configs=None):
    if not context["enabled"]:
        return model
    ddp_kwargs = {
        "find_unused_parameters": False,
        "broadcast_buffers": False,
        "gradient_as_bucket_view": True,
    }
    if _ddp_static_graph_enabled(configs):
        ddp_kwargs["static_graph"] = True
    if context["device"].type == "cuda":
        return DDP(
            model,
            device_ids=[context["local_rank"]],
            output_device=context["local_rank"],
            **ddp_kwargs,
        )
    return DDP(model, **ddp_kwargs)


def _dataset_seq_length(dataset, index):
    if hasattr(dataset, "get_seq_length"):
        return int(dataset.get_seq_length(index))
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return _dataset_seq_length(dataset.dataset, int(dataset.indices[int(index)]))
    item = dataset[int(index)]
    if isinstance(item, dict):
        if "seq_length" in item:
            return int(item["seq_length"])
        if "sequence" in item:
            return len(str(item["sequence"]))
    return 0


def _worst_case_batch(loader, batch_size=None):
    dataset = loader.dataset
    if len(dataset) == 0:
        return None
    batch_size = int(batch_size or loader.batch_size or 1)
    ranked = sorted(
        range(len(dataset)),
        key=lambda idx: _dataset_seq_length(dataset, idx),
        reverse=True,
    )
    indices = ranked[:batch_size]
    if len(indices) < batch_size:
        indices += [indices[0]] * (batch_size - len(indices))
    return default_collate([dataset[idx] for idx in indices])


def _cuda_peak_message(device):
    if torch.device(device).type != "cuda":
        return ""
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    return f" peak_allocated={allocated:.2f}GiB peak_reserved={reserved:.2f}GiB"


def _is_cuda_oom(exc):
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _zero_lr_optimizer_step(optimizer):
    old_lrs = [group.get("lr", 0.0) for group in optimizer.param_groups]
    try:
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        optimizer.step()
    finally:
        for group, lr in zip(optimizer.param_groups, old_lrs):
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()


def run_memory_preflight(trainer, loaders, context):
    device = context["device"]
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    ci_cfg = trainer.model_cfgs.get("ci_params", {})
    train_batch = _worst_case_batch(loaders["train"], loaders["train"].batch_size)
    valid_batch = _worst_case_batch(loaders["valid"], loaders["valid"].batch_size)
    trainer.log_fn(
        "Memory preflight: worst train batch size={} valid batch size={} "
        "mutation_batch_size={} backward_row_chunk_size={}".format(
            len(train_batch["row_id"]) if train_batch is not None else 0,
            len(valid_batch["row_id"]) if valid_batch is not None else 0,
            ci_cfg.get("mutation_batch_size", 16),
            ci_cfg.get("backward_row_chunk_size", 0),
        )
    )

    try:
        if train_batch is not None:
            trainer.model.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            row_chunk_size = int(ci_cfg.get("backward_row_chunk_size", 0) or 0)
            if row_chunk_size > 0:
                loss_dct = trainer.backward_chunked_ci_loss(train_batch, row_chunk_size)
            else:
                loss_dct = trainer.loss_fn(train_batch)
                loss_dct["loss"].backward()
            _zero_lr_optimizer_step(trainer.optimizer)
            trainer.log_fn(
                "Memory preflight train passed: loss={:.4g}{}".format(
                    float(loss_dct["loss"].detach().cpu()),
                    _cuda_peak_message(device),
                )
            )

        if valid_batch is not None:
            trainer.model.eval()
            if torch.device(device).type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                loss_dct = trainer.loss_fn(valid_batch)
            trainer.log_fn(
                "Memory preflight valid passed: loss={:.4g}{}".format(
                    float(loss_dct["loss"].detach().cpu()),
                    _cuda_peak_message(device),
                )
            )
    except RuntimeError as exc:
        if _is_cuda_oom(exc):
            trainer.log_fn(
                "Memory preflight failed with CUDA OOM. Reduce one of: batch_size, "
                "model.ci_params.mutation_batch_size, model.ci_params.backward_row_chunk_size, "
                "or model.max_length."
            )
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        raise
    finally:
        trainer.optimizer.zero_grad(set_to_none=True)
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()


@record
def main():
    args = parse_args()
    context = setup_distributed(args.gpu_id)

    configs = resolve_config(args.config, data_dir=args.data_dir)
    if args.seed_override is not None:
        configs["setting"]["seed"] = int(args.seed_override)
    seed = configs["setting"]["seed"]
    set_all_seeds(seed + context["rank"])
    rng = np.random.RandomState(seed=seed)

    data_version = configs["setting"]["data_name"]
    model_key = args.model_key or configs["setting"].get("model_key", "")
    checkpoint = args.model_checkpoint or configs["setting"].get("model_checkpoint") or args.model_name
    if model_key and checkpoint == args.model_name and not args.model_checkpoint and not configs["setting"].get("model_checkpoint"):
        checkpoint = ""
    model_tag = args.model_name
    if model_key and args.model_name == "rna_lm":
        model_tag = model_key
    elif args.model_checkpoint and args.model_name == "rna_lm":
        model_tag = os.path.basename(args.model_checkpoint.rstrip("/"))
    configs.update(
        {
            "model_name": model_tag,
            "model_key": model_key,
            "model_checkpoint": checkpoint,
            "model_dir": args.model_dir or str(RESULTS_ROOT / "training" / configs["round"] / configs["model_alias"]),
            "testing": 0,
            "distributed": {
                "enabled": context["enabled"],
                "rank": context["rank"],
                "local_rank": context["local_rank"],
                "world_size": context["world_size"],
            },
        }
    )
    if args.max_length > 0:
        configs["model"]["max_length"] = args.max_length
    if context["is_main"]:
        try_makedir(configs["model_dir"])
    maybe_barrier(context)

    torch.backends.cudnn.benchmark = True
    device = context["device"]
    rna_model, tokenizer, hidden_size = load_rna_model(
        checkpoint,
        trust_remote_code=configs["setting"].get("trust_remote_code", True),
        cache_dir=configs["setting"].get("cache_dir", None),
        model_key=model_key,
        device=device,
    )
    configs = merge_overrides(configs, getattr(rna_model, "rcompass_config_overrides", {}))
    rna_model = rna_model.to(device)
    # hidden_size is kept for compatibility; Rcompass computes CI from language-head logits.
    trainer = RNAcompass_distill(rna_model, tokenizer, configs, hidden_size)

    initial_checkpoint = args.initial_checkpoint or configs["setting"].get("initial_checkpoint", "")
    if initial_checkpoint:
        if not Path(initial_checkpoint).is_file():
            raise FileNotFoundError(f"Initial distilled checkpoint not found: {initial_checkpoint}")
        trainer.log_fn(f"Load initial distilled model: {initial_checkpoint}")
        trainer.load_model(initial_checkpoint)
    if context["is_main"]:
        with (Path(configs["model_dir"]) / "training_config.json").open("w") as handle:
            import json
            json.dump(configs, handle, indent=2)

    train_fpath = args.train_file or configs["data"]["train_file"]
    valid_fpath = args.valid_file or configs["data"]["valid_file"]
    train_path = Path(train_fpath)
    if train_path.is_dir() and not valid_fpath:
        train_records = train_path / "train_records.pkl"
        valid_records = train_path / "valid_records.pkl"
        if train_records.exists() and valid_records.exists():
            train_fpath = str(train_records)
            valid_fpath = str(valid_records)

    val_bs = args.batch_size * args.val_batch_size_factor
    sampler_type = configs["training"].get("sampler_type", configs["training"].get("sampler", "balanced"))
    length_bucket_boundaries = configs["training"].get("length_bucket_boundaries")
    if valid_fpath:
        train_dct = load_ci_data(train_fpath)
        valid_dct = load_ci_data(valid_fpath)
        if context["is_main"]:
            trainer.log_fn(f"Train records: {train_fpath} ({len(train_dct)} RNAs)")
            trainer.log_fn(f"Valid records: {valid_fpath} ({len(valid_dct)} RNAs)")
        loaders = get_ci_loaders_from_splits(
            train_dct,
            valid_dct,
            args.batch_size,
            val_bs,
            rna_order=configs["setting"].get("data_order", 0),
            max_length=configs["model"].get("max_length", 1024),
            matrix_offset=configs["model"].get("matrix_offset", 1),
            max_special_tokens=configs["model"].get("max_special_tokens", 2),
            num_workers=configs["training"].get("num_workers", 4),
            distributed=context["enabled"],
            rank=context["rank"],
            world_size=context["world_size"],
            seed=seed,
            stem_candidate_cfg=configs["model"].get("stem_loss", {}),
            sampler_type=sampler_type,
            length_bucket_boundaries=length_bucket_boundaries,
        )
    else:
        data_dct = load_ci_data(train_fpath)
        rna_ids = rng.permutation(list(data_dct.keys()))
        if context["is_main"]:
            trainer.log_fn(f"Train/valid source: {train_fpath} ({len(data_dct)} RNAs)")
        loaders = get_ci_loaders(
            data_dct,
            rna_ids,
            args.batch_size,
            val_bs,
            val_frac=configs["setting"].get("val_frac", 0.1),
            rna_order=configs["setting"].get("data_order", 0),
            min_rnas=configs["setting"].get("min_valid_rnas", 200),
            max_length=configs["model"].get("max_length", 1024),
            matrix_offset=configs["model"].get("matrix_offset", 1),
            max_special_tokens=configs["model"].get("max_special_tokens", 2),
            num_workers=configs["training"].get("num_workers", 4),
            distributed=context["enabled"],
            rank=context["rank"],
            world_size=context["world_size"],
            seed=seed,
            stem_candidate_cfg=configs["model"].get("stem_loss", {}),
            sampler_type=sampler_type,
            length_bucket_boundaries=length_bucket_boundaries,
        )
    trainer.set_loaders(loaders)

    trainer.log_fn(f"Training {model_tag} on {data_version} for {configs['training']['nepochs']} epochs")
    trainer.log_fn("Skipping initial valid; first validation will run during training.")
    min_loss = float("inf")
    best_score = -float("inf")
    best_stem_score = -float("inf")

    trainer.init_training(len(loaders["train"]))
    if context["enabled"] and _uses_chunked_ci_backward(configs):
        trainer.log_fn("DDP static_graph disabled for chunked CI backward.")
    trainer.model = wrap_distributed_model(trainer.raw_model, context, configs)
    memory_preflight_only = args.memory_preflight_only or configs["training"].get("memory_preflight_only", False)
    if args.memory_preflight or memory_preflight_only or configs["training"].get("memory_preflight", False):
        run_memory_preflight(trainer, loaders, context)
        if memory_preflight_only:
            trainer.log_fn("Memory preflight only completed; exiting before training.")
            cleanup_distributed(context)
            return
    stopped_by_runtime_cap = False
    for epoch in range(configs["training"]["nepochs"]):
        train_sampler = getattr(loaders["train"], "sampler", None)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        min_loss, best_score, best_stem_score = trainer.train_epoch(
            epoch,
            [min_loss, best_score, best_stem_score],
            log_freq=configs["training"]["log_freq"],
        )
        should_stop = args.stop_after_epochs > 0 and (epoch + 1) >= args.stop_after_epochs
        should_stop = distributed_any(context, should_stop)
        if should_stop:
            stopped_by_runtime_cap = True
            trainer.log_fn(
                f"Runtime stop_after_epochs={args.stop_after_epochs}; exiting after epoch {epoch + 1} "
                f"while config nepochs remains {configs['training']['nepochs']}."
            )
            break
        if trainer.early_stops["counter"] >= trainer.early_stops["patience"] and epoch >= trainer.train_cfgs["minimum_epochs"]:
            break
        if context["is_main"]:
            print(f"End of epoch {epoch} training")

    trainer.log_fn(f"Best score {best_score:.4f}")
    trainer.log_fn(f"Best stem recall@topL {best_stem_score:.4f}")
    if stopped_by_runtime_cap:
        logging.shutdown()
        os._exit(0)
    if not stopped_by_runtime_cap:
        maybe_barrier(context)
    cleanup_distributed(context)


if __name__ == "__main__":
    main()
