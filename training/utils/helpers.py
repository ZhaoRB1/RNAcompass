import torch
import os, json
import pickle
import random
import numpy as np

def set_all_seeds(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def infer_hidden_size(model):
    config = getattr(model, "config", None)
    if config is not None:
        for name in ("hidden_size", "d_model", "n_embd", "embed_dim", "embedding_dim"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if embeddings is not None and hasattr(embeddings, "embedding_dim"):
        return int(embeddings.embedding_dim)
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            return int(module.embedding_dim)
    raise ValueError("Could not infer model hidden size from config or embeddings.")


def load_rna_model(model_name_or_path="", trust_remote_code=True, cache_dir=None, model_key="", device="cpu"):
    if model_key:
        from utils.rna_model_registry import load_local_collective_model

        checkpoint = model_name_or_path or None
        return load_local_collective_model(model_key, checkpoint=checkpoint, device=device)

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if not model_name_or_path:
        raise ValueError("model_name_or_path is required when model_key is not set.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    kwargs = {"trust_remote_code": trust_remote_code, "local_files_only": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **kwargs)
    return model, tokenizer, infer_hidden_size(model)


def load_model(model_name, inference=False):
    return load_rna_model(model_name)

def read_json(fpath):
    with open(fpath, 'r') as file:
        data = json.load(file)
    return data

def read_pkl(fpath):
    with open(fpath, 'rb') as fp:
        dct = pickle.load(fp)
    return dct

def dict2json(json_pth, dct):
    with open(json_pth, "w") as outfile:
        json.dump(dct, outfile, indent=2)

def try_makedir(fdir):
    os.makedirs(fdir, exist_ok=True)

def write_pkl(dct, fpath):
    with open(fpath, 'wb') as fp:
        pickle.dump(dct, fp)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        n = n.item() if torch.is_tensor(n) else n
        self.val = val
        if n <= 0:
            return
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeterList(object):
    def __init__(self, measure_dct):
        # measures: "avg", "cat", "list"
        self.measure_dct = measure_dct
        self.meter_lst = {}
        self.reset()
    def reset(self):
        for m, metric_type in self.measure_dct.items():
            if metric_type == "avg":
                self.meter_lst[m] = AverageMeter()
            else:
                self.meter_lst[m] = []
    def update(self, val_dct, n=1):
        for m, metric_type in self.measure_dct.items():
            if metric_type == "avg":
                x = val_dct[m]
                x = x.item() if torch.is_tensor(x) else x
                metric_n = val_dct.get(f"{m}_weight", n)
                metric_n = metric_n.item() if torch.is_tensor(metric_n) else metric_n
                self.meter_lst[m].update(x, metric_n)
            elif metric_type == "cat":
                self.meter_lst[m].append(val_dct[m].cpu())
            elif metric_type == "list":
                self.meter_lst[m] += val_dct[m]
    def get_avg(self):
        return_dct = {}
        for m, metric_type in self.measure_dct.items():
            if metric_type == "avg":
                return_dct[m] = self.meter_lst[m].avg
            elif metric_type == "cat":
                return_dct[m] = torch.cat(self.meter_lst[m]).numpy()
            elif metric_type == "list":
                return_dct[m] = self.meter_lst[m]
        return return_dct

def get_stats(data_dct):
    n = 0; mu = 0; ss = 0
    min_e = 1000; max_e = -1000
    for _, arr in data_dct.items():
        size = arr.shape[0] * arr.shape[1]
        arr_mean = arr.mean()
        arr_std = arr.std()
        n += size
        mu += arr_mean * size
        ss_arr = (arr_std**2 + arr_mean**2) * size
        ss += ss_arr
    # aggregate
    mu = mu / n
    variance = ss/n - mu**2
    return mu, np.sqrt(variance) #, min_e, max_e)
