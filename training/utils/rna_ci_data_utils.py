import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from utils.local_paths import is_readable_file, localize_path


SEQUENCE_KEYS = ("sequence", "seq", "rna_sequence", "wt_seq")
CI_KEYS = ("ci_matrix", "ci", "target_ci", "contact_map", "contacts")
CI_PATH_KEYS = ("ci_matrix_path", "ci_path", "target_ci_path", "matrix_path", "output_file")
MASK_KEYS = ("ci_mask", "mask", "target_mask", "contact_mask")
MASK_PATH_KEYS = ("ci_mask_path", "mask_path", "target_mask_path", "contact_mask_path")
FILTERED_PAIR_PATH_KEYS = ("filtered_pairs_path", "stem_pairs_path")
ALLOWED_SEQUENCE_CHARS = frozenset("ACGUN")


def _first_present(record, keys):
    for key in keys:
        if key in record:
            return record[key]
    return None


def _normalize_sequence(sequence):
    return str(sequence or "").strip().upper()


def _sequence_from_record(record):
    if isinstance(record, dict):
        return _first_present(record, SEQUENCE_KEYS)
    if isinstance(record, (tuple, list)) and len(record) >= 1:
        return record[0]
    return None


def invalid_sequence_chars(sequence):
    return sorted(set(_normalize_sequence(sequence)).difference(ALLOWED_SEQUENCE_CHARS))


def is_valid_training_sequence(sequence):
    sequence = _normalize_sequence(sequence)
    return bool(sequence) and not invalid_sequence_chars(sequence)


def filter_valid_sequence_records(data_dct, ids=None):
    ids = list(data_dct.keys()) if ids is None else list(ids)
    filtered = {}
    skipped = {}
    for seq_id in ids:
        record = data_dct[seq_id]
        sequence = _sequence_from_record(record)
        if is_valid_training_sequence(sequence):
            filtered[seq_id] = record
        else:
            skipped[seq_id] = invalid_sequence_chars(sequence) or ["<empty>"]
    return filtered, skipped


def _require_valid_sequence_records(data_dct, ids=None):
    ids = list(data_dct) if ids is None else list(ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sequence IDs in the selected training records")
    records, skipped = filter_valid_sequence_records(data_dct, ids=ids)
    if skipped:
        examples = "; ".join(f"{seq_id}: {chars}" for seq_id, chars in list(skipped.items())[:5])
        raise ValueError(f"Invalid RNA sequences; expected non-empty A/C/G/U/N sequences: {examples}")
    return records


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(text, source):
    try:
        return json.loads(text, object_pairs_hook=_unique_json_object)
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from exc


def _add_record(records, seq_id, record, source):
    if seq_id is None or not str(seq_id).strip():
        raise ValueError(f"{source}: sequence ID must not be empty")
    seq_id = str(seq_id)
    if seq_id in records:
        raise ValueError(f"{source}: duplicate sequence ID {seq_id!r}")
    records[seq_id] = record


def _add_manifest_record(records, record, index, source):
    if not isinstance(record, dict):
        raise ValueError(f"{source}: each manifest entry must be an object")
    seq_id = record.get("id", record.get("sequence_name", index))
    _add_record(records, seq_id, record, source)


def _as_tensor_2d(value, name):
    tensor = torch.as_tensor(value)
    if torch.is_complex(tensor):
        raise ValueError(f"{name} must contain real values")
    tensor = tensor.float()
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix; got shape {tuple(tensor.shape)}")
    return tensor


def _resolve_readable_path(path, base_dir=None):
    return localize_path(path, base_dir=base_dir, require_readable_file=True)


def _load_array(value, base_dir=None):
    if not isinstance(value, (str, Path)):
        return value
    path = _resolve_readable_path(value, base_dir=base_dir)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "ci_matrix" in data:
                return data["ci_matrix"]
            if not data.files:
                raise ValueError(f"Empty CI matrix archive: {path}")
            return data[data.files[0]]
    raise ValueError(f"Unsupported CI matrix file type: {path}")


def _read_reference_sequence(csv_path):
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        first_sequence = None
        for row in reader:
            sequence = row.get("sequence")
            if not sequence:
                continue
            if first_sequence is None:
                first_sequence = sequence
            if str(row.get("id", "")).strip() == "0" or str(row.get("mutation_info", "")).lower() == "reference":
                return _normalize_sequence(sequence)
        if first_sequence is not None:
            return _normalize_sequence(first_sequence)
    return None


def _read_reference_sequence_from_source_paths(metadata):
    sources = metadata.get("sources") or {}
    if not isinstance(sources, dict):
        return None
    for source_path in sources.values():
        source_path = localize_path(source_path, require_readable_file=True)
        rf_id = source_path.parent.name
        candidates = (
            source_path.parent / f"{rf_id}_mutations.csv",
            source_path.parent / source_path.name.replace("_full_ci_matrix_raw.npy", "_mutations.csv"),
            source_path.parent / source_path.name.replace("_ci_matrix_raw.npy", "_mutations.csv"),
        )
        for csv_path in candidates:
            if is_readable_file(csv_path):
                sequence = _read_reference_sequence(csv_path)
                if sequence:
                    return sequence
    return None


def _resolve_metadata_output_file(metadata, metadata_path):
    result = metadata.get("full_result") or {}
    output_file = result.get("output_file") or metadata.get("output_file")
    if output_file is None and metadata.get("sliding_windows"):
        output_file = metadata["sliding_windows"][0].get("output_file")
    if output_file is None:
        raise KeyError(f"No CI output_file found in metadata: {metadata_path}")
    path = Path(output_file)
    return _resolve_readable_path(path, base_dir=Path(metadata_path).parent)


def _record_from_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    with open(metadata_path, "r") as handle:
        metadata = _parse_json(handle.read(), metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path}: metadata must be an object")
    seq_id = metadata.get("sequence_name") or metadata.get("rf_id") or metadata_path.name.replace("_metadata.json", "")
    sequence = metadata.get("sequence") or metadata.get("reference_sequence")
    if sequence is None:
        csv_path = metadata_path.parent / f"{seq_id}_mutations.csv"
        if csv_path.exists():
            sequence = _read_reference_sequence(csv_path)
    if sequence is None:
        sequence = _read_reference_sequence_from_source_paths(metadata)
    if sequence is None:
        raise ValueError(
            f"Cannot recover RNA sequence for {seq_id}. Provide sequence/reference_sequence "
            f"in {metadata_path} or keep {seq_id}_mutations.csv next to it."
        )
    return seq_id, {
        "sequence": _normalize_sequence(sequence),
        "ci_matrix_path": str(_resolve_metadata_output_file(metadata, metadata_path)),
        "metadata_path": str(metadata_path),
        "_base_dir": str(metadata_path.parent),
    }


def _load_json_records(path):
    with open(path, "r") as handle:
        data = _parse_json(handle.read(), path)
    if isinstance(data, dict) and ("full_result" in data or "output_file" in data):
        seq_id, record = _record_from_metadata(path)
        return {seq_id: record}
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        records = {}
        for idx, record in enumerate(data):
            _add_manifest_record(records, record, idx, f"{path}: entry {idx + 1}")
        return records
    raise ValueError(f"Unsupported JSON CI data format: {path}")


def _load_ci_data(path):
    path = Path(path)
    if path.is_dir():
        records = {}
        for metadata_path in sorted(path.rglob("*_metadata.json")):
            seq_id, record = _record_from_metadata(metadata_path)
            _add_record(records, seq_id, record, metadata_path)
        if not records:
            raise FileNotFoundError(f"No *_metadata.json files found under {path}")
        return records
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    if suffix == ".json":
        return _load_json_records(path)
    if suffix == ".jsonl":
        records = {}
        with open(path, "r") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                source = f"{path}:{idx + 1}"
                record = _parse_json(line, source)
                _add_manifest_record(records, record, idx, source)
        return records
    raise ValueError(f"Unsupported CI data source: {path}")


def load_ci_data(path):
    path = Path(path).resolve()
    loaded = _load_ci_data(path)
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError(f"{path}: expected a non-empty collection of training records")
    records = {}
    for seq_id, record in loaded.items():
        _add_record(records, seq_id, record, path)
    for record in records.values():
        if isinstance(record, dict):
            record.setdefault("_base_dir", str(path.parent if path.is_file() else path))
    return records


def _extract_record(record):
    if isinstance(record, dict):
        sequence = _first_present(record, SEQUENCE_KEYS)
        ci_matrix = _first_present(record, CI_KEYS)
        if ci_matrix is None:
            ci_matrix = _first_present(record, CI_PATH_KEYS)
        ci_mask = _first_present(record, MASK_KEYS)
        if ci_mask is None:
            ci_mask = _first_present(record, MASK_PATH_KEYS)
        base_dir = record.get("_base_dir")
    elif isinstance(record, (tuple, list)) and len(record) >= 2:
        sequence, ci_matrix = record[0], record[1]
        ci_mask = record[2] if len(record) >= 3 else None
        base_dir = None
    else:
        raise ValueError(
            "Each record must be a dict with sequence and ci_matrix, "
            "or a tuple/list like (sequence, ci_matrix[, ci_mask])."
        )
    if sequence is None or ci_matrix is None:
        raise KeyError(
            "Missing RNA sequence or CI matrix. Supported sequence keys: "
            f"{SEQUENCE_KEYS}; supported CI keys: {CI_KEYS}."
        )
    ci_matrix = _load_array(ci_matrix, base_dir=base_dir)
    ci_mask = _load_array(ci_mask, base_dir=base_dir) if ci_mask is not None else None
    return _normalize_sequence(sequence), ci_matrix, ci_mask


def _validated_ci_record(seq_id, record):
    try:
        sequence, ci_matrix, ci_mask = _extract_record(record)
        if not is_valid_training_sequence(sequence):
            raise ValueError("expected a non-empty A/C/G/U/N sequence")
        ci_matrix = _as_tensor_2d(ci_matrix, "ci_matrix")
        expected_shape = (len(sequence), len(sequence))
        if tuple(ci_matrix.shape) != expected_shape:
            raise ValueError(f"ci_matrix shape {tuple(ci_matrix.shape)} must match sequence shape {expected_shape}")
        if not torch.isfinite(ci_matrix).all():
            raise ValueError("ci_matrix contains NaN or infinite values")
        if ci_mask is None:
            ci_mask = torch.ones_like(ci_matrix)
        else:
            ci_mask = _as_tensor_2d(ci_mask, "ci_mask")
            if tuple(ci_mask.shape) != expected_shape:
                raise ValueError(f"ci_mask shape {tuple(ci_mask.shape)} must match sequence shape {expected_shape}")
            if not torch.isfinite(ci_mask).all() or (ci_mask < 0).any():
                raise ValueError("ci_mask must contain finite, non-negative values")
        if not (ci_mask > 0).any():
            raise ValueError("ci_mask has no supervised cells")
        return sequence, ci_matrix, ci_mask
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"Training record {seq_id!r}: {exc}") from exc


def _stem_candidates_enabled(stem_candidate_cfg):
    return bool(stem_candidate_cfg and stem_candidate_cfg.get("enabled", False))


def _read_filtered_pair_positions(path):
    positions = []
    seen = set()
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        required = {"pos_i_0based", "pos_j_0based"}
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError(f"{path}: expected unique TSV columns pos_i_0based and pos_j_0based")
        for row in reader:
            source = f"{path}:{reader.line_num}"
            if None in row:
                raise ValueError(f"{source}: too many TSV columns")
            try:
                i = int(row["pos_i_0based"])
                j = int(row["pos_j_0based"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}: candidate coordinates must be integers") from exc
            if i < 0 or j < 0 or i == j:
                raise ValueError(f"{source}: invalid candidate pair ({i}, {j})")
            pair = (min(i, j), max(i, j))
            if pair in seen:
                raise ValueError(f"{source}: duplicate candidate pair {pair}")
            seen.add(pair)
            positions.append((i, j))
    return positions


def _first_readable_path(paths):
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if is_readable_file(path):
            return path
    return None


def _filtered_pair_path_candidates(seq_id, record, stem_candidate_cfg):
    if isinstance(record, dict):
        for key in FILTERED_PAIR_PATH_KEYS:
            if key in record:
                yield localize_path(record[key], base_dir=record.get("_base_dir"))

    if not _stem_candidates_enabled(stem_candidate_cfg):
        return

    candidate_root = stem_candidate_cfg.get("candidate_root") or stem_candidate_cfg.get("root")
    if not candidate_root:
        return
    candidate_root = Path(candidate_root).expanduser()
    pair_file_name = stem_candidate_cfg.get("pair_file_name", "filtered_pairs.tsv")
    model = stem_candidate_cfg.get("model") or stem_candidate_cfg.get("candidate_model")

    seq_ids = [str(seq_id)]
    if isinstance(record, dict):
        for key in ("sequence_id", "safe_sequence_name"):
            record_seq_id = record.get(key)
            if record_seq_id and str(record_seq_id) not in seq_ids:
                seq_ids.append(str(record_seq_id))

    for current_seq_id in seq_ids:
        bases = (candidate_root / "results" / current_seq_id, candidate_root / current_seq_id)
        for base in bases:
            if model:
                yield base / str(model) / pair_file_name
            else:
                for path in sorted(base.glob(f"*/{pair_file_name}")) if base.exists() else []:
                    yield path


def _load_filtered_pair_positions(seq_id, record, stem_candidate_cfg):
    if isinstance(record, dict):
        for key in FILTERED_PAIR_PATH_KEYS:
            if key not in record:
                continue
            value = record[key]
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{seq_id}: {key} must name a candidate TSV file")
            path = localize_path(value, base_dir=record.get("_base_dir"))
            if not is_readable_file(path):
                raise FileNotFoundError(f"{seq_id}: candidate file is missing or unreadable: {path}")
            return _read_filtered_pair_positions(path)
    path = _first_readable_path(_filtered_pair_path_candidates(seq_id, record, stem_candidate_cfg))
    if path is None:
        if _stem_candidates_enabled(stem_candidate_cfg):
            raise FileNotFoundError(
                f"{seq_id}: candidate supervision is enabled but no candidate TSV was found; "
                "set filtered_pairs_path to an existing file"
            )
        return []
    return _read_filtered_pair_positions(path)


class CiMatrixDataset(Dataset):
    def __init__(
        self,
        data_dct,
        max_length=1024,
        matrix_offset=1,
        max_special_tokens=2,
        rna_order=0,
        stem_candidate_cfg=None,
    ):
        self.data_dct = _require_valid_sequence_records(data_dct)
        self.seq_ids = list(self.data_dct)
        self.skipped_invalid_sequences = {}
        self.max_length = int(max_length)
        self.matrix_offset = int(matrix_offset)
        self.max_special_tokens = int(max_special_tokens)
        if not 0 <= self.matrix_offset <= self.max_special_tokens < self.max_length:
            raise ValueError("Expected 0 <= matrix_offset <= max_special_tokens < max_length")
        self.stem_candidate_cfg = stem_candidate_cfg or {}
        self.filtered_pair_positions = {}
        means = []
        for seq_id in self.seq_ids:
            record = self.data_dct[seq_id]
            sequence, ci_matrix, ci_mask = _validated_ci_record(seq_id, record)
            positions = _load_filtered_pair_positions(seq_id, record, self.stem_candidate_cfg)
            for pos_i, pos_j in positions:
                if pos_i >= len(sequence) or pos_j >= len(sequence):
                    raise ValueError(
                        f"{seq_id}: candidate pair ({pos_i}, {pos_j}) is outside "
                        f"sequence length {len(sequence)}"
                    )
            self.filtered_pair_positions[seq_id] = positions
            length = min(len(sequence), self.max_length - self.max_special_tokens)
            if not (ci_mask[:length, :length] > 0).any():
                raise ValueError(f"{seq_id}: no supervised cells remain after max_length cropping")
            if rna_order:
                means.append(np.nanmean(np.asarray(ci_matrix, dtype=np.float32)))
        if rna_order:
            self.seq_ids = list(np.asarray(self.seq_ids)[np.argsort(means)])
        self.seq_lengths = [self._estimate_length(self.data_dct[seq_id]) for seq_id in self.seq_ids]

    def _estimate_length(self, record):
        if isinstance(record, dict):
            sequence = _first_present(record, SEQUENCE_KEYS)
        elif isinstance(record, (tuple, list)) and len(record) >= 1:
            sequence = record[0]
        else:
            sequence = ""
        available = max(1, self.max_length - self.max_special_tokens)
        return min(len(_normalize_sequence(sequence)), available)

    def get_seq_length(self, index):
        return int(self.seq_lengths[int(index)])

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, item):
        seq_id = self.seq_ids[item]
        sequence, ci_matrix, ci_mask = _validated_ci_record(seq_id, self.data_dct[seq_id])

        available = self.max_length - self.max_special_tokens
        length = min(len(sequence), available)
        if not (ci_mask[:length, :length] > 0).any():
            raise ValueError(f"{seq_id}: no supervised cells remain after max_length cropping")

        padded_ci = torch.zeros((self.max_length, self.max_length), dtype=torch.float32)
        padded_mask = torch.zeros((self.max_length, self.max_length), dtype=torch.float32)
        filtered_pair_mask = torch.zeros((self.max_length, self.max_length), dtype=torch.float32)
        start = self.matrix_offset
        stop = start + length
        padded_ci[start:stop, start:stop] = ci_matrix[:length, :length]
        padded_mask[start:stop, start:stop] = ci_mask[:length, :length]
        for pos_i, pos_j in self.filtered_pair_positions.get(seq_id, []):
            pos_i = int(pos_i)
            pos_j = int(pos_j)
            if pos_i >= len(sequence) or pos_j >= len(sequence):
                raise ValueError(f"{seq_id}: candidate coordinates exceed the current sequence length")
            if pos_i >= length or pos_j >= length:
                continue
            row = start + pos_i
            col = start + pos_j
            filtered_pair_mask[row, col] = 1.0
            filtered_pair_mask[col, row] = 1.0

        return {
            "row_id": seq_id,
            "sequence": sequence[:length],
            "ci_matrix": padded_ci,
            "ci_mask": padded_mask,
            "filtered_pair_mask": filtered_pair_mask,
            "seq_length": length,
        }


class LengthBalancedDistributedSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas, rank, shuffle=True, seed=0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        global_batch = max(1, self.batch_size * self.num_replicas)
        self.num_batches = (len(self.dataset) + global_batch - 1) // global_batch
        self.num_samples = self.num_batches * self.batch_size
        self.total_size = self.num_batches * global_batch

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _cost(self, index):
        length = max(1, int(self.dataset.get_seq_length(index)))
        return length * length * length

    def __iter__(self):
        if len(self.dataset) == 0:
            return iter([])
        indices = list(range(len(self.dataset)))
        indices.sort(key=self._cost, reverse=True)
        if len(indices) < self.total_size:
            repeats = self.total_size - len(indices)
            padding = list(reversed(indices))
            indices += (padding * ((repeats + len(padding) - 1) // len(padding)))[:repeats]
        else:
            indices = indices[:self.total_size]
        global_batch = self.batch_size * self.num_replicas
        groups = [indices[i:i + global_batch] for i in range(0, self.total_size, global_batch)]
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(groups), generator=generator).tolist()
            groups = [groups[i] for i in order]
        rank_indices = []
        for group in groups:
            bins = [[] for _ in range(self.num_replicas)]
            loads = [0 for _ in range(self.num_replicas)]
            for index in sorted(group, key=self._cost, reverse=True):
                candidates = [r for r in range(self.num_replicas) if len(bins[r]) < self.batch_size]
                target = min(candidates, key=lambda r: loads[r])
                bins[target].append(index)
                loads[target] += self._cost(index)
            rank_indices.extend(bins[self.rank])
        return iter(rank_indices)


class LengthBucketDistributedSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas,
        rank,
        boundaries=None,
        shuffle=True,
        seed=0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.boundaries = tuple(int(x) for x in (boundaries or (128, 256, 512, 768)))
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive.")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank={self.rank} is outside [0, {self.num_replicas}).")
        if any(boundary <= 0 for boundary in self.boundaries):
            raise ValueError("length bucket boundaries must be positive.")
        if tuple(sorted(self.boundaries)) != self.boundaries:
            raise ValueError("length bucket boundaries must be sorted ascending.")

        self.global_batch = max(1, self.batch_size * self.num_replicas)
        self.bucket_indices = self._bucket_dataset_indices()
        self.total_size = sum(self._padded_bucket_size(len(indices)) for indices in self.bucket_indices)
        self.num_batches = self.total_size // self.global_batch if self.global_batch > 0 else 0
        self.num_samples = self.total_size // self.num_replicas if self.num_replicas > 0 else 0

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _cost(self, index):
        length = max(1, int(self.dataset.get_seq_length(index)))
        return length * length * length

    def _bucket_id(self, index):
        length = int(self.dataset.get_seq_length(index))
        for bucket_id, boundary in enumerate(self.boundaries):
            if length <= boundary:
                return bucket_id
        return len(self.boundaries)

    def _bucket_dataset_indices(self):
        buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for index in range(len(self.dataset)):
            buckets[self._bucket_id(index)].append(index)
        return [indices for indices in buckets if indices]

    def _padded_bucket_size(self, size):
        if size <= 0:
            return 0
        return ((int(size) + self.global_batch - 1) // self.global_batch) * self.global_batch

    def _shuffle_indices(self, indices, generator):
        if not self.shuffle or len(indices) <= 1:
            return list(indices)
        order = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[i] for i in order]

    def _pad_bucket(self, indices):
        padded_size = self._padded_bucket_size(len(indices))
        if len(indices) >= padded_size:
            return list(indices)
        need = padded_size - len(indices)
        padding = list(reversed(indices))
        indices = list(indices)
        indices += (padding * ((need + len(padding) - 1) // len(padding)))[:need]
        return indices

    def __iter__(self):
        if len(self.dataset) == 0:
            return iter([])

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        groups = []
        for bucket_indices in self.bucket_indices:
            indices = self._shuffle_indices(bucket_indices, generator)
            indices = self._pad_bucket(indices)
            groups.extend(
                indices[offset:offset + self.global_batch]
                for offset in range(0, len(indices), self.global_batch)
            )

        if self.shuffle and len(groups) > 1:
            order = torch.randperm(len(groups), generator=generator).tolist()
            groups = [groups[i] for i in order]

        rank_indices = []
        for group in groups:
            bins = [[] for _ in range(self.num_replicas)]
            loads = [0 for _ in range(self.num_replicas)]
            for index in sorted(group, key=self._cost, reverse=True):
                candidates = [r for r in range(self.num_replicas) if len(bins[r]) < self.batch_size]
                target = min(candidates, key=lambda r: loads[r])
                bins[target].append(index)
                loads[target] += self._cost(index)
            rank_indices.extend(bins[self.rank])
        return iter(rank_indices)


def _make_distributed_train_sampler(
    dataset,
    batch_size,
    world_size,
    rank,
    shuffle,
    seed,
    sampler_type="balanced",
    length_bucket_boundaries=None,
):
    sampler_type = str(sampler_type or "balanced").lower()
    if sampler_type in {"balanced", "length_balanced"}:
        return LengthBalancedDistributedSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=shuffle,
            seed=int(seed),
        )
    if sampler_type in {"length_bucket", "bucket", "bucketed"}:
        return LengthBucketDistributedSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=int(world_size),
            rank=int(rank),
            boundaries=length_bucket_boundaries,
            shuffle=shuffle,
            seed=int(seed),
        )
    raise ValueError(
        "Unknown distributed train sampler "
        f"{sampler_type!r}; expected 'balanced' or 'length_bucket'."
    )


def _make_loader(dataset, batch_size, shuffle, num_workers, sampler=None):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "sampler": sampler,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def get_ci_loaders(
    data_dct,
    dev_ids,
    batch_size,
    val_batch_size,
    val_frac=0.1,
    rna_order=0,
    min_rnas=200,
    max_length=1024,
    matrix_offset=1,
    max_special_tokens=2,
    num_workers=4,
    distributed=False,
    rank=0,
    world_size=1,
    seed=0,
    stem_candidate_cfg=None,
    sampler_type="balanced",
    length_bucket_boundaries=None,
):
    data_dct = _require_valid_sequence_records(data_dct, ids=dev_ids)
    dev_ids = list(data_dct.keys())
    n = len(dev_ids)
    if n < 2:
        raise ValueError("Need at least two RNA records to build train/valid loaders")
    nvals = max(min(min_rnas, int(n * val_frac)), min(10, n // 2))
    nvals = min(max(1, nvals), n - 1)
    ntrains = n - nvals
    train_ids = dev_ids[:ntrains]
    valid_ids = dev_ids[ntrains:]
    train_dct = {uid: data_dct[uid] for uid in train_ids}
    valid_dct = {uid: data_dct[uid] for uid in valid_ids}

    train_dataset = CiMatrixDataset(
        train_dct,
        max_length=max_length,
        matrix_offset=matrix_offset,
        max_special_tokens=max_special_tokens,
        rna_order=rna_order,
        stem_candidate_cfg=stem_candidate_cfg,
    )
    valid_dataset = CiMatrixDataset(
        valid_dct,
        max_length=max_length,
        matrix_offset=matrix_offset,
        max_special_tokens=max_special_tokens,
        rna_order=0,
        stem_candidate_cfg=stem_candidate_cfg,
    )
    train_sampler = None
    if distributed:
        train_sampler = _make_distributed_train_sampler(
            train_dataset,
            batch_size=batch_size,
            world_size=int(world_size),
            rank=int(rank),
            shuffle=(rna_order == 0),
            seed=int(seed),
            sampler_type=sampler_type,
            length_bucket_boundaries=length_bucket_boundaries,
        )
        valid_indices = list(range(int(rank), len(valid_dataset), int(world_size)))
        valid_dataset = Subset(valid_dataset, valid_indices)

    return {
        "train": _make_loader(
            train_dataset,
            batch_size,
            rna_order == 0,
            num_workers,
            sampler=train_sampler,
        ),
        "valid": _make_loader(
            valid_dataset,
            val_batch_size,
            False,
            num_workers,
        ),
    }


def get_ci_loaders_from_splits(
    train_dct,
    valid_dct,
    batch_size,
    val_batch_size,
    rna_order=0,
    max_length=1024,
    matrix_offset=1,
    max_special_tokens=2,
    num_workers=4,
    distributed=False,
    rank=0,
    world_size=1,
    seed=0,
    stem_candidate_cfg=None,
    sampler_type="balanced",
    length_bucket_boundaries=None,
):
    train_dct = _require_valid_sequence_records(train_dct)
    valid_dct = _require_valid_sequence_records(valid_dct)
    if len(train_dct) < 1:
        raise ValueError("Need at least one RNA record to build the train loader")
    if len(valid_dct) < 1:
        raise ValueError("Need at least one RNA record to build the valid loader")

    train_dataset = CiMatrixDataset(
        train_dct,
        max_length=max_length,
        matrix_offset=matrix_offset,
        max_special_tokens=max_special_tokens,
        rna_order=rna_order,
        stem_candidate_cfg=stem_candidate_cfg,
    )
    valid_dataset = CiMatrixDataset(
        valid_dct,
        max_length=max_length,
        matrix_offset=matrix_offset,
        max_special_tokens=max_special_tokens,
        rna_order=0,
        stem_candidate_cfg=stem_candidate_cfg,
    )

    train_sampler = None
    if distributed:
        train_sampler = _make_distributed_train_sampler(
            train_dataset,
            batch_size=batch_size,
            world_size=int(world_size),
            rank=int(rank),
            shuffle=(rna_order == 0),
            seed=int(seed),
            sampler_type=sampler_type,
            length_bucket_boundaries=length_bucket_boundaries,
        )
        valid_indices = list(range(int(rank), len(valid_dataset), int(world_size)))
        valid_dataset = Subset(valid_dataset, valid_indices)

    return {
        "train": _make_loader(
            train_dataset,
            batch_size,
            rna_order == 0,
            num_workers,
            sampler=train_sampler,
        ),
        "valid": _make_loader(
            valid_dataset,
            val_batch_size,
            False,
            num_workers,
        ),
    }
