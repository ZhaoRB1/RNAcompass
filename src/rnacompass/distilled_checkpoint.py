from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

import torch


Logger = Optional[Callable[[str], None]]
StateTransform = Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]]


def _log(logger: Logger, message: str) -> None:
    if logger is not None:
        logger(message)


def _torch_load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _unwrap_state_dict(payload) -> Mapping[str, torch.Tensor]:
    state = payload
    if isinstance(state, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            nested = state.get(key)
            if isinstance(nested, Mapping):
                state = nested
                break
    if not isinstance(state, Mapping):
        raise ValueError(f"Unsupported checkpoint payload type: {type(state)!r}")
    return state


def _normalize_key(key: object) -> str:
    text = str(key)
    for prefix in ("module.", "_orig_mod."):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text


def _normalize_state_dict(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {_normalize_key(key): value for key, value in state.items()}


def _prefixed_view(state: Mapping[str, torch.Tensor], prefix: str) -> Optional[Dict[str, torch.Tensor]]:
    selected = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix) and len(key) > len(prefix)
    }
    return selected or None


def _candidate_state_dicts(state: Mapping[str, torch.Tensor]) -> Iterable[Tuple[str, Dict[str, torch.Tensor]]]:
    normalized = _normalize_state_dict(state)
    yield "as_saved", dict(normalized)

    for prefix in (
        "model.",
        "encoder.",
        "model.model.",
        "model.encoder.",
        "raw_model.",
        "raw_model.model.",
        "raw_model.encoder.",
    ):
        view = _prefixed_view(normalized, prefix)
        if view:
            yield f"strip:{prefix}", view


def _summarize(keys: Iterable[str], limit: int = 8) -> str:
    keys = list(keys)
    preview = ", ".join(keys[:limit])
    if len(keys) > limit:
        preview += f", ... ({len(keys)} total)"
    return preview


def load_distilled_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    logger: Logger = None,
    state_transform: StateTransform = None,
    ignored_missing_suffixes: Tuple[str, ...] = (),
) -> str:
    """Load an RNAcompass distilled state_dict into an already constructed model.

    RNAcompass saves adapter state_dicts for some backbones, so keys may be
    prefixed with ``model.`` or ``encoder.`` even though the downstream raw-CI
    script holds the underlying backbone directly.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Distilled checkpoint does not exist: {path}")

    state = _unwrap_state_dict(_torch_load_checkpoint(path))
    strict_errors = []
    for candidate_name, candidate in _candidate_state_dicts(state):
        if state_transform is not None:
            candidate = state_transform(candidate)
        try:
            model.load_state_dict(candidate, strict=True)
            _log(logger, f"Loaded distilled checkpoint {path} with {candidate_name} keys.")
            return candidate_name
        except RuntimeError as exc:
            strict_errors.append((candidate_name, str(exc)))

    target_keys = set(model.state_dict().keys())
    best = None
    for candidate_name, candidate in _candidate_state_dicts(state):
        if state_transform is not None:
            candidate = state_transform(candidate)
        candidate_keys = set(candidate.keys())
        matched = len(target_keys & candidate_keys)
        missing = sorted(target_keys - candidate_keys)
        unexpected = sorted(candidate_keys - target_keys)
        score = (matched, -len(missing), -len(unexpected))
        if best is None or score > best[0]:
            best = (score, candidate_name, candidate, missing, unexpected)

    if best is not None:
        _score, candidate_name, candidate, missing, unexpected = best
        allowed_missing = [
            key for key in missing if any(key.endswith(suffix) for suffix in ignored_missing_suffixes)
        ]
        if not unexpected and len(allowed_missing) == len(missing):
            model.load_state_dict(candidate, strict=False)
            _log(
                logger,
                f"Loaded distilled checkpoint {path} with {candidate_name} keys "
                f"(ignored missing: {_summarize(missing)}).",
            )
            return candidate_name
        message = (
            f"Could not strictly load distilled checkpoint {path}. "
            f"Best candidate={candidate_name}; missing={_summarize(missing)}; "
            f"unexpected={_summarize(unexpected)}"
        )
    elif strict_errors:
        message = f"Could not load distilled checkpoint {path}: {strict_errors[0][1]}"
    else:
        message = f"Could not load distilled checkpoint {path}: no state_dict candidates found"
    raise RuntimeError(message)
