"""Paths shared by the portable training entry points."""
import os
from pathlib import Path

from rnacompass.weights import resolve_weights_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DATA_ROOT = Path(os.environ.get('RNACOMPASS_DATA_DIR', REPOSITORY_ROOT / 'data')).expanduser()
RESULTS_ROOT = Path(os.environ.get('RNACOMPASS_RESULTS_DIR', REPOSITORY_ROOT / 'results')).expanduser()


def is_readable_file(path):
    path = Path(path)
    return path.is_file() and os.access(path, os.R_OK)


def localize_path(path, base_dir=None, require_readable_file=False):
    path = Path(path).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path.resolve()


def resolve_config(path, data_dir=None, results_dir=None):
    import json
    path = Path(path)
    if not path.is_file():
        path = PROJECT_ROOT / 'configs' / (str(path).removesuffix('.json') + '.json')
    with path.open() as handle:
        config = json.load(handle)
    replacements = {
        '{data_root}': str(Path(data_dir or DATA_ROOT / 'training').resolve()),
        '{weights_root}': str(resolve_weights_dir()),
        '{results_root}': str(Path(results_dir or RESULTS_ROOT).resolve()),
    }
    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            for key, replacement in replacements.items():
                value = value.replace(key, replacement)
        return value
    return expand(config)
