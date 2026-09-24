"""Environment, CUDA, packaged-resource and checkpoint validation."""

from __future__ import annotations

import importlib
import platform
import sys
from importlib import metadata, resources
from pathlib import Path

from .inference import patch_omegaconf_get_ref_type
from .weights import resolve_weights_dir, verify_weights


EXPECTED_PYTHON = "3.9.16"
EXPECTED_PACKAGES = {
    "torch": "1.10.0",
    "numpy": "1.26.4",
    "scipy": "1.10.1",
    "matplotlib": "3.9.4",
    "fairseq": "0.12.2",
    "omegaconf": "2.0.6",
    "hydra-core": "1.0.7",
    "scikit-learn": "1.3.0",
    "tqdm": "4.66.2",
}
IMPORT_NAMES = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "fairseq": "fairseq",
    "omegaconf": "omegaconf",
    "hydra-core": "hydra",
    "scikit-learn": "sklearn",
    "tqdm": "tqdm",
}


def _version_matches(actual: str, expected: str) -> bool:
    return actual == expected or actual.startswith(expected + "+")


def run_preflight(weights_dir: str | Path | None = None, device: str = "cuda:0") -> dict[str, object]:
    rows: list[dict[str, object]] = []
    success = True
    actual_python = platform.python_version()
    python_ok = actual_python == EXPECTED_PYTHON
    rows.append({"check": "python", "status": "OK" if python_ok else "VERSION_MISMATCH", "actual": actual_python, "expected": EXPECTED_PYTHON})
    success = success and python_ok

    patch_omegaconf_get_ref_type()
    for distribution, expected in EXPECTED_PACKAGES.items():
        try:
            actual = metadata.version(distribution)
            importlib.import_module(IMPORT_NAMES[distribution])
            ok = _version_matches(actual, expected)
            status = "OK" if ok else "VERSION_MISMATCH"
        except Exception as exc:
            actual = f"{type(exc).__name__}: {exc}"
            ok = False
            status = "IMPORT_ERROR"
        rows.append({"check": f"package:{distribution}", "status": status, "actual": actual, "expected": expected})
        success = success and ok

    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        index = int(device.split(":", 1)[1]) if device.startswith("cuda:") else -1
        device_ok = cuda_ok and 0 <= index < torch.cuda.device_count()
        actual = {
            "torch_cuda": torch.version.cuda,
            "available": cuda_ok,
            "device_count": torch.cuda.device_count(),
            "selected": device,
            "name": torch.cuda.get_device_name(index) if device_ok else None,
        }
        rows.append({"check": "cuda", "status": "OK" if device_ok else "UNAVAILABLE", "actual": actual, "expected": "CUDA 11.3-compatible NVIDIA GPU"})
        success = success and device_ok
    except Exception as exc:
        rows.append({"check": "cuda", "status": "IMPORT_ERROR", "actual": str(exc), "expected": "CUDA 11.3-compatible NVIDIA GPU"})
        success = False

    dictionary = resources.files("rnacompass._vendor.ernie_project.src").joinpath("dict").joinpath("dict.txt")
    resource_ok = dictionary.is_file()
    rows.append({"check": "resource:dict.txt", "status": "OK" if resource_ok else "MISSING", "actual": str(dictionary), "expected": "packaged"})
    success = success and resource_ok

    weight_report = verify_weights(resolve_weights_dir(weights_dir), check_hashes=True)
    for row in weight_report["weights"]:
        rows.append({"check": f"weight:{row['name']}", "status": row["status"], "actual": {"path": row["path"], "size": row["actual_size"], "sha256": row["actual_sha256"]}, "expected": {"size": row["expected_size"], "sha256": row["expected_sha256"]}})
    success = success and bool(weight_report["success"])
    return {
        "success": success,
        "platform": sys.platform,
        "weights_dir": weight_report["weights_dir"],
        "checks": rows,
    }
