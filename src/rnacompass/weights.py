"""Manual, verified installation of the external RNAcompass weight archive."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from .model import BASE_WEIGHT_SHA256, DISTILLED_WEIGHT_SHA256, MODEL_ID


ARCHIVE_ROOT = "rnacompass-weights-v1"
MANIFEST_NAME = "manifest.json"
ENV_WEIGHTS_DIR = "RNACOMPASS_WEIGHTS_DIR"


@dataclass(frozen=True)
class WeightSpec:
    name: str
    relative_path: str
    size: int
    sha256: str


WEIGHT_SPECS = (
    WeightSpec(
        name="RNAcompass-trained ERNIE-RNA-giga base checkpoint",
        relative_path="base/checkpoint_12_960000.pt",
        size=3_653_738_558,
        sha256=BASE_WEIGHT_SHA256,
    ),
    WeightSpec(
        name="RNAcompass Paper-R2 epoch-2 distilled checkpoint",
        relative_path="distilled/vernie_rna_giga_irf_epoch002.pth",
        size=1_218_010_975,
        sha256=DISTILLED_WEIGHT_SHA256,
    ),
)

LICENSE_FILES = (
    "LICENSE",
)


def default_weights_dir() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return (root / "rnacompass" / "weights" / "v1").resolve()


def resolve_weights_dir(value: str | Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    configured = os.environ.get(ENV_WEIGHTS_DIR)
    if configured:
        return Path(configured).expanduser().resolve()
    return default_weights_dir()


def weight_path(root: Path, spec: WeightSpec) -> Path:
    return root / spec.relative_path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(
    root: str | Path | None = None,
    *,
    check_hashes: bool = True,
    specs: Iterable[WeightSpec] = WEIGHT_SPECS,
) -> dict[str, object]:
    destination = resolve_weights_dir(root)
    rows: list[dict[str, object]] = []
    success = True
    for spec in specs:
        path = weight_path(destination, spec)
        row: dict[str, object] = {
            "name": spec.name,
            "path": str(path),
            "expected_size": spec.size,
            "expected_sha256": spec.sha256,
        }
        if not path.is_file():
            row.update(status="MISSING", actual_size=None, actual_sha256=None)
            success = False
        else:
            actual_size = path.stat().st_size
            row["actual_size"] = actual_size
            if actual_size != spec.size:
                row.update(status="SIZE_MISMATCH", actual_sha256=None)
                success = False
            elif check_hashes:
                actual_hash = sha256_file(path)
                row["actual_sha256"] = actual_hash
                row["status"] = "OK" if actual_hash == spec.sha256 else "HASH_MISMATCH"
                success = success and actual_hash == spec.sha256
            else:
                row.update(status="PRESENT", actual_sha256=None)
        rows.append(row)
    return {"success": success, "weights_dir": str(destination), "weights": rows}


def _normalized_archive_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    parts = list(path.parts)
    if parts and parts[0] == ARCHIVE_ROOT:
        parts = parts[1:]
    if not parts:
        return ""
    return PurePosixPath(*parts).as_posix()


def _validate_archive_members(archive: tarfile.TarFile, specs: Iterable[WeightSpec]) -> dict[str, tarfile.TarInfo]:
    required = {MANIFEST_NAME, *(spec.relative_path for spec in specs), *LICENSE_FILES}
    files: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        relative = _normalized_archive_name(member.name)
        if not relative:
            continue
        if member.isdir():
            continue
        if not member.isfile() or member.issym() or member.islnk():
            raise ValueError(f"archive member must be a regular file: {member.name}")
        if relative not in required:
            raise ValueError(f"unexpected archive member: {member.name}")
        if relative in files:
            raise ValueError(f"duplicate archive member: {member.name}")
        files[relative] = member
    missing = sorted(required - set(files))
    if missing:
        raise ValueError(f"archive is missing required files: {', '.join(missing)}")
    return files


def _validate_embedded_manifest(path: Path, specs: Iterable[WeightSpec]) -> None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid archive manifest: {exc}") from exc
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError(f"archive manifest model_id must be {MODEL_ID!r}")
    expected = {
        spec.relative_path: {"size": spec.size, "sha256": spec.sha256}
        for spec in specs
    }
    observed = {
        item.get("path"): {"size": item.get("size"), "sha256": item.get("sha256")}
        for item in manifest.get("weights", [])
        if isinstance(item, dict)
    }
    if observed != expected:
        raise ValueError("archive manifest does not match the published RNAcompass v1 weight specification")


def install_archive(
    archive_path: str | Path,
    destination: str | Path | None = None,
    *,
    specs: Iterable[WeightSpec] = WEIGHT_SPECS,
) -> Path:
    source = Path(archive_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"weight archive not found: {source}")
    target = resolve_weights_dir(destination)
    if target.exists():
        report = verify_weights(target, check_hashes=True, specs=specs)
        if report["success"]:
            return target
        raise FileExistsError(f"weight destination already exists but is incomplete or invalid: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".rnacompass-weights-", dir=target.parent))
    try:
        with tarfile.open(source, mode="r:*") as archive:
            members = _validate_archive_members(archive, specs)
            for relative, member in members.items():
                output = staging / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read archive member: {member.name}")
                with extracted, output.open("wb") as handle:
                    shutil.copyfileobj(extracted, handle)
        _validate_embedded_manifest(staging / MANIFEST_NAME, specs)
        report = verify_weights(staging, check_hashes=True, specs=specs)
        if not report["success"]:
            bad = [row["name"] for row in report["weights"] if row["status"] != "OK"]
            raise ValueError(f"weight verification failed: {', '.join(str(item) for item in bad)}")
        staging.replace(target)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def expected_archive_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "weights": [
            {"path": spec.relative_path, "size": spec.size, "sha256": spec.sha256}
            for spec in WEIGHT_SPECS
        ],
    }
