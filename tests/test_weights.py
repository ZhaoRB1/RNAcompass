from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from rnacompass.model import MODEL_ID
from rnacompass.weights import (
    ARCHIVE_ROOT,
    ENV_WEIGHTS_DIR,
    LICENSE_FILES,
    WeightSpec,
    install_archive,
    resolve_weights_dir,
    verify_weights,
)


def _spec(payload: bytes) -> WeightSpec:
    return WeightSpec("tiny test weight", "base/test.pt", len(payload), hashlib.sha256(payload).hexdigest())


def _archive(path: Path, payload: bytes, spec: WeightSpec, extra_name: str | None = None) -> None:
    manifest = json.dumps({
        "schema_version": 1,
        "model_id": MODEL_ID,
        "weights": [{"path": spec.relative_path, "size": spec.size, "sha256": spec.sha256}],
    }).encode()
    entries = {
        "manifest.json": manifest,
        spec.relative_path: payload,
        LICENSE_FILES[0]: b"upstream license",
        LICENSE_FILES[1]: b"model license",
    }
    if extra_name:
        entries[extra_name] = b"unexpected"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in entries.items():
            info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_weight_install_is_verified_and_atomic(tmp_path: Path):
    payload = b"test-checkpoint"
    spec = _spec(payload)
    archive = tmp_path / "weights.tar.gz"
    destination = tmp_path / "installed"
    _archive(archive, payload, spec)
    installed = install_archive(archive, destination, specs=(spec,))
    assert installed == destination.resolve()
    assert (installed / spec.relative_path).read_bytes() == payload
    assert verify_weights(installed, specs=(spec,))["success"] is True


def test_archive_rejects_unknown_or_unsafe_members(tmp_path: Path):
    payload = b"test-checkpoint"
    spec = _spec(payload)
    bad = tmp_path / "bad.tar.gz"
    _archive(bad, payload, spec, extra_name="unknown.txt")
    with pytest.raises(ValueError, match="unexpected"):
        install_archive(bad, tmp_path / "bad-dest", specs=(spec,))

    traversal = tmp_path / "traversal.tar.gz"
    with tarfile.open(traversal, "w:gz") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        install_archive(traversal, tmp_path / "traversal-dest", specs=(spec,))


def test_archive_rejects_checkpoint_hash_mismatch(tmp_path: Path):
    declared = _spec(b"expected")
    archive = tmp_path / "corrupt.tar.gz"
    _archive(archive, b"corrupt!", declared)
    with pytest.raises(ValueError, match="verification failed"):
        install_archive(archive, tmp_path / "corrupt-dest", specs=(declared,))
    assert not (tmp_path / "corrupt-dest").exists()


def test_weight_path_precedence(monkeypatch, tmp_path: Path):
    configured = tmp_path / "configured"
    explicit = tmp_path / "explicit"
    monkeypatch.setenv(ENV_WEIGHTS_DIR, str(configured))
    assert resolve_weights_dir() == configured.resolve()
    assert resolve_weights_dir(explicit) == explicit.resolve()
