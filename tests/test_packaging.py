import subprocess
import sys
from importlib import resources
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_packaged_dictionary_exists():
    dictionary = resources.files("rnacompass._vendor.ernie_project.src").joinpath("dict").joinpath("dict.txt")
    assert dictionary.is_file()
    assert dictionary.read_text(encoding="utf-8").startswith("G ")


def test_source_tree_and_release_audits_pass():
    checker = ROOT / "scripts" / "check_release_tree.py"
    source = subprocess.run([sys.executable, str(checker)], cwd=ROOT, text=True, capture_output=True)
    assert source.returncode == 0, source.stdout + source.stderr
    release = subprocess.run([sys.executable, str(checker), "--release"], cwd=ROOT, text=True, capture_output=True)
    assert release.returncode == 0, release.stdout + release.stderr
    assert "Release tree audit passed." in release.stdout
