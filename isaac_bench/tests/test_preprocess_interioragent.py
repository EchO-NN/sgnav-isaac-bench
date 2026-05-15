from __future__ import annotations

import os
import sys

from isaac_bench.scripts.preprocess_interioragent import _reexec_pythonpath, main as preprocess_main


def test_reexec_pythonpath_preserves_active_environment_paths(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    site_packages = tmp_path / "env" / "lib" / "python3.11" / "site-packages"
    site_packages.mkdir(parents=True)
    usd_path = tmp_path / "omni.usd.libs"
    usd_path.mkdir()
    existing_path = tmp_path / "existing"
    existing_path.mkdir()
    monkeypatch.setattr(sys, "path", ["", str(site_packages), str(site_packages), str(tmp_path / "missing")])

    pythonpath = _reexec_pythonpath(repo_root, str(usd_path), str(existing_path))
    parts = pythonpath.split(os.pathsep)

    assert parts[:2] == [str(repo_root), str(usd_path)]
    assert str(site_packages.resolve()) in parts
    assert str(existing_path) in parts
    assert parts.count(str(site_packages.resolve())) == 1


def test_preprocess_cli_fails_clearly_for_missing_dataset_root(tmp_path, capsys):
    status = preprocess_main(["--dataset-root", str(tmp_path / "missing"), "--out", str(tmp_path / "out")])
    captured = capsys.readouterr()

    assert status == 2
    assert "InteriorAgent dataset root not found" in captured.err
