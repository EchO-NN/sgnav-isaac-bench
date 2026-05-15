from __future__ import annotations

import sys
import os

from isaac_bench.scripts.preprocess_interioragent import _reexec_pythonpath


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
