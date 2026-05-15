from __future__ import annotations

import pytest

from isaac_bench.perception import sam2_segmenter


def test_sam2_required_mode_raises_when_loader_fails(monkeypatch, tmp_path):
    def fail_loader(*args, **kwargs):
        raise RuntimeError("missing sam2")

    monkeypatch.setattr(sam2_segmenter, "SAM2BoxSegmenter", fail_loader)

    with pytest.raises(RuntimeError, match="missing sam2"):
        sam2_segmenter.build_sam2_segmenter(
            "sam2",
            checkpoint=str(tmp_path / "missing.pt"),
            model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
            required=True,
        )


def test_sam2_auto_mode_returns_none_when_loader_fails(monkeypatch, tmp_path, capsys):
    def fail_loader(*args, **kwargs):
        raise RuntimeError("missing sam2")

    monkeypatch.setattr(sam2_segmenter, "SAM2BoxSegmenter", fail_loader)

    segmenter = sam2_segmenter.build_sam2_segmenter(
        "auto",
        checkpoint=str(tmp_path / "missing.pt"),
        model_cfg="configs/sam2.1/sam2.1_hiera_s.yaml",
        required=False,
    )
    captured = capsys.readouterr()

    assert segmenter is None
    assert "continuing without masks" in captured.out
