import sys

import numpy as np

from isaac_bench.perception.detection_types import Detection2D
from isaac_bench.perception.detector_ipc import (
    SubprocessDetector,
    decode_rgb_png_b64,
    detection_from_payload,
    detection_to_payload,
    encode_rgb_png_b64,
)


def test_detection_payload_roundtrip():
    det = Detection2D("table", "Table", 0.7, (1.0, 2.0, 3.0, 4.0), class_id=5)
    out = detection_from_payload(detection_to_payload(det))
    assert out.category == "table"
    assert out.raw_label == "Table"
    assert out.confidence == 0.7
    assert out.bbox_xyxy == (1.0, 2.0, 3.0, 4.0)
    assert out.class_id == 5


def test_rgb_png_roundtrip():
    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    rgb[2, 3] = [10, 20, 30]
    out = decode_rgb_png_b64(encode_rgb_png_b64(rgb))
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, rgb)


def test_subprocess_detector_dry_run():
    detector = SubprocessDetector(
        "dry_run",
        "unused",
        python_executable=sys.executable,
        startup_timeout_s=20,
        response_timeout_s=20,
    )
    try:
        detector.set_vocabulary(["table", "chair"])
        assert detector.detect(np.zeros((8, 8, 3), dtype=np.uint8)) == []
    finally:
        detector.close()
