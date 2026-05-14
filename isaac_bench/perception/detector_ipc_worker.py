from __future__ import annotations

import argparse
import contextlib
import json
import sys
from typing import Any, Dict, List

from isaac_bench.perception.detector_ipc import decode_rgb_image_path, decode_rgb_png_b64, detection_to_payload
from isaac_bench.perception.sam2_segmenter import SegmentingDetector, build_sam2_segmenter
from isaac_bench.perception.yolo_world_detector import build_detector


def _write(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _write_error(message: str) -> None:
    _write({"type": "error", "message": message})


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", default="yolo_world", choices=["dry_run", "yolo_world", "none"])
    parser.add_argument("--model", default="data/models/yolov8l-worldv2.pt")
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--categories-json", default="[]")
    parser.add_argument("--segmenter", default="none", choices=["none", "auto", "sam2"])
    parser.add_argument("--sam2-checkpoint", default="")
    parser.add_argument("--sam2-model-cfg", default="")
    parser.add_argument("--sam2-device", default="cuda")
    args = parser.parse_args(argv)

    try:
        categories = json.loads(args.categories_json)
        if not isinstance(categories, list):
            categories = []
        with contextlib.redirect_stdout(sys.stderr):
            detector = build_detector(args.detector, args.model, conf=args.conf, iou=args.iou)
            segmenter = build_sam2_segmenter(
                args.segmenter,
                checkpoint=args.sam2_checkpoint,
                model_cfg=args.sam2_model_cfg,
                device=args.sam2_device,
                required=args.segmenter == "sam2",
            )
            if segmenter is not None:
                detector = SegmentingDetector(detector, segmenter)
            if detector is not None:
                detector.set_vocabulary([str(cat) for cat in categories])
    except Exception as exc:
        _write_error(str(exc))
        return 1

    _write({"type": "ready"})
    for line in sys.stdin:
        try:
            req = json.loads(line)
            req_type = req.get("type")
            if req_type == "close":
                break
            if req_type == "set_vocabulary":
                categories = req.get("categories", [])
                if not isinstance(categories, list):
                    categories = []
                with contextlib.redirect_stdout(sys.stderr):
                    detector.set_vocabulary([str(cat) for cat in categories])
                _write({"type": "ok"})
                continue
            if req_type == "detect":
                rgb = decode_rgb_png_b64(str(req["image_png_b64"]))
                with contextlib.redirect_stdout(sys.stderr):
                    detections = detector.detect(rgb)
                _write({"type": "detections", "detections": [detection_to_payload(det) for det in detections]})
                continue
            if req_type == "detect_path":
                rgb = decode_rgb_image_path(str(req["path"]))
                with contextlib.redirect_stdout(sys.stderr):
                    detections = detector.detect(rgb)
                _write({"type": "detections", "detections": [detection_to_payload(det) for det in detections]})
                continue
            raise ValueError("unknown detector IPC request type: %s" % req_type)
        except Exception as exc:
            _write_error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
