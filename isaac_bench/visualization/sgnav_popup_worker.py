from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def _decode_image(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload.encode("ascii"))
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _write_ready() -> None:
    print(json.dumps({"type": "ready"}), flush=True)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-name", default="SG-Nav Isaac Debug")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args(argv)

    try:
        import cv2

        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, int(args.width), int(args.height))
    except Exception as exc:
        print("[sgnav-viz-worker] popup unavailable: %s" % exc, file=sys.stderr, flush=True)
        return 1

    _write_ready()
    for line in sys.stdin:
        try:
            req = json.loads(line)
            req_type = req.get("type")
            if req_type == "close":
                break
            if req_type == "frame_path":
                frame_path = Path(str(req["path"]))
                rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
            elif req_type == "frame":
                rgb = _decode_image(str(req["image_png_b64"]))
            else:
                continue
            cv2.imshow(args.window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
        except Exception as exc:
            print("[sgnav-viz-worker] frame update failed: %s" % exc, file=sys.stderr, flush=True)
            break
    try:
        cv2.destroyWindow(args.window_name)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
