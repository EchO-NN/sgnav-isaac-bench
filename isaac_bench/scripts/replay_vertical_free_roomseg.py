from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Mapping

import numpy as np

from isaac_bench.config import get_nested, load_config
from isaac_bench.mapping.vertical_free_roomseg import (
    VerticalFreeRoomSegConfig,
    run_vertical_free_roomseg,
    save_vertical_free_roomseg_debug,
)


def _first_existing(data: Mapping[str, np.ndarray], names: list[str]) -> np.ndarray:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError("missing any of required fields: %s" % ", ".join(names))


def _infer_step(path: Path) -> int:
    match = re.search(r"(?:step_|_step_)(\d+)", path.name)
    return int(match.group(1)) if match else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay vertical-free geodesic room segmentation from saved masks.")
    parser.add_argument("--input", required=True, help="NPZ produced by ROSE2/source, roomseg, or vertical-free debug.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="isaac_bench/configs/isaac_bench.yaml")
    parser.add_argument("--resolution-m", type=float, default=None)
    parser.add_argument("--step", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    roomseg_cfg = dict(get_nested(cfg, "mapping.room_segmentation", {}) or {})
    vf_cfg = VerticalFreeRoomSegConfig.from_mapping(
        dict(roomseg_cfg.get("vertical_free_roomseg", {}) or {}),
        resolution_m=float(args.resolution_m or get_nested(cfg, "mapping.online_resolution_m", 0.05)),
        min_room_area_m2=float(roomseg_cfg.get("min_room_area_m2", 1.2)),
        debug_dump=False,
    )

    input_path = Path(args.input).expanduser()
    with np.load(input_path, allow_pickle=False) as npz:
        data = {name: np.asarray(npz[name]) for name in npz.files}
    observed_free = _first_existing(
        data,
        ["vertical_free_room_domain", "repaired_roomseg_free", "observed_free", "input_free", "clean_free"],
    ).astype(bool)
    observed_occupied = _first_existing(
        data,
        ["repaired_roomseg_occupied", "observed_occupied", "input_wall", "wall", "clean_wall"],
    ).astype(bool)
    unknown = _first_existing(
        data,
        ["repaired_roomseg_unknown", "unknown", "input_unknown"],
    ).astype(bool)

    result = run_vertical_free_roomseg(
        observed_free=observed_free,
        observed_occupied=observed_occupied,
        unknown=unknown,
        resolution_m=float(vf_cfg.resolution_m),
        config=vf_cfg,
    )
    dump = save_vertical_free_roomseg_debug(
        result=result,
        out_dir=Path(args.out_dir),
        step=int(args.step if args.step is not None else _infer_step(input_path)),
    )
    summary = dict(dump.get("summary", {}))
    summary["input"] = str(input_path)
    summary["out_dir"] = str(Path(args.out_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
