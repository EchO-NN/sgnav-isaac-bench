from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isaac_bench.mapping.online_roomseg import (
    ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND,
    OnlineRoseStyleConfig,
    run_online_rose_style_roomseg,
)
from isaac_bench.mapping.online_watershed_roomseg import (
    ONLINE_WATERSHED_ROOMSEG_BACKEND,
    OnlineWatershedRoomSegConfig,
    run_online_watershed_roomseg,
)
from isaac_bench.mapping.vertical_profile import VerticalProfileMap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay one saved roomseg NPZ through online room segmentation without Isaac.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--backend", default=ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND, choices=[ONLINE_LINE_EXTEND_ROOMSEG_V2_BACKEND, "online_rose_style_v1", ONLINE_WATERSHED_ROOMSEG_BACKEND])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resolution-m", type=float, default=0.05)
    parser.add_argument("--save-layers", action="store_true")
    args = parser.parse_args(argv)

    src = Path(args.input)
    data = np.load(src, allow_pickle=False)
    free = _mask(data, "vertical_free_raw", "vertical_free_room_domain", "initial_roomseg_free_after_ray_wall", "observed_free_mask")
    occupied = _mask(data, "vertical_occupied_raw", "vertical_occupied_0p2_2p0", "initial_roomseg_occupied_after_ray_wall", "obstacle_mask")
    observed = _mask(data, "vertical_observed_raw", "vertical_observed_0p2_2p0", "vertical_observed_map", default=(free | occupied))
    unknown = _mask(data, "vertical_unknown_raw", "initial_roomseg_unknown_after_ray_wall", "unknown_mask", default=~observed)
    shape = free.shape
    vp = _vertical_profile_from_masks(free, occupied, observed)
    out_dir = Path(args.out_dir)
    if str(args.backend) == ONLINE_WATERSHED_ROOMSEG_BACKEND:
        cfg = OnlineWatershedRoomSegConfig.from_mapping(
            {
                "online_watershed_roomseg": {
                    "debug": {"save_layers": bool(args.save_layers), "save_json": True},
                    "debug_dir": str(out_dir),
                }
            },
            resolution_m=float(args.resolution_m),
        )
        result = run_online_watershed_roomseg(
            occupancy_map=occupied.astype(bool),
            observed_free_mask=free.astype(bool),
            obstacle_mask=occupied.astype(bool),
            unknown_mask=unknown.astype(bool),
            vertical_profile=vp,
            config=cfg,
            step=0,
        )
        report = dict(result.debug.get("watershed_region_report", {}))
        accepted_count = 0
        rejected_count = 0
        report_name = "watershed_region_report.json"
    else:
        cfg = OnlineRoseStyleConfig.from_mapping(
            {
                "online_roomseg": {
                    "debug": {"save_layers": bool(args.save_layers), "save_candidate_json": True},
                    "debug_dir": str(out_dir),
                }
            },
            resolution_m=float(args.resolution_m),
        )
        result = run_online_rose_style_roomseg(
            occupancy_map=occupied.astype(bool),
            observed_free_mask=free.astype(bool),
            obstacle_mask=occupied.astype(bool),
            unknown_mask=unknown.astype(bool),
            vertical_profile=vp,
            config=cfg,
            step=0,
        )
        report = dict(result.debug.get("separator_report", {}))
        accepted_count = int(report.get("accepted_count", 0))
        rejected_count = int(report.get("rejected_count", 0))
        report_name = "separator_report.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_payload = {key: value for key, value in result.layers.items() if isinstance(value, np.ndarray)}
    layer_payload.pop("final_room_labels", None)
    layer_payload.pop("accepted_separators", None)
    np.savez_compressed(
        out_dir / "online_roomseg_result.npz",
        **layer_payload,
        final_room_labels=result.room_label_map.astype(np.int32),
        accepted_separators=getattr(result, "separator_map", np.zeros(shape, dtype=bool)).astype(np.uint8),
    )
    (out_dir / report_name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "input": str(src),
        "backend": str(args.backend),
        "room_count": int(result.debug.get("room_count", 0)),
        "accepted_count": int(accepted_count),
        "rejected_count": int(rejected_count),
        "region_type_counts": dict(report.get("region_type_counts", {})),
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _mask(data, *names: str, default=None) -> np.ndarray:
    for name in names:
        if name in data.files:
            return np.asarray(data[name], dtype=bool)
    if default is not None:
        return np.asarray(default, dtype=bool)
    raise KeyError("none of the requested masks were present: %s" % ", ".join(names))


def _vertical_profile_from_masks(free: np.ndarray, occupied: np.ndarray, observed: np.ndarray) -> VerticalProfileMap:
    shape = tuple(free.shape)
    vp = VerticalProfileMap(
        band_names=("roomseg_0p2_2p0",),
        band_ranges_m=((0.2, 2.0),),
        occupied_count=np.zeros((1, *shape), dtype=np.uint16),
        free_ray_count=np.zeros((1, *shape), dtype=np.uint16),
        observed_count=np.zeros((1, *shape), dtype=np.uint16),
        unknown_count=np.zeros((1, *shape), dtype=np.uint16),
    )
    vp.free_ray_count[0][free] = 1
    vp.occupied_count[0][occupied] = 1
    vp.observed_count[0][observed] = 1
    vp.unknown_count[0][~observed] = 1
    return vp


if __name__ == "__main__":
    raise SystemExit(main())
