from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy, world_xy_to_grid
from isaac_bench.mapping.room_map_from_rooms_json import RoomGT, point_in_polygon

PASSABLE_CATEGORIES = {
    "floor",
    "ceiling",
    "roof",
    "light_switch",
    "outlet",
    "door",
    "doorsill",
    "door_handle",
}
PASSABLE_OPENING_CATEGORIES = {"door", "doorsill"}


def infer_map_info(objects: Sequence[Mapping], rooms: Sequence[RoomGT], resolution_m: float, padding_m: float = 1.0) -> MapInfo:
    xs: List[float] = []
    ys: List[float] = []
    for room in rooms:
        for x, y in room.polygon_xy:
            xs.append(float(x))
            ys.append(float(y))
    for obj in objects:
        mn = obj.get("bbox_min_world", [0, 0, 0])
        mx = obj.get("bbox_max_world", [0, 0, 0])
        xs.extend([float(mn[0]), float(mx[0])])
        ys.extend([float(mn[1]), float(mx[1])])
    if not xs or not ys:
        raise ValueError("Cannot infer map bounds without rooms or objects")
    min_x = math.floor((min(xs) - padding_m) / resolution_m) * resolution_m
    max_x = math.ceil((max(xs) + padding_m) / resolution_m) * resolution_m
    min_y = math.floor((min(ys) - padding_m) / resolution_m) * resolution_m
    max_y = math.ceil((max(ys) + padding_m) / resolution_m) * resolution_m
    width = int(math.ceil((max_x - min_x) / resolution_m))
    height = int(math.ceil((max_y - min_y) / resolution_m))
    return MapInfo(resolution_m, min_x, max_x, min_y, max_y, width, height)


def rasterize_bbox_occupancy(objects: Sequence[Mapping], map_info: MapInfo, min_z: float = 0.05, max_z: float = 1.50) -> np.ndarray:
    occ = np.zeros((map_info.height, map_info.width), dtype=np.uint8)
    for obj in objects:
        category = str(obj.get("category", "unknown"))
        if category in PASSABLE_CATEGORIES:
            continue
        mn = obj.get("bbox_min_world", None)
        mx = obj.get("bbox_max_world", None)
        if mn is None or mx is None:
            continue
        if float(mx[2]) < min_z or float(mn[2]) > max_z:
            continue
        r1, c1 = world_xy_to_grid(float(mn[0]), float(mn[1]), map_info)
        r2, c2 = world_xy_to_grid(float(mx[0]), float(mx[1]), map_info)
        r0, r3 = sorted((r1, r2))
        c0, c3 = sorted((c1, c2))
        r0 = max(0, min(map_info.height - 1, r0))
        r3 = max(0, min(map_info.height - 1, r3))
        c0 = max(0, min(map_info.width - 1, c0))
        c3 = max(0, min(map_info.width - 1, c3))
        if r0 <= r3 and c0 <= c3:
            occ[r0 : r3 + 1, c0 : c3 + 1] = 1
    return occ


def carve_passable_openings(
    occupancy: np.ndarray,
    objects: Sequence[Mapping],
    map_info: MapInfo,
    min_z: float = 0.05,
    max_z: float = 1.50,
) -> np.ndarray:
    out = np.array(occupancy, copy=True)
    for obj in objects:
        category = str(obj.get("category", "unknown"))
        if category not in PASSABLE_OPENING_CATEGORIES:
            continue
        mn = obj.get("bbox_min_world", None)
        mx = obj.get("bbox_max_world", None)
        if mn is None or mx is None:
            continue
        if float(mx[2]) < min_z or float(mn[2]) > max_z:
            continue
        r1, c1 = world_xy_to_grid(float(mn[0]), float(mn[1]), map_info)
        r2, c2 = world_xy_to_grid(float(mx[0]), float(mx[1]), map_info)
        r0, r3 = sorted((r1, r2))
        c0, c3 = sorted((c1, c2))
        r0 = max(0, min(map_info.height - 1, r0))
        r3 = max(0, min(map_info.height - 1, r3))
        c0 = max(0, min(map_info.width - 1, c0))
        c3 = max(0, min(map_info.width - 1, c3))
        if r0 <= r3 and c0 <= c3:
            out[r0 : r3 + 1, c0 : c3 + 1] = 0
    return out


def rasterize_room_mask(rooms: Sequence[RoomGT], map_info: MapInfo) -> np.ndarray:
    mask = np.zeros((map_info.height, map_info.width), dtype=np.uint8)
    if not rooms:
        mask[:, :] = 1
        return mask
    for r in range(map_info.height):
        for c in range(map_info.width):
            x, y = grid_to_world_xy(r, c, map_info)
            for room in rooms:
                if point_in_polygon(x, y, room.polygon_xy):
                    mask[r, c] = 1
                    break
    return mask


def disk_offsets(radius_cells: int) -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    rr = int(max(0, radius_cells))
    for dr in range(-rr, rr + 1):
        for dc in range(-rr, rr + 1):
            if dr * dr + dc * dc <= rr * rr:
                offsets.append((dr, dc))
    return offsets


def dilate_binary(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return mask.astype(bool)
    src = mask.astype(bool)
    out = np.array(src, copy=True)
    ys, xs = np.nonzero(src)
    offsets = disk_offsets(radius_cells)
    h, w = src.shape
    for r, c in zip(ys, xs):
        for dr, dc in offsets:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out


def build_navigable(
    occupancy: np.ndarray,
    resolution_m: float,
    robot_radius_m: float,
    inflation_radius_m: float,
    room_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    radius_cells = int(math.ceil((float(robot_radius_m) + float(inflation_radius_m)) / float(resolution_m)))
    inflated = dilate_binary(occupancy > 0, radius_cells)
    navigable = ~inflated
    if room_mask is not None:
        navigable &= room_mask.astype(bool)
    return navigable.astype(np.uint8), inflated.astype(np.uint8)


def save_debug_topdown(path: str, occupancy: np.ndarray, navigable: np.ndarray) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    rgb = np.zeros((occupancy.shape[0], occupancy.shape[1], 3), dtype=np.uint8)
    rgb[navigable.astype(bool)] = (245, 245, 245)
    rgb[occupancy.astype(bool)] = (20, 20, 20)
    Image.fromarray(rgb).save(path)


def save_nav2_map(scene_out: Path, occupancy: np.ndarray, map_info: MapInfo, navigable: np.ndarray | None = None) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    # ROS map origin is bottom-left; flip rows when exporting the image.
    img = np.full(occupancy.shape, 254, dtype=np.uint8)
    img[occupancy.astype(bool)] = 0
    if navigable is not None:
        img[~navigable.astype(bool)] = 0
    img = np.flipud(img)
    Image.fromarray(img).save(scene_out / "nav2_map.png")
    yaml_text = "\n".join(
        [
            "image: nav2_map.png",
            "mode: trinary",
            "resolution: %.8f" % map_info.resolution_m,
            "origin: [%.8f, %.8f, 0.0]" % (map_info.min_x, map_info.min_y),
            "negate: 0",
            "occupied_thresh: 0.65",
            "free_thresh: 0.25",
            "",
        ]
    )
    (scene_out / "nav2_map.yaml").write_text(yaml_text, encoding="utf-8")


def save_preprocessed_scene(
    scene_out: str,
    objects_all: Sequence[Mapping],
    rooms: Sequence[RoomGT],
    resolution_m: float,
    robot_radius_m: float,
    inflation_radius_m: float,
    obstacle_min_height_m: float = 0.05,
    obstacle_max_height_m: float = 1.50,
) -> MapInfo:
    out = Path(scene_out)
    out.mkdir(parents=True, exist_ok=True)
    map_info = infer_map_info(objects_all, rooms, resolution_m)
    occupancy = rasterize_bbox_occupancy(objects_all, map_info, obstacle_min_height_m, obstacle_max_height_m)
    occupancy = carve_passable_openings(occupancy, objects_all, map_info, obstacle_min_height_m, obstacle_max_height_m)
    room_mask = rasterize_room_mask(rooms, map_info)
    navigable, inflated = build_navigable(occupancy, resolution_m, robot_radius_m, inflation_radius_m, room_mask=room_mask)
    np.save(out / "occupancy.npy", occupancy.astype(np.uint8))
    np.save(out / "room_mask.npy", room_mask.astype(np.uint8))
    np.save(out / "navigable.npy", navigable.astype(np.uint8))
    np.save(out / "inflated_obstacles.npy", inflated.astype(np.uint8))
    with open(out / "map_info.json", "w", encoding="utf-8") as handle:
        json.dump(map_info.to_dict(), handle, indent=2)
    save_debug_topdown(str(out / "debug_topdown.png"), occupancy, navigable)
    save_nav2_map(out, occupancy, map_info, navigable)
    return map_info
