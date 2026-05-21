from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from isaac_bench.config import get_nested, load_config, str_to_bool
from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.dataset.episode_generator import point_to_bbox_2d_distance, read_jsonl
from isaac_bench.env.habitat_like_env import MapSimHabitatLikeEnv
from isaac_bench.debug.graph_debug_dump import save_graph_debug_dump
from isaac_bench.debug.roomseg_layer_dump import ROOMSEG_SNAPSHOT_ARRAY_KEYS, save_roomseg_layer_dump
from isaac_bench.graph.decision import NavigationDecision, SGNavDecision
from isaac_bench.graph.room_context import (
    RoomContextCache,
    RoomContextResult,
    prepare_room_context_for_frontier_scoring,
    room_context_not_invoked_metadata,
)
from isaac_bench.graph.sgnav_scenegraph_adapter import SGNAV_ROOM_NAMES, SGNavSceneGraphAdapter
from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy, is_inside_grid, world_xy_to_grid
from isaac_bench.mapping.frontier import extract_frontiers, frontier_debug_layers
from isaac_bench.mapping.frontier_room_context import assign_frontier_room_context
from isaac_bench.mapping.frontier_debug import save_frontier_debug_snapshot
from isaac_bench.mapping.online_mapper import OnlineMapper
from isaac_bench.mapping.room_map_from_rooms_json import build_room_index_map, load_rooms
from isaac_bench.mapping.room_segmentation import (
    OnlineRoomSegmenter,
    RoomSegmentationConfig,
    room_segmentation_debug,
)
from isaac_bench.mapping.rose2_room_segmentation import OnlineROSE2RoomSegmenter
from isaac_bench.mapping.online_roomseg import (
    ONLINE_ROSE_STYLE_BACKEND,
    ONLINE_ROSE_STYLE_CONTEXT,
    OnlineRoseStyleConfig,
    OnlineRoseStyleRoomSegmenter,
)
from isaac_bench.mapping.online_watershed_roomseg import (
    ONLINE_WATERSHED_ROOMSEG_BACKEND,
    ONLINE_WATERSHED_ROOMSEG_CONTEXT,
    OnlineWatershedRoomSegConfig,
    OnlineWatershedRoomSegmenter,
)
from isaac_bench.mapping.upstream_rose2_pure_python_adapter import (
    UpstreamROSE2Config,
    UpstreamROSE2PurePythonSegmenter,
)
from isaac_bench.mapping.vertical_free_roomseg import (
    VERTICAL_FREE_ROOMSEG_ALGORITHM,
    VERTICAL_FREE_ROOMSEG_BACKEND,
    VERTICAL_FREE_ROOMSEG_CONTEXT,
)
from isaac_bench.mapping.vertical_free_gap_closure_roomseg import (
    VERTICAL_FREE_GAP_CLOSURE_ALGORITHM,
    VERTICAL_FREE_GAP_CLOSURE_BACKEND,
    VERTICAL_FREE_GAP_CLOSURE_CONTEXT,
)
from isaac_bench.graph.room_semantics import (
    DEFAULT_ROOM_CATEGORIES,
    VLMRoomLabeler,
)
from isaac_bench.metrics.episode_logger import JsonlEpisodeLogger, make_jsonable
from isaac_bench.metrics.evaluator import EpisodeEvaluator
from isaac_bench.metrics.result_schema import BenchmarkAssetError, complete_result_row, validate_strict_benchmark_assets
from isaac_bench.navigation.astar import GridAStarPlanner, astar_distance_map
from isaac_bench.navigation.frontier_commitment import FrontierCommitmentManager
from isaac_bench.navigation.waypoint_follower import HolonomicWaypointFollower
from isaac_bench.perception.detection_types import (
    MIN_VALID_DETECTION_CONFIDENCE,
    Detection2D,
    Detection3D,
    bbox_touches_image_edge,
    detection_confidence_is_valid,
)
from isaac_bench.perception.detector_ipc import SubprocessDetector
from isaac_bench.perception.fused_instance_registry import FusedInstanceRegistry
from isaac_bench.perception.object_memory import ObjectMemory
from isaac_bench.perception.sam2_segmenter import build_sam2_segmenter
from isaac_bench.perception.yolo_world_detector import build_detector
from isaac_bench.sensors.camera_geometry import CameraIntrinsics
from isaac_bench.sensors.depth_backproject import detections_to_3d
from isaac_bench.visualization.draw_map import save_map_png
from isaac_bench.visualization.sgnav_popup import SGNavPopupVisualizer


def load_preprocessed_for_episode(episode: dict):
    scene_dir = Path(episode["preprocessed_scene_dir"])
    with open(scene_dir / "map_info.json", "r", encoding="utf-8") as handle:
        map_info = MapInfo.from_dict(json.load(handle))
    occupancy = np.load(scene_dir / "occupancy.npy")
    navigable = np.load(scene_dir / "navigable.npy").astype(bool)
    return scene_dir, map_info, occupancy, navigable


def load_passable_opening_mask(scene_dir: Path, map_info: MapInfo) -> np.ndarray:
    objects_path = scene_dir / "objects_all.json"
    if objects_path.exists():
        from isaac_bench.dataset.occupancy_builder import rasterize_passable_openings

        with open(objects_path, "r", encoding="utf-8") as handle:
            objects_all = json.load(handle)
        return rasterize_passable_openings(objects_all, map_info).astype(bool)
    mask_path = scene_dir / "passable_openings.npy"
    if mask_path.exists():
        mask = np.load(mask_path).astype(bool)
        if mask.shape == (int(map_info.height), int(map_info.width)):
            return mask
    return np.zeros((int(map_info.height), int(map_info.width)), dtype=bool)


def _positive_int_or_none(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_roomseg_depth_stride_px(room_segmentation_config: Mapping[str, object] | None, default_stride_px: int) -> int:
    """Return the depth stride used to build vertical-profile roomseg evidence.

    The vertical-free room-segmentation map is generated during OnlineMapper's
    depth ray pass. If roomseg asks for denser sampling, that request has to
    feed the mapper pass instead of only the later segmentation stage.
    """

    default_stride = max(1, int(default_stride_px))
    strides = [default_stride]
    cfg = dict(room_segmentation_config or {})
    candidate_blocks: list[Mapping[str, object]] = []
    top_level_depth = cfg.get("depth")
    if isinstance(top_level_depth, Mapping):
        candidate_blocks.append(top_level_depth)
    for section in (
        "online_roomseg",
        "online_watershed_roomseg",
        "vertical_free_roomseg",
        "vertical_free_gap_closure",
    ):
        section_cfg = cfg.get(section)
        if not isinstance(section_cfg, Mapping):
            continue
        depth_cfg = section_cfg.get("depth")
        if isinstance(depth_cfg, Mapping):
            candidate_blocks.append(depth_cfg)
    for block in candidate_blocks:
        parsed = _positive_int_or_none(block.get("roomseg_depth_stride_px"))
        if parsed is not None:
            strides.append(parsed)
    return min(strides)


def _roomseg_debug_for_layer_dump(room_debug: Mapping[str, object], room_segmenter: object | None) -> dict:
    debug = dict(room_debug or {})
    result = getattr(room_segmenter, "last_result", None)
    layers = getattr(result, "layers", None)
    if isinstance(layers, Mapping):
        aliases = {
            "vertical_free_room_domain": "vertical_free_raw",
            "vertical_occupied_0p2_2p0": "vertical_occupied_raw",
            "vertical_observed_map": "vertical_observed_raw",
            "vertical_observed_0p2_2p0": "vertical_observed_raw",
            "vertical_unknown_before_overlay": "vertical_unknown_raw",
            "navigation_free_room_domain": "free_clean",
            "initial_roomseg_free": "free_clean",
            "initial_roomseg_occupied": "wall_candidate_clean",
            "initial_roomseg_unknown": "unknown_clean",
            "repaired_roomseg_free": "free_clean",
            "repaired_roomseg_occupied": "wall_candidate_clean",
            "repaired_roomseg_unknown": "unknown_clean",
            "pass2_extension_intersection_targets": "pass2_extension_intersection_targets",
            "pass2_line_extension_completion": "pass2_line_extension_completion",
            "wall_target_after_line_extension": "wall_target_after_line_extension",
            "completed_wall_after_line_extension": "completed_wall_after_line_extension",
            "structural_free_mask": "free_clean",
            "wall_boundary_map": "wall_candidate_clean",
            "candidate_wall": "wall_candidate_clean",
            "final_room_label_map": "room_labels_after_corridor_merge",
            "room_labels_after_merge": "room_labels_after_corridor_merge",
            "room_proposal_labels_before_merge": "room_labels_before_separators",
        }
        for out_key, layer_key in aliases.items():
            if out_key not in debug and layer_key in layers:
                debug[out_key] = np.asarray(layers[layer_key])
    if "final_room_label_map" not in debug and result is not None and hasattr(result, "room_label_map"):
        debug["final_room_label_map"] = np.asarray(getattr(result, "room_label_map"))
    return debug


def apply_episode_planning_clearance(
    scene_dir: Path,
    map_info: MapInfo,
    navigable: np.ndarray,
    episode: dict,
    runtime_planning_clearance_m: Optional[float] = None,
) -> np.ndarray:
    if runtime_planning_clearance_m is None:
        min_clearance = float(episode.get("metadata", {}).get("min_planning_clearance_m", 0.0))
    else:
        min_clearance = float(runtime_planning_clearance_m)
    if min_clearance <= 0.0:
        return navigable
    objects_all_path = scene_dir / "objects_all.json"
    if not objects_all_path.exists():
        return navigable
    from isaac_bench.dataset.episode_generator import build_clearance_mask, filter_start_clearance_objects

    with open(objects_all_path, "r", encoding="utf-8") as handle:
        clearance_objects = filter_start_clearance_objects(json.load(handle))
    return build_clearance_mask(navigable, map_info, clearance_objects, min_clearance)


def apply_dynamic_astar_edge_clearance(
    traversible: np.ndarray,
    occupied_walls: np.ndarray,
    resolution_m: float,
    extra_clearance_m: float,
    current_grid: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Keep A* an extra margin away from raw occupied wall/obstacle cells."""
    out = np.asarray(traversible, dtype=bool).copy()
    clearance = max(0.0, float(extra_clearance_m))
    resolution = max(1e-6, float(resolution_m))
    radius_cells = int(math.ceil(clearance / resolution))
    if radius_cells > 0:
        out &= ~_disk_dilate_bool(np.asarray(occupied_walls, dtype=bool), radius_cells)
    if current_grid is not None:
        rr, cc = int(current_grid[0]), int(current_grid[1])
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and bool(np.asarray(traversible, dtype=bool)[rr, cc]):
            out[rr, cc] = True
    return out


def _disk_dilate_bool(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    src = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_cells))
    if radius <= 0 or not np.any(src):
        return src.copy()
    try:
        from scipy import ndimage

        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        structure = (yy * yy + xx * xx) <= radius * radius
        return ndimage.binary_dilation(src, structure=structure).astype(bool)
    except Exception:
        out = np.array(src, copy=True)
        rows, cols = np.nonzero(src)
        h, w = src.shape
        offsets = [
            (dr, dc)
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if dr * dr + dc * dc <= radius * radius
        ]
        for row, col in zip(rows, cols):
            for dr, dc in offsets:
                rr, cc = int(row + dr), int(col + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
        return out


def apply_success_distance_override(episode: dict, args) -> dict:
    success_distance_m = getattr(args, "success_distance_m", None)
    if success_distance_m is None:
        return dict(episode)
    success_distance = float(success_distance_m)
    updated = dict(episode)
    scene_dir, map_info, _occupancy, navigable = load_preprocessed_for_episode(updated)
    navigable = apply_episode_planning_clearance(
        scene_dir,
        map_info,
        navigable,
        updated,
        runtime_planning_clearance_m=getattr(args, "runtime_planning_clearance_m", 0.0),
    )
    goal_objects = load_episode_goal_objects(scene_dir, updated)
    goal_cells = build_exact_goal_cells(goal_objects, navigable, map_info, success_distance)
    if goal_cells:
        start = (int(updated["start_grid"][0]), int(updated["start_grid"][1]))
        planner = GridAStarPlanner(navigable, map_info.resolution_m, allow_diagonal=True)
        shortest = planner.distance(start, goal_cells)
        if math.isfinite(shortest):
            updated["goal_regions_grid"] = [[int(r), int(c)] for r, c in goal_cells]
            updated["shortest_path_distance_m"] = float(shortest)
    updated["success_distance_m"] = success_distance
    metadata = dict(updated.get("metadata", {}))
    metadata["runtime_success_distance_override_m"] = success_distance
    updated["metadata"] = metadata
    return updated


def effective_perception_every_steps(detector_name: str | None, requested_steps: int | None) -> int:
    requested = max(1, int(requested_steps or 1))
    if str(detector_name or "").strip().lower() in {"yolo_world", "grounding_dino"}:
        return 1
    return requested


def load_episode_goal_objects(scene_dir: Path, episode: Mapping[str, object]) -> List[Mapping[str, object]]:
    objects_path = scene_dir / "objects.json"
    with open(objects_path, "r", encoding="utf-8") as handle:
        objects = json.load(handle)
    goal_ids = {str(value) for value in episode.get("goal_instance_ids", []) or []}
    if goal_ids:
        matched = [obj for obj in objects if str(obj.get("instance_id")) in goal_ids]
        if matched:
            return matched
    goal_category = normalize_category(str(episode.get("goal_category", "")))
    return [obj for obj in objects if normalize_category(str(obj.get("category", ""))) == goal_category]


def build_exact_goal_cells(
    goal_objects: Sequence[Mapping[str, object]],
    navigable: np.ndarray,
    map_info: MapInfo,
    success_distance_m: float,
) -> List[Tuple[int, int]]:
    cells = []
    fallback: List[Tuple[int, int]] = []
    best_distance = math.inf
    for r, c in np.argwhere(np.asarray(navigable).astype(bool)):
        rr, cc = int(r), int(c)
        wx, wy = grid_to_world_xy(rr, cc, map_info)
        distance = min_goal_bbox_distance(wx, wy, goal_objects)
        if not math.isfinite(distance):
            continue
        if distance <= float(success_distance_m):
            cells.append((rr, cc))
            continue
        if distance + 1e-6 < best_distance:
            best_distance = distance
            fallback = [(rr, cc)]
        elif abs(distance - best_distance) <= 1e-6:
            fallback.append((rr, cc))
    return cells if cells else fallback


def min_goal_bbox_distance(x: float, y: float, goal_objects: Sequence[Mapping[str, object]]) -> float:
    distances = []
    for obj in goal_objects:
        bbox_min = obj.get("bbox_min_world")
        bbox_max = obj.get("bbox_max_world")
        if bbox_min is None or bbox_max is None:
            continue
        distances.append(point_to_bbox_2d_distance(x, y, bbox_min, bbox_max))
    return min(distances) if distances else math.inf


def maybe_build_detector(
    detector_name: str,
    model_path: str,
    categories: List[str],
    conf: float = 0.7,
    iou: float = 0.5,
    allow_ipc_fallback: bool = False,
):
    try:
        detector = build_detector(
            detector_name,
            model_path,
            conf=conf,
            iou=iou,
            grounding_dino_config=getattr(maybe_build_detector, "grounding_dino_config", None),
            grounding_dino_text_threshold=float(getattr(maybe_build_detector, "grounding_dino_text_threshold", 0.25)),
            grounding_dino_device=str(getattr(maybe_build_detector, "grounding_dino_device", "cuda")),
        )
    except Exception as exc:
        if detector_name not in {"yolo_world", "grounding_dino"} or not allow_ipc_fallback:
            raise
        print(
            "[detector-ipc] direct %s load failed in this process; using external SG-Nav env worker: %s"
            % (detector_name, exc),
            flush=True,
        )
        detector = SubprocessDetector(
            detector_name,
            model_path,
            conf=conf,
            iou=iou,
            grounding_dino_config=str(getattr(maybe_build_detector, "grounding_dino_config", "") or ""),
            grounding_dino_text_threshold=float(getattr(maybe_build_detector, "grounding_dino_text_threshold", 0.25)),
            grounding_dino_device=str(getattr(maybe_build_detector, "grounding_dino_device", "cuda")),
        )
    detector.set_vocabulary(categories)
    return detector


def load_scene_categories(scene_dir: Path) -> List[str]:
    with open(scene_dir / "objects.json", "r", encoding="utf-8") as handle:
        objects = json.load(handle)
    return sorted({str(obj.get("category", "unknown")) for obj in objects})


def ensure_detector_loaded(args, scene_dir: Path, allow_ipc_fallback: bool = False) -> None:
    if args.detector == "none":
        old_detector = getattr(args, "_detector_instance", None)
        if old_detector is not None and hasattr(old_detector, "close"):
            old_detector.close()
        args._detector_instance = None
        args._detector_key = None
        return
    categories = load_scene_categories(scene_dir)
    conf = max(float(getattr(args, "detector_conf", 0.7)), float(getattr(args, "min_valid_detection_confidence", MIN_VALID_DETECTION_CONFIDENCE)))
    iou = float(getattr(args, "detector_iou", 0.5))
    model_path = detector_model_path(args)
    grounding_dino_config = str(getattr(args, "grounding_dino_config", "") or "")
    grounding_dino_text_threshold = float(getattr(args, "grounding_dino_text_threshold", 0.25))
    grounding_dino_device = str(getattr(args, "grounding_dino_device", "cuda") or "cuda")
    key = (
        str(args.detector),
        str(model_path),
        conf,
        iou,
        grounding_dino_config,
        grounding_dino_text_threshold,
        grounding_dino_device,
        bool(allow_ipc_fallback),
        tuple(categories),
    )
    if getattr(args, "_detector_key", None) == key:
        return
    old_detector = getattr(args, "_detector_instance", None)
    if old_detector is not None and hasattr(old_detector, "close"):
        old_detector.close()
    args._detector_instance = None
    args._detector_key = None
    maybe_build_detector.grounding_dino_config = grounding_dino_config
    maybe_build_detector.grounding_dino_text_threshold = grounding_dino_text_threshold
    maybe_build_detector.grounding_dino_device = grounding_dino_device
    args._detector_instance = maybe_build_detector(
        args.detector,
        model_path,
        categories,
        conf=conf,
        iou=iou,
        allow_ipc_fallback=allow_ipc_fallback,
    )
    args._detector_key = key


def detector_model_path(args) -> str:
    detector = str(getattr(args, "detector", "") or "").strip().lower()
    if detector == "grounding_dino":
        return str(getattr(args, "grounding_dino_checkpoint", "") or "")
    return str(getattr(args, "yolo_world_model", "") or "")


def frontier_room_contexts_for_debug(
    *,
    frontiers: Sequence[object],
    room_debug: Mapping[str, object],
    room_masks: Sequence[object],
    room_semantic_labels: Mapping[str, object],
    observed_free: np.ndarray,
    unknown: np.ndarray,
    agent_grid: Tuple[int, int],
    resolution_m: float,
    config: Mapping[str, object],
) -> list[dict]:
    cfg = dict(config or {})
    if not bool(cfg.get("enabled", True)) or not bool(cfg.get("use_known_free_side", True)):
        return []
    label_map_key = "context_room_label_map" if bool(cfg.get("use_context_overlay_labels", True)) else "final_room_label_map"
    labels = np.asarray(room_debug.get(label_map_key, room_debug.get("final_room_label_map", [])), dtype=np.int32)
    if labels.shape != np.asarray(observed_free).shape:
        return []
    label_to_room = _label_id_to_room_metadata(room_masks, room_semantic_labels)
    out: list[dict] = []
    for index, frontier in enumerate(frontiers):
        members = getattr(frontier, "members", [])
        context = assign_frontier_room_context(
            members,
            labels,
            observed_free,
            unknown,
            agent_grid,
            resolution_m,
            local_radius_m=float(cfg.get("local_radius_m", 0.35)),
            nearest_fallback_radius_m=float(cfg.get("nearest_fallback_radius_m", 1.25)),
            min_label_ratio=float(cfg.get("min_label_ratio", 0.20)),
        )
        meta = label_to_room.get(int(context["room_id"])) if context.get("room_id") is not None else None
        out.append(
            {
                "frontier_id": int(index),
                "center_grid": [int(v) for v in getattr(frontier, "center_grid", (-1, -1))],
                "room_context": {
                    **context,
                    "room_id": meta.get("room_id") if meta else context.get("room_id"),
                    "room_label_id": int(context["room_id"]) if context.get("room_id") is not None else None,
                    "room_label": meta.get("category", "unknown") if meta else "unknown",
                },
            }
        )
    return out


def _label_id_to_room_metadata(room_masks: Sequence[object], room_semantic_labels: Mapping[str, object]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for room in room_masks:
        label_id = int((getattr(room, "metadata", {}) or {}).get("label_id", 0) or 0)
        if label_id <= 0:
            continue
        room_id = str(getattr(room, "room_id", ""))
        semantic = room_semantic_labels.get(room_id)
        out[label_id] = {
            "room_id": room_id,
            "category": str(getattr(semantic, "category", "unknown") if semantic is not None else "unknown"),
        }
    return out


def ensure_segmenter_loaded(args) -> None:
    mode = str(getattr(args, "segmenter", "none") or "none").strip().lower()
    if mode in {"none", "false", "0", ""}:
        args._segmenter_instance = None
        args._segmenter_key = None
        return
    key = (
        mode,
        str(getattr(args, "sam2_checkpoint", "") or ""),
        str(getattr(args, "sam2_model_cfg", "") or ""),
        str(getattr(args, "sam2_device", "cuda") or "cuda"),
    )
    if getattr(args, "_segmenter_key", None) == key:
        return
    args._segmenter_instance = build_sam2_segmenter(
        mode=mode,
        checkpoint=str(getattr(args, "sam2_checkpoint", "") or ""),
        model_cfg=str(getattr(args, "sam2_model_cfg", "") or ""),
        device=str(getattr(args, "sam2_device", "cuda") or "cuda"),
        required=mode == "sam2",
    )
    args._segmenter_key = key


def build_sgnav_room_map(scene_dir: Path, map_info: MapInfo) -> Optional[np.ndarray]:
    rooms_path = scene_dir / "rooms.json"
    if not rooms_path.exists():
        return None
    try:
        rooms = load_rooms(str(rooms_path))
    except Exception:
        return None
    room_index = build_room_index_map(rooms, map_info)
    room_map = np.zeros((1, len(SGNAV_ROOM_NAMES), map_info.height, map_info.width), dtype=np.float32)
    normalized_room_names = [name.replace("_", " ") for name in SGNAV_ROOM_NAMES]
    for idx, room in enumerate(rooms):
        room_type = str(room.room_type).replace("_", " ")
        if room_type not in normalized_room_names:
            continue
        ch = normalized_room_names.index(room_type)
        room_map[0, ch][room_index == idx] = 1.0
    return room_map


def observed_room_map(full_room_map: Optional[np.ndarray], observed: np.ndarray) -> Optional[np.ndarray]:
    if full_room_map is None:
        return None
    return full_room_map * observed.astype(np.float32)[None, None, :, :]


def path_cells_to_world(path_cells: Iterable[Tuple[int, int]], map_info: MapInfo) -> List[Tuple[float, float]]:
    return [grid_to_world_xy(int(r), int(c), map_info) for r, c in path_cells]


def distance_to_target_cells_m(
    current_grid: Tuple[int, int],
    target_cells: Iterable[Tuple[int, int]],
    resolution_m: float,
) -> float:
    cells = [tuple(int(v) for v in cell) for cell in target_cells]
    if not cells:
        return float("inf")
    cur = np.asarray(current_grid, dtype=np.float32)
    arr = np.asarray(cells, dtype=np.float32)
    return float(np.min(np.linalg.norm(arr - cur[None, :], axis=1)) * float(resolution_m))


def navigation_target_reached(
    current_grid: Tuple[int, int],
    nav_decision: Optional[NavigationDecision],
    resolution_m: float,
    reached_radius_m: float,
) -> Tuple[bool, float]:
    if nav_decision is None or nav_decision.mode not in {"frontier", "candidate", "reperception"}:
        return False, float("inf")
    distance_m = distance_to_target_cells_m(current_grid, nav_decision.target_cells or [], resolution_m)
    return bool(distance_m <= float(reached_radius_m)), float(distance_m)


def planning_target_cells_within_radius(
    target_cells: Iterable[Tuple[int, int]],
    traversible: np.ndarray,
    resolution_m: float,
    radius_m: float,
) -> List[Tuple[int, int]]:
    cells = [tuple(int(v) for v in cell) for cell in target_cells]
    if not cells:
        return []
    nav = np.asarray(traversible, dtype=bool)
    h, w = nav.shape
    radius_cells = max(0, int(math.ceil(float(radius_m) / max(float(resolution_m), 1e-6))))
    candidates: dict[Tuple[int, int], Tuple[int, int, int]] = {}
    for target_idx, (row, col) in enumerate(cells):
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                if dr * dr + dc * dc > radius_cells * radius_cells:
                    continue
                rr, cc = int(row + dr), int(col + dc)
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not bool(nav[rr, cc]):
                    continue
                key = (rr, cc)
                rank = (int(dr * dr + dc * dc), int(target_idx), int(abs(dr) + abs(dc)))
                if key not in candidates or rank < candidates[key]:
                    candidates[key] = rank
    return [cell for cell, _rank in sorted(candidates.items(), key=lambda item: item[1])]


@dataclass
class LongTermGoalState:
    mode: str = "none"
    target_cells: List[Tuple[int, int]] = field(default_factory=list)
    center_grid: Optional[Tuple[int, int]] = None
    selected_step: int = -1
    reached: bool = False
    invalid_reason: str = ""
    nav_decision: Optional[NavigationDecision] = None

    def exists(self) -> bool:
        return self.mode not in {"", "none"} and bool(self.target_cells) and not self.invalid_reason

    def set_from(self, nav_decision: NavigationDecision, step: int) -> None:
        self.mode = str(nav_decision.mode or "none")
        self.target_cells = [tuple(int(v) for v in cell) for cell in (nav_decision.target_cells or [])]
        self.center_grid = self._center_from_decision(nav_decision)
        self.selected_step = int(step)
        self.reached = False
        self.invalid_reason = ""
        self.nav_decision = nav_decision

    def clear(self, reason: str = "cleared") -> None:
        self.mode = "none"
        self.target_cells = []
        self.center_grid = None
        self.selected_step = -1
        self.reached = reason == "reached"
        self.invalid_reason = ""
        self.nav_decision = None

    def invalidate(self, reason: str) -> None:
        self.invalid_reason = str(reason)
        self.mode = "none"
        self.target_cells = []
        self.center_grid = None
        self.nav_decision = None

    def to_navigation_decision(self) -> NavigationDecision:
        if self.nav_decision is None:
            return NavigationDecision(
                self.mode,
                list(self.target_cells),
                False,
                None,
                None,
                "continue_long_term_goal",
                metadata={
                    "long_term_goal_locked": True,
                    "long_term_goal_lock_reason": "locked_long_term_goal",
                    "long_term_goal_selected_step": int(self.selected_step),
                    "long_term_goal_mode": self.mode,
                },
            )
        metadata = {
            **dict(self.nav_decision.metadata or {}),
            "long_term_goal_locked": True,
            "long_term_goal_lock_reason": "locked_long_term_goal",
            "long_term_goal_selected_step": int(self.selected_step),
            "long_term_goal_mode": self.mode,
        }
        return NavigationDecision(
            self.nav_decision.mode,
            list(self.target_cells),
            bool(self.nav_decision.stop),
            self.nav_decision.selected_candidate,
            self.nav_decision.frontier_decision,
            self.nav_decision.reason,
            state=self.nav_decision.state,
            metadata=metadata,
        )

    @staticmethod
    def _center_from_decision(nav_decision: NavigationDecision) -> Optional[Tuple[int, int]]:
        if nav_decision.frontier_decision is not None and nav_decision.frontier_decision.selected_frontier is not None:
            return tuple(int(v) for v in nav_decision.frontier_decision.selected_frontier.center_grid)
        if nav_decision.selected_candidate is not None:
            return tuple(int(v) for v in nav_decision.selected_candidate.center_grid)
        if nav_decision.target_cells:
            return tuple(int(v) for v in nav_decision.target_cells[0])
        return None


def trim_path_to_current(path: List[Tuple[int, int]], current_grid: Tuple[int, int]) -> List[Tuple[int, int]]:
    for idx, cell in enumerate(path):
        if cell == current_grid:
            return path[idx:]
    return []


def trim_path_to_nearest(path: List[Tuple[int, int]], current_grid: Tuple[int, int], max_dist_cells: int = 3) -> List[Tuple[int, int]]:
    if not path:
        return []
    cur = np.asarray(current_grid, dtype=np.float32)
    arr = np.asarray(path, dtype=np.float32)
    dists = np.linalg.norm(arr - cur[None, :], axis=1)
    idx = int(np.argmin(dists))
    if float(dists[idx]) <= float(max_dist_cells):
        return path[idx:]
    return []


def mark_observed_disc(observed: np.ndarray, center: Tuple[int, int], radius_cells: int) -> None:
    r0, c0 = int(center[0]), int(center[1])
    h, w = observed.shape
    r_min, r_max = max(0, r0 - radius_cells), min(h - 1, r0 + radius_cells)
    c_min, c_max = max(0, c0 - radius_cells), min(w - 1, c0 + radius_cells)
    rr, cc = np.ogrid[r_min : r_max + 1, c_min : c_max + 1]
    mask = (rr - r0) ** 2 + (cc - c0) ** 2 <= radius_cells**2
    observed[r_min : r_max + 1, c_min : c_max + 1][mask] = True


def filter_detections_by_confidence(detections: List[Detection2D], min_confidence: float) -> List[Detection2D]:
    threshold = float(min_confidence)
    return [det for det in detections if detection_confidence_is_valid(float(det.confidence), threshold)]


def filter_edge_touching_detections(
    detections: List[Detection2D],
    *,
    image_width: int,
    image_height: int,
    step_idx: int,
    reject_edge_touching_bboxes: bool,
    margin_px: float,
    margin_ratio: float,
    min_confidence: Optional[float] = None,
    raw_log: Optional[List[dict]] = None,
) -> List[Detection2D]:
    kept: List[Detection2D] = []
    for idx, det in enumerate(detections):
        touches = bbox_touches_image_edge(
            det.bbox_xyxy,
            int(image_width),
            int(image_height),
            margin_px=float(margin_px),
            margin_ratio=float(margin_ratio),
        )
        low_confidence = (
            min_confidence is not None
            and not detection_confidence_is_valid(float(det.confidence), float(min_confidence))
        )
        det.bbox_touches_edge = bool(touches)
        # Edge-touching detector/SAM2 detections are partial visual evidence, not a
        # discard condition. The legacy flag is kept for CLI compatibility and
        # recorded below, but strict object tracking now decides policy use from
        # mask/depth association and track stability.
        rejected = bool(low_confidence)
        det.used_for_object_track = not rejected
        if low_confidence:
            det.reject_reason = "low_confidence"
        else:
            det.reject_reason = None
        record = {
            "raw_detection_id": "frame_%04d_det_%04d" % (int(step_idx), int(idx)),
            "step": int(step_idx),
            "category": normalize_category(det.category),
            "raw_label": str(det.raw_label),
            "confidence": float(det.confidence),
            "bbox_xyxy": [float(v) for v in det.bbox_xyxy],
            "bbox_touches_edge": bool(touches),
            "legacy_reject_edge_touching_bboxes_requested": bool(reject_edge_touching_bboxes),
            "visibility_status": "partial_edge" if touches else "unknown",
            "used_for_object_track": not rejected,
            "reject_reason": det.reject_reason,
        }
        if raw_log is not None:
            raw_log.append(record)
        if not rejected:
            kept.append(det)
    return kept


def goal_candidate_pair_distances(object_memory: ObjectMemory, goal_category: str, max_distance_m: float = 1.0) -> List[dict]:
    goal = normalize_category(goal_category)
    nodes = [node for node in object_memory.nodes if normalize_category(node.category) == goal]
    out = []
    for idx, a in enumerate(nodes):
        for b in nodes[idx + 1 :]:
            dist = float(np.linalg.norm(np.asarray(a.center_world[:2], dtype=np.float32) - np.asarray(b.center_world[:2], dtype=np.float32)))
            if dist <= float(max_distance_m):
                out.append({"a": int(a.node_id), "b": int(b.node_id), "dist": dist})
    return out


def candidate_center_payload(object_memory: ObjectMemory, goal_category: str, selected_candidate_id: Optional[int] = None) -> List[dict]:
    goal = normalize_category(goal_category)
    payload = []
    for node in object_memory.nodes:
        if normalize_category(node.category) != goal:
            continue
        status = "selected" if selected_candidate_id is not None and int(node.node_id) == int(selected_candidate_id) else "candidate"
        payload.append(
            {
                "node_id": int(node.node_id),
                "center_grid": tuple(int(v) for v in node.center_grid),
                "confidence": float(node.confidence),
                "observed_count": int(node.observed_count),
                "status": status,
            }
        )
    return payload


def build_reperception_state_payload(decision_metadata: Mapping[str, object]) -> dict:
    reperception = dict(decision_metadata.get("reperception") or {})
    candidate_id = reperception.get("candidate_id", decision_metadata.get("selected_candidate_id"))
    decision = reperception.get("decision")
    if decision is None:
        if bool(decision_metadata.get("candidate_accepted", False)):
            decision = "ACCEPT_GOAL"
        elif bool(decision_metadata.get("candidate_rejected", False)):
            decision = "REJECT_GOAL"
        elif candidate_id is not None:
            decision = "CONTINUE_OBSERVING"
    return {
        "candidate_id": candidate_id,
        "num_reperception_steps": int(
            reperception.get(
                "num_reperception_steps",
                decision_metadata.get("candidate_reperception_steps", 0),
            )
            or 0
        ),
        "accumulated_credibility": float(
            reperception.get(
                "accumulated_credibility",
                decision_metadata.get("candidate_credibility", 0.0),
            )
            or 0.0
        ),
        "last_s_k": float(reperception.get("last_s_k", reperception.get("s_k", 0.0)) or 0.0),
        "detector_confidence": float(reperception.get("detector_confidence", 0.0) or 0.0),
        "supporting_subgraphs": list(reperception.get("supporting_subgraphs") or []),
        "decision": decision,
    }


def build_stop_state_payload(nav_decision: Optional[NavigationDecision], row: Mapping[str, object]) -> dict:
    metadata = dict(getattr(nav_decision, "metadata", {}) or {}) if nav_decision is not None else {}
    stop_allowed = bool(getattr(nav_decision, "stop", False)) if nav_decision is not None else False
    candidate_confirmed = bool(metadata.get("candidate_accepted", False)) or str(getattr(nav_decision, "mode", "")) == "stop"
    if row.get("stop_blocked_reason"):
        reason = str(row["stop_blocked_reason"])
    elif not stop_allowed and not candidate_confirmed:
        reason = "candidate_not_confirmed"
    else:
        reason = str(row.get("stop_reason") or metadata.get("stop_reason") or getattr(nav_decision, "reason", ""))
    return {
        "stop_allowed": bool(stop_allowed),
        "stop_reason": reason,
        "candidate_confirmed": bool(candidate_confirmed),
        "policy_stop_confirmed": bool(row.get("policy_stop_confirmed", False)),
        "success_requires_sgnav_stop": bool(row.get("success_requires_sgnav_stop", False)),
        "gt_success_region_reached": bool(row.get("gt_success_region_reached", False)),
        "stop_blocked_reason": row.get("stop_blocked_reason"),
        "mode": getattr(nav_decision, "mode", None) if nav_decision is not None else None,
        "success": bool(row.get("success", False)),
        "distance_to_goal": row.get("distance_to_goal"),
    }


def success_region_can_finish(
    distance_to_goal: float,
    success_distance: float,
    *,
    require_sgnav_stop: bool,
    policy_stop_confirmed: bool,
    ignore_goal_success: bool = False,
) -> bool:
    if bool(ignore_goal_success):
        return False
    inside_success_region = float(distance_to_goal) <= float(success_distance)
    if not inside_success_region:
        return False
    return bool(policy_stop_confirmed or not require_sgnav_stop)


def final_log_row(row: dict) -> dict:
    if row.get("failure_reason"):
        stop_reason = row["failure_reason"]
    elif bool(row.get("success", False)):
        stop_reason = "success"
    elif row.get("terminal_reason"):
        stop_reason = row["terminal_reason"]
    else:
        stop_reason = "not_success"
    out = {
        "goal_category": row.get("goal_category"),
        "success": bool(row.get("success", False)),
        "distance_to_goal": row.get("distance_to_goal"),
        "spl": row.get("spl"),
        "stop_reason": stop_reason,
        "sgnav_decision_mode": row.get("sgnav_decision_mode"),
        "sgnav_decision_reason": row.get("sgnav_decision_reason"),
    }
    for key in (
        "frontier_target_mode",
        "frontier_center_grid",
        "frontier_actual_target_grid",
        "frontier_unreachable_recovery",
        "frontier_unreachable_reason",
        "frontier_stop_at_current_grid",
        "frontier_blacklisted",
        "active_long_term_goal_mode",
        "active_long_term_goal_age",
        "long_term_goal",
        "paper_llm_enabled",
        "paper_llm_requests",
        "hcot_llm_enabled",
        "hcot_llm_attempts",
        "hcot_llm_failures",
        "hcot_llm_fallbacks",
        "hcot_llm_last_error",
        "hc_p_num_subgraphs_total",
        "hc_p_num_subgraphs_scored",
        "stop_called",
        "policy_stop_confirmed",
        "success_requires_sgnav_stop",
        "gt_success_region_reached",
        "gt_success_without_sgnav_stop_steps",
        "explore_until_no_frontiers",
        "goal_success_ignored_steps",
        "stop_blocked_reason",
        "mapping_latency_ms",
        "mapping_latency_breakdown_avg_ms",
        "mapping_latency_breakdown_counts",
    ):
        if key in row:
            out[key] = row.get(key)
    return out


def detections_to_3d_static_map_ray(
    detections: List[Detection2D],
    camera_pose_world: Tuple[float, float, float, float],
    image_width: int,
    camera_hfov_deg: float,
    map_info: MapInfo,
    occupancy: np.ndarray,
    navigable: np.ndarray,
    max_range_m: float = 6.0,
    min_range_m: float = 0.20,
) -> List[Detection3D]:
    """Approximate RGB-only detections on the static map without depth reads."""

    if image_width <= 0:
        return []
    cam_x, cam_y, cam_z, cam_yaw = [float(v) for v in camera_pose_world]
    hfov = math.radians(float(camera_hfov_deg))
    fx = (float(image_width) * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
    cx = float(image_width) * 0.5
    step_m = max(float(map_info.resolution_m) * 0.5, 0.02)
    max_range = max(float(max_range_m), float(min_range_m) + step_m)
    out: List[Detection3D] = []

    for det in detections:
        x1, _y1, x2, _y2 = [float(v) for v in det.bbox_xyxy]
        u = 0.5 * (x1 + x2)
        ray_yaw = cam_yaw + math.atan2(u - cx, fx)
        cos_yaw = math.cos(ray_yaw)
        sin_yaw = math.sin(ray_yaw)
        hit_cell = None
        last_free_cell = None
        dist = float(min_range_m)
        while dist <= max_range:
            row, col = world_xy_to_grid(cam_x + cos_yaw * dist, cam_y + sin_yaw * dist, map_info)
            if not is_inside_grid(row, col, map_info):
                break
            if bool(occupancy[row, col]):
                hit_cell = (row, col)
                break
            if bool(navigable[row, col]):
                last_free_cell = (row, col)
            dist += step_m
        cell = hit_cell or last_free_cell
        if cell is None:
            continue
        wx, wy = grid_to_world_xy(cell[0], cell[1], map_info)
        out.append(
            Detection3D(
                category=det.category,
                raw_label=det.raw_label,
                confidence=float(det.confidence),
                center_world=(float(wx), float(wy), float(cam_z)),
                bbox_xyxy=det.bbox_xyxy,
            )
        )
    return out


def predict_kinematic_pose(pose: Tuple[float, float, float, float], cmd: Tuple[float, float, float], dt: float) -> Tuple[float, float, float, float]:
    x, y, z, yaw = [float(v) for v in pose]
    vx, vy, wz = [float(v) for v in cmd]
    dx = math.cos(yaw) * vx - math.sin(yaw) * vy
    dy = math.sin(yaw) * vx + math.cos(yaw) * vy
    yaw = yaw + wz * float(dt)
    while yaw > math.pi:
        yaw -= 2.0 * math.pi
    while yaw < -math.pi:
        yaw += 2.0 * math.pi
    return x + dx * float(dt), y + dy * float(dt), z, yaw


def pose_is_grid_safe(
    pose: Tuple[float, float, float, float],
    navigable: np.ndarray,
    map_info: MapInfo,
    camera_forward_offset_m: float = 0.0,
) -> bool:
    x, y, _z, yaw = [float(v) for v in pose]
    samples = [(x, y)]
    if abs(float(camera_forward_offset_m)) > 1e-6:
        samples.append((x + math.cos(yaw) * float(camera_forward_offset_m), y + math.sin(yaw) * float(camera_forward_offset_m)))
    h, w = navigable.shape
    for sx, sy in samples:
        r, c = world_xy_to_grid(sx, sy, map_info)
        if not (0 <= r < h and 0 <= c < w) or not bool(navigable[r, c]):
            return False
    return True


def guard_kinematic_cmd(
    pose: Tuple[float, float, float, float],
    cmd: Tuple[float, float, float],
    dt: float,
    navigable: np.ndarray,
    map_info: MapInfo,
    camera_forward_offset_m: float,
) -> Tuple[Tuple[float, float, float], bool]:
    for scale in (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625):
        scaled = (float(cmd[0]) * scale, float(cmd[1]) * scale, float(cmd[2]))
        if pose_is_grid_safe(predict_kinematic_pose(pose, scaled, dt), navigable, map_info, camera_forward_offset_m):
            return scaled, False
    rotate_only = (0.0, 0.0, float(cmd[2]))
    if abs(rotate_only[2]) > 1e-6 and pose_is_grid_safe(predict_kinematic_pose(pose, rotate_only, dt), navigable, map_info, camera_forward_offset_m):
        return rotate_only, False
    stopped = (0.0, 0.0, 0.0)
    if pose_is_grid_safe(predict_kinematic_pose(pose, stopped, dt), navigable, map_info, camera_forward_offset_m):
        return stopped, True
    return stopped, True


def seed_object_memory_from_preprocessed(scene_dir: Path, map_info: MapInfo, object_memory: ObjectMemory) -> int:
    objects_path = scene_dir / "objects.json"
    if not objects_path.exists():
        return 0
    with open(objects_path, "r", encoding="utf-8") as handle:
        objects = json.load(handle)
    detections: List[Detection3D] = []
    for obj in objects:
        center = obj.get("center_world")
        if not center or len(center) < 3:
            continue
        detections.append(
            Detection3D(
                category=str(obj.get("category", "unknown")),
                raw_label=str(obj.get("raw_label", obj.get("category", "unknown"))),
                confidence=1.0,
                center_world=(float(center[0]), float(center[1]), float(center[2])),
                bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
            )
        )
    object_memory.update(detections, step_id=0, map_info=map_info)
    return len(detections)


def detector_requires_rgb(detector, detector_name: str) -> bool:
    return detector is not None and str(detector_name) not in {"dry_run", "none"}


def detector_can_use_cuda_rgb(detector, detector_name: str, camera_annotator_device: str) -> bool:
    return (
        detector_requires_rgb(detector, detector_name)
        and not isinstance(detector, SubprocessDetector)
        and str(detector_name) == "yolo_world"
        and str(camera_annotator_device).lower() == "cuda"
    )


def run_episode_isaac_closed_loop(episode: dict, args) -> dict:
    from isaac_bench.env.isaac_process import IsaacSimServer

    scene_dir, static_map_info, static_occupancy, static_navigable = load_preprocessed_for_episode(episode)
    static_openings = load_passable_opening_mask(scene_dir, static_map_info)
    static_navigable = apply_episode_planning_clearance(
        scene_dir,
        static_map_info,
        static_navigable,
        episode,
        runtime_planning_clearance_m=getattr(args, "runtime_planning_clearance_m", 0.0),
    )
    ensure_detector_loaded(args, scene_dir, allow_ipc_fallback=True)
    detector = getattr(args, "_detector_instance", None)
    if detector_requires_rgb(detector, args.detector):
        ensure_segmenter_loaded(args)
        segmenter = getattr(args, "_segmenter_instance", None)
    else:
        segmenter = None
    camera_annotator_device = str(getattr(args, "camera_annotator_device", "cuda")).strip().lower()
    if camera_annotator_device not in {"cpu", "cuda"}:
        camera_annotator_device = "cpu"

    metric_planner = GridAStarPlanner(static_navigable, static_map_info.resolution_m, allow_diagonal=True)
    evaluator = EpisodeEvaluator(episode, metric_planner)
    object_memory = ObjectMemory(
        merge_radius_m=float(args.object_merge_radius_m),
        min_valid_confidence=float(args.min_valid_detection_confidence),
    )
    fused_instance_registry = FusedInstanceRegistry(
        merge_distance_m=float(getattr(args, "instance_merge_distance_m", args.object_merge_radius_m)),
        merge_iou_3d=float(getattr(args, "instance_merge_iou_3d", 0.15)),
        min_valid_confidence=float(args.min_valid_detection_confidence),
        reject_edge_touching_bboxes=bool(args.reject_edge_touching_bboxes),
        bbox_edge_margin_px=float(args.bbox_edge_margin_px),
        bbox_edge_margin_ratio=float(args.bbox_edge_margin_ratio),
        partial_class_weight=float(args.partial_class_weight),
        min_geometry_confidence=float(args.min_geometry_confidence),
        partial_stability_min_observations=int(args.partial_stability_min_observations),
        mask_iou_association_threshold=float(args.mask_iou_association_threshold),
        mask_containment_track_match_threshold=float(args.mask_containment_track_match_threshold),
        footprint_iou_association_threshold=float(args.footprint_iou_association_threshold),
        child_containment_threshold=float(args.child_containment_threshold),
        child_area_ratio_threshold=float(args.child_area_ratio_threshold),
    )
    if args.seed_gt_object_memory:
        print("[sgnav-loop] seed_gt_object_memory ignored for depth-online mapping", flush=True)
    seeded = 0
    scenegraph = SGNavSceneGraphAdapter(
        args.sgnav_repo,
        use_original=args.use_original_scenegraph,
        semantic_priors_path=getattr(args, "semantic_priors_path", None),
        vllm_config={
            "enabled": bool(getattr(args, "vllm_frontier_scoring", False)),
            "base_url": getattr(args, "vllm_base_url", None),
            "model": getattr(args, "vllm_model", None),
            "timeout_s": float(getattr(args, "vllm_timeout_s", 8.0)),
            "temperature": float(getattr(args, "vllm_temperature", 0.0)),
            "max_frontiers": int(getattr(args, "vllm_max_frontiers", 32)),
            "include_image": bool(getattr(args, "vllm_image_scoring", True)),
            "image_max_width": int(getattr(args, "vllm_image_max_width", 640)),
            "image_jpeg_quality": int(getattr(args, "vllm_image_jpeg_quality", 75)),
        },
        llm_config={
            "enabled": bool(getattr(args, "llm_enabled", False)),
            "base_url": getattr(args, "llm_base_url", None),
            "model": getattr(args, "llm_model", None),
            "api_key": getattr(args, "llm_api_key", None),
            "timeout_s": float(getattr(args, "llm_timeout_s", 30.0)),
            "temperature": float(getattr(args, "llm_temperature", 0.0)),
            "max_tokens": int(getattr(args, "llm_max_tokens", 512)),
            "max_hcot_subgraphs_per_decision": int(getattr(args, "max_hcot_subgraphs_per_decision", 8)),
            "strict_benchmark": bool(getattr(args, "strict_benchmark", False)),
        },
        sgnav_mode=str(getattr(args, "sgnav_mode", "legacy")),
    )
    if bool(getattr(args, "vllm_frontier_scoring", False)):
        print(
            "[sgnav-vllm] frontier scoring enabled: base_url=%s model=%s image=%s"
            % (
                getattr(args, "vllm_base_url", "http://127.0.0.1:8000/v1"),
                getattr(args, "vllm_model", "qwen3-vl-8b-instruct"),
                bool(getattr(args, "vllm_image_scoring", True)),
            ),
            flush=True,
        )
    scenegraph.reset(episode["goal_category"])
    full_room_map = None
    scenegraph.update(object_memory, room_map=full_room_map)
    evaluator.num_scenegraph_updates += 1
    decision_policy = SGNavDecision(
        scenegraph,
        frontier_distance_weight=float(args.frontier_distance_weight),
        frontier_min_select_distance_m=float(args.frontier_min_distance_m),
        frontier_allow_near_fallback=bool(args.frontier_allow_near_fallback),
        candidate_min_hits=int(args.candidate_min_detector_hits),
        candidate_start_min_confidence=float(args.candidate_start_min_confidence),
        candidate_start_min_hits=int(args.candidate_start_min_hits),
        candidate_recent_max_age_steps=int(args.candidate_recent_max_age_steps),
        candidate_match_substring=bool(args.candidate_match_substring),
        candidate_accept_requires_reperception=bool(args.candidate_accept_requires_reperception),
        candidate_reject_ttl_steps=int(args.candidate_reject_ttl_steps),
        candidate_accept_threshold=float(args.candidate_accept_threshold),
        candidate_stop_distance_m=float(args.candidate_stop_distance_m),
        candidate_standoff_min_m=float(args.candidate_standoff_min_m),
        candidate_standoff_max_m=float(args.candidate_standoff_max_m),
        candidate_standoff_max_cells=int(args.candidate_standoff_max_cells),
        candidate_standoff_ideal_m=float(args.candidate_standoff_ideal_m),
        reperception_enabled=bool(args.reperception_enabled),
        reperception_min_observations=int(args.reperception_min_observations),
        reperception_max_steps=int(args.reperception_max_steps),
        reperception_same_goal_radius_m=float(args.reperception_same_goal_radius_m),
        stop_verification_steps=int(args.stop_verification_steps),
        stop_verification_min_hits=int(args.stop_verification_min_hits),
        found_goal_stop_distance_m=float(args.found_goal_stop_distance_m),
        score_frontiers_before_candidate=bool(args.score_frontiers_before_candidate),
        frontier_scenegraph_score_norm=str(args.frontier_scenegraph_score_norm),
        frontier_selection_mode=str(args.frontier_selection_mode),
        frontier_random_seed=int(args.frontier_random_seed),
    )
    follower = HolonomicWaypointFollower(
        max_vx=float(args.max_vx_mps),
        max_vy=float(args.max_vy_mps),
        max_wz=float(args.max_wz_radps),
        lookahead_m=float(args.lookahead_m),
    )
    vertical_or_free_cfg = dict(getattr(args, "room_segmentation_config", {}).get("vertical_or_free", {}) or {})
    roomseg_depth_stride_px = _resolve_roomseg_depth_stride_px(
        getattr(args, "room_segmentation_config", {}),
        int(args.depth_stride_px),
    )
    if int(roomseg_depth_stride_px) != int(args.depth_stride_px):
        print(
            "[sgnav-roomseg] vertical-free depth stride: mapping=%d roomseg_effective=%d"
            % (int(args.depth_stride_px), int(roomseg_depth_stride_px)),
            flush=True,
        )
    mapper = OnlineMapper(
        size_m=float(args.online_map_size_m),
        resolution_m=float(args.online_resolution_m),
        depth_max_m=float(args.depth_max_m),
        depth_min_m=float(args.depth_min_m),
        depth_stride_px=int(roomseg_depth_stride_px),
        obstacle_min_height_m=float(args.obstacle_min_height_m),
        obstacle_max_height_m=float(args.obstacle_max_height_m),
        free_min_height_m=float(args.free_min_height_m),
        free_max_height_m=float(args.free_max_height_m),
        vertical_profile_free_min_height_m=float(vertical_or_free_cfg.get("z_min_m", 0.20)),
        vertical_profile_free_max_height_m=float(vertical_or_free_cfg.get("z_max_m", 2.00)),
        splat_point_threshold=int(args.splat_point_threshold),
        free_splat_point_threshold=int(args.free_splat_point_threshold),
        robot_radius_m=float(args.robot_radius_m),
        inflation_radius_m=float(args.online_inflation_radius_m),
    )
    static_goal_cells = [(int(r), int(c)) for r, c in episode["goal_regions_grid"]]
    start_pose = tuple(float(v) for v in episode["start_pose_world"])
    mapper.reset((float(start_pose[0]), float(start_pose[1])))
    dynamic_map_info = mapper.grid.map_info
    frontier_commitment = (
        FrontierCommitmentManager(
            resolution_m=dynamic_map_info.resolution_m,
            match_radius_m=float(args.frontier_commit_match_radius_m),
            reached_radius_m=float(args.frontier_commit_reached_radius_m),
            min_commit_steps=int(args.frontier_commit_min_steps),
            max_commit_steps=int(args.frontier_commit_max_steps),
            switch_margin=float(args.frontier_commit_switch_margin),
            switch_ratio=float(args.frontier_commit_switch_ratio),
            no_progress_steps=int(args.frontier_commit_no_progress_steps),
            progress_min_delta_m=float(args.frontier_commit_progress_min_delta_m),
            blacklist_ttl_steps=int(args.frontier_blacklist_ttl_steps),
        )
        if bool(args.frontier_commitment_enabled)
        else None
    )
    room_map_mode = str(getattr(args, "room_map_mode", "observed_rooms_json") or "none").strip().lower()
    room_segmenter = None
    room_labeler = None
    room_semantic_labels = {}
    last_room_masks = []
    room_context_cache = RoomContextCache()
    last_room_context_result: Optional[RoomContextResult] = None
    last_room_context_metadata = room_context_not_invoked_metadata()
    last_room_segmentation_debug = {
        "source": VERTICAL_FREE_GAP_CLOSURE_ALGORITHM,
        "algorithm": VERTICAL_FREE_GAP_CLOSURE_ALGORITHM,
        "room_count": 0,
        "rooms": [],
    }
    last_room_semantics_debug = {
        "backend": "unavailable",
        "allowed_categories": list(getattr(args, "room_label_allowed_categories", DEFAULT_ROOM_CATEGORIES)),
        "labels": [],
    }
    if room_map_mode in {"observed_rooms_json", "rooms_json", "observed"}:
        if bool(getattr(args, "strict_benchmark", False)):
            raise BenchmarkAssetError(
                "strict SG-Nav metric path requires rose2_source_external_runner room masks; rooms.json/oracle room maps are not allowed"
            )
        full_room_map = build_sgnav_room_map(scene_dir, dynamic_map_info)
        scenegraph.update(object_memory, room_map=full_room_map)
    elif room_map_mode in {
        "rose2_source_form",
        "rose2_source_form_vlm",
        "rose2_source_form_v2",
        "rose2_source_form_v2_vlm",
        "rose2_source_faithful_v1",
        "rose2_source_faithful_v1_vlm",
        "upstream_rose2_vertical_or_free",
        "upstream_rose2_vertical_or_free_vlm",
        "upstream_rose2_pure_python",
        "upstream_rose2_pure_python_vlm",
        VERTICAL_FREE_ROOMSEG_ALGORITHM,
        VERTICAL_FREE_ROOMSEG_CONTEXT,
        VERTICAL_FREE_ROOMSEG_BACKEND,
        VERTICAL_FREE_GAP_CLOSURE_ALGORITHM,
        VERTICAL_FREE_GAP_CLOSURE_CONTEXT,
        VERTICAL_FREE_GAP_CLOSURE_BACKEND,
    }:
        roomseg_backend = str(getattr(args, "room_segmentation_config", {}).get("backend", "rose2_source_external_runner") or "rose2_source_external_runner").strip().lower()
        strict_roomseg_backends = {
            "rose2_source_external_runner",
            VERTICAL_FREE_GAP_CLOSURE_BACKEND,
            VERTICAL_FREE_GAP_CLOSURE_ALGORITHM,
        }
        legacy_vertical_free_ablation = (
            roomseg_backend in {VERTICAL_FREE_ROOMSEG_BACKEND, VERTICAL_FREE_ROOMSEG_ALGORITHM}
            and str(getattr(args, "ablation_name", "") or "") == "vertical_free_geodesic_room_ablation"
        )
        if bool(getattr(args, "strict_benchmark", False)) and roomseg_backend not in strict_roomseg_backends and not legacy_vertical_free_ablation:
            raise BenchmarkAssetError(
                "strict SG-Nav metric path requires room_segmentation.backend=vertical_free_gap_closure_v1 or rose2_source_external_runner; %s is debug/ablation-only"
                % roomseg_backend
            )
        upstream_cfg = UpstreamROSE2Config.from_mapping(
            getattr(args, "room_segmentation_config", {}),
            resolution_m=float(dynamic_map_info.resolution_m),
            fail_on_missing_source=bool(getattr(args, "strict_benchmark", False))
            and not bool(getattr(args, "allow_debug_fallbacks", False))
            and roomseg_backend == "rose2_source_external_runner"
            and bool(getattr(args, "room_segmentation_config", {}).get("require_upstream_source_for_strict", True)),
        )
        room_segmenter = UpstreamROSE2PurePythonSegmenter(upstream_cfg, dynamic_map_info)
        room_label_client = (
            getattr(scenegraph, "paper_llm_client", None)
            if str(getattr(args, "room_label_backend", "vlm")).strip().lower() == "vlm"
            else None
        )
        room_labeler = VLMRoomLabeler(
            client=room_label_client,
            allowed_categories=getattr(args, "room_label_allowed_categories", DEFAULT_ROOM_CATEGORIES),
            min_confidence=float(getattr(args, "room_label_min_confidence", 0.60)),
            ambiguity_margin=float(getattr(args, "room_label_ambiguity_margin", 0.15)),
            min_reliable_objects=int(getattr(args, "room_label_min_reliable_objects", 2)),
            unknown_category=str(getattr(args, "room_label_unknown_category", "unknown")),
            require_backend=bool(getattr(args, "strict_benchmark", False))
            and str(getattr(args, "room_label_backend", "vlm")).strip().lower() == "vlm",
            max_room_objects_in_prompt=int(getattr(args, "max_room_objects_in_prompt", 25)),
        )
        last_room_semantics_debug["backend"] = room_labeler.backend
    elif room_map_mode in {"online_rose2_structure", "rose2_structure", "online_rose2_structure_vlm"}:
        if bool(getattr(args, "strict_benchmark", False)) and str(getattr(args, "ablation_name", "") or "") != "local_rose2_lite_room_ablation":
            raise BenchmarkAssetError(
                "strict SG-Nav metric path requires rose2_source_external_runner room masks; local ROSE2-lite is debug/ablation-only"
            )
        room_cfg = RoomSegmentationConfig.from_mapping(
            getattr(args, "room_segmentation_config", {}),
            resolution_m=float(dynamic_map_info.resolution_m),
            map_info=dynamic_map_info,
        )
        room_segmenter = OnlineROSE2RoomSegmenter(room_cfg)
        room_label_client = (
            getattr(scenegraph, "paper_llm_client", None)
            if str(getattr(args, "room_label_backend", "vlm")).strip().lower() == "vlm"
            else None
        )
        room_labeler = VLMRoomLabeler(
            client=room_label_client,
            allowed_categories=getattr(args, "room_label_allowed_categories", DEFAULT_ROOM_CATEGORIES),
            min_confidence=float(getattr(args, "room_label_min_confidence", 0.60)),
            ambiguity_margin=float(getattr(args, "room_label_ambiguity_margin", 0.15)),
            min_reliable_objects=int(getattr(args, "room_label_min_reliable_objects", 2)),
            unknown_category=str(getattr(args, "room_label_unknown_category", "unknown")),
            require_backend=bool(getattr(args, "strict_benchmark", False))
            and str(getattr(args, "room_label_backend", "vlm")).strip().lower() == "vlm",
            max_room_objects_in_prompt=int(getattr(args, "max_room_objects_in_prompt", 25)),
        )
        last_room_semantics_debug["backend"] = room_labeler.backend
    elif room_map_mode in {
        ONLINE_ROSE_STYLE_BACKEND,
        ONLINE_ROSE_STYLE_CONTEXT,
        "online_rose_style",
        "online_rose_style_vlm",
    }:
        roomseg_backend = str(getattr(args, "room_segmentation_config", {}).get("backend", ONLINE_ROSE_STYLE_BACKEND) or ONLINE_ROSE_STYLE_BACKEND).strip().lower()
        if roomseg_backend != ONLINE_ROSE_STYLE_BACKEND:
            raise ValueError("online_rose_style room_map_mode requires --roomseg-backend %s" % ONLINE_ROSE_STYLE_BACKEND)
        online_cfg = OnlineRoseStyleConfig.from_mapping(
            getattr(args, "room_segmentation_config", {}),
            resolution_m=float(dynamic_map_info.resolution_m),
            map_info=dynamic_map_info,
        )
        room_segmenter = OnlineRoseStyleRoomSegmenter(online_cfg, map_info=dynamic_map_info)
        room_label_client = (
            getattr(scenegraph, "paper_llm_client", None)
            if str(getattr(args, "room_label_backend", "vlm")).strip().lower() == "vlm"
            else None
        )
        room_labeler = VLMRoomLabeler(
            client=room_label_client,
            allowed_categories=getattr(args, "room_label_allowed_categories", DEFAULT_ROOM_CATEGORIES),
            min_confidence=float(getattr(args, "room_label_min_confidence", 0.60)),
            ambiguity_margin=float(getattr(args, "room_label_ambiguity_margin", 0.15)),
            min_reliable_objects=int(getattr(args, "room_label_min_reliable_objects", 2)),
            unknown_category=str(getattr(args, "room_label_unknown_category", "unknown")),
            require_backend=bool(getattr(args, "strict_benchmark", False))
            and str(getattr(args, "room_label_backend", "vlm")).strip().lower() == "vlm",
            max_room_objects_in_prompt=int(getattr(args, "max_room_objects_in_prompt", 25)),
        )
        last_room_semantics_debug["backend"] = room_labeler.backend
    elif room_map_mode in {
        ONLINE_WATERSHED_ROOMSEG_BACKEND,
        ONLINE_WATERSHED_ROOMSEG_CONTEXT,
        "online_watershed_roomseg",
        "online_watershed_roomseg_vlm",
    }:
        roomseg_backend = str(
            getattr(args, "room_segmentation_config", {}).get("backend", ONLINE_WATERSHED_ROOMSEG_BACKEND)
            or ONLINE_WATERSHED_ROOMSEG_BACKEND
        ).strip().lower()
        if roomseg_backend != ONLINE_WATERSHED_ROOMSEG_BACKEND:
            raise ValueError("online_watershed_roomseg room_map_mode requires --roomseg-backend %s" % ONLINE_WATERSHED_ROOMSEG_BACKEND)
        watershed_cfg = OnlineWatershedRoomSegConfig.from_mapping(
            getattr(args, "room_segmentation_config", {}),
            resolution_m=float(dynamic_map_info.resolution_m),
            map_info=dynamic_map_info,
        )
        room_segmenter = OnlineWatershedRoomSegmenter(watershed_cfg, map_info=dynamic_map_info)
        room_label_client = (
            getattr(scenegraph, "paper_llm_client", None)
            if str(getattr(args, "room_label_backend", "vlm")).strip().lower() == "vlm"
            else None
        )
        room_labeler = VLMRoomLabeler(
            client=room_label_client,
            allowed_categories=getattr(args, "room_label_allowed_categories", DEFAULT_ROOM_CATEGORIES),
            min_confidence=float(getattr(args, "room_label_min_confidence", 0.60)),
            ambiguity_margin=float(getattr(args, "room_label_ambiguity_margin", 0.15)),
            min_reliable_objects=int(getattr(args, "room_label_min_reliable_objects", 2)),
            unknown_category=str(getattr(args, "room_label_unknown_category", "unknown")),
            require_backend=bool(getattr(args, "strict_benchmark", False))
            and str(getattr(args, "room_label_backend", "vlm")).strip().lower() == "vlm",
            max_room_objects_in_prompt=int(getattr(args, "max_room_objects_in_prompt", 25)),
        )
        last_room_semantics_debug["backend"] = room_labeler.backend
    elif room_map_mode in {
        "online_geometry_watershed",
        "online_geometry_watershed_vlm",
        "online_geometry_watershed_vertical_free",
        "online_geometry_watershed_vertical_free_vlm",
    }:
        if bool(getattr(args, "strict_benchmark", False)) and str(getattr(args, "ablation_name", "") or "") != "legacy_watershed_room_ablation":
            raise BenchmarkAssetError(
                "strict SG-Nav metric path requires rose2_source_external_runner room masks; watershed is debug/ablation-only"
            )
        room_cfg = RoomSegmentationConfig.from_mapping(
            getattr(args, "room_segmentation_config", {}),
            resolution_m=float(dynamic_map_info.resolution_m),
            map_info=dynamic_map_info,
        )
        if "vertical_free" in room_map_mode:
            room_cfg.algorithm = "legacy_watershed_vertical_free_ablation"
            room_cfg.source_grid = "vertical_profile_free_0p2_2p0"
        else:
            room_cfg.algorithm = "legacy_watershed_ablation"
        room_segmenter = OnlineRoomSegmenter(room_cfg)
        room_label_client = (
            getattr(scenegraph, "paper_llm_client", None)
            if str(getattr(args, "room_label_backend", "vlm")).strip().lower() == "vlm"
            else None
        )
        room_labeler = VLMRoomLabeler(
            client=room_label_client,
            allowed_categories=getattr(args, "room_label_allowed_categories", DEFAULT_ROOM_CATEGORIES),
            min_confidence=float(getattr(args, "room_label_min_confidence", 0.60)),
            ambiguity_margin=float(getattr(args, "room_label_ambiguity_margin", 0.15)),
            min_reliable_objects=int(getattr(args, "room_label_min_reliable_objects", 2)),
            unknown_category=str(getattr(args, "room_label_unknown_category", "unknown")),
            require_backend=bool(getattr(args, "strict_benchmark", False))
            and str(getattr(args, "room_label_backend", "vlm")).strip().lower() == "vlm",
            max_room_objects_in_prompt=int(getattr(args, "max_room_objects_in_prompt", 25)),
        )
        last_room_semantics_debug["backend"] = room_labeler.backend
    elif room_map_mode in {"none", "disabled", ""}:
        pass
    else:
        raise ValueError("unsupported mapping.room_map_mode: %s" % room_map_mode)
    goal_cells = []
    for goal_r, goal_c in static_goal_cells:
        gx, gy = grid_to_world_xy(goal_r, goal_c, static_map_info)
        dyn_goal = world_xy_to_grid(gx, gy, dynamic_map_info)
        if is_inside_grid(dyn_goal[0], dyn_goal[1], dynamic_map_info):
            goal_cells.append(dyn_goal)
    max_steps = int(args.max_control_steps)
    replan_every = max(1, int(args.replan_every_steps))
    requested_perception_every = max(1, int(args.perception_every_steps))
    perception_every = effective_perception_every_steps(args.detector, requested_perception_every)
    if perception_every != requested_perception_every:
        print(
            "[sgnav-loop] open-vocabulary detector requires every-frame perception; overriding perception_every_steps "
            "%d -> %d" % (requested_perception_every, perception_every),
            flush=True,
        )
    success_distance = float(episode.get("success_distance_m", 1.0))
    explore_until_no_frontiers = bool(getattr(args, "explore_until_no_frontiers", False))
    frontier_mask_probe = str(getattr(args, "policy", "") or "") in {
        "random_frontier_mask_probe",
        "nearest_frontier_mask_probe",
    }
    current_path: List[Tuple[int, int]] = []
    full_path: List[Tuple[int, int]] = []
    failure_reason = None
    stop_called = False
    policy_stop_confirmed = False
    gt_success_region_reached = False
    gt_success_without_sgnav_stop_steps = 0
    gt_success_ignored_steps = 0
    target_radius_reached_steps = 0
    nav_execution_progress_key = None
    nav_execution_best_distance_m = float("inf")
    nav_execution_no_progress_steps = 0
    stop_blocked_reason = None
    logged_gt_success_without_sgnav_stop = False
    last_detections_2d: List[Detection2D] = []
    detection_category_counts = Counter()
    goal_detection_history = []
    last_frontiers = []
    last_nav_decision = None
    last_frontier_commitment_metadata = {}
    last_frontier_commitment_reason = ""
    last_selected_candidate = None
    logged_selected_candidate_id = None
    raw_detection_debug_log: List[dict] = []
    last_dynamic_occupancy = mapper.grid.occupied.astype(bool)
    last_dynamic_free = mapper.grid.free.astype(bool)
    last_dynamic_navigable = mapper.traversible(unknown_is_obstacle=True)
    last_dynamic_astar_navigable = apply_dynamic_astar_edge_clearance(
        last_dynamic_navigable,
        mapper.grid.occupied,
        mapper.grid.map_info.resolution_m,
        float(args.runtime_planning_clearance_m),
    )
    last_dynamic_observed = mapper.grid.observed.astype(bool)
    last_frontier_raw_cells = 0
    last_frontier_clusters = 0
    last_roomseg_snapshot_frontier_key = None
    frontier_target_mode = None
    frontier_center_grid = None
    frontier_actual_target_grid = None
    frontier_unreachable_recovery = False
    frontier_unreachable_reason = None
    frontier_stop_at_current_grid = None
    frontier_blacklisted = False
    paper_mode = str(getattr(args, "sgnav_mode", "legacy")).strip().lower() == "paper"
    long_term_goal = LongTermGoalState()
    sgnav_viz_enabled = bool(getattr(args, "sgnav_viz", False))
    sgnav_viz_save_dir = getattr(args, "sgnav_viz_save_dir", None)
    detection_localization = str(getattr(args, "detection_localization", "static_map_ray")).strip().lower()
    if detection_localization in {"static_map_ray", "map_ray", "rgb_map_ray"}:
        print("[sgnav-loop] static-map detection localization disabled; using depth projection", flush=True)
        detection_localization = "depth"
    args.read_depth = True
    roomseg_debug_only = bool(getattr(args, "roomseg_debug_only", False))
    viz = None
    viz_requested = bool(sgnav_viz_enabled or sgnav_viz_save_dir)
    viz_every = max(1, int(getattr(args, "sgnav_viz_every_steps", 1)))
    detector_cuda_rgb = detector_can_use_cuda_rgb(detector, args.detector, camera_annotator_device)
    vllm_needs_cpu_rgb = bool(getattr(args, "vllm_frontier_scoring", False) and getattr(args, "vllm_image_scoring", True))
    logged_detector_rgb_device = False
    latency_totals_ms = {
        "perception": 0.0,
        "mapping": 0.0,
        "graph": 0.0,
        "llm": 0.0,
        "planning": 0.0,
    }
    latency_counts = {key: 0 for key in latency_totals_ms}
    mapping_breakdown_totals_ms: dict[str, float] = {}
    mapping_breakdown_counts: dict[str, int] = {}

    def record_latency(name: str, started_at: float) -> None:
        latency_totals_ms[name] += max(0.0, (time.perf_counter() - started_at) * 1000.0)
        latency_counts[name] += 1

    def record_mapping_timing(name: str, elapsed_ms: float) -> None:
        key = str(name)
        value = max(0.0, float(elapsed_ms))
        mapping_breakdown_totals_ms[key] = mapping_breakdown_totals_ms.get(key, 0.0) + value
        mapping_breakdown_counts[key] = mapping_breakdown_counts.get(key, 0) + 1

    def record_mapping_breakdown(breakdown: Mapping[str, object]) -> None:
        for key, value in dict(breakdown).items():
            if key == "reason" or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                record_mapping_timing(str(key), float(value))

    def total_llm_requests() -> int:
        vllm_scorer = getattr(scenegraph, "vllm_scorer", None)
        paper_llm_client = getattr(scenegraph, "paper_llm_client", None)
        hcot_scorer = getattr(scenegraph, "hcot_scorer", None)
        return (
            int(getattr(vllm_scorer, "request_count", 0))
            + int(getattr(paper_llm_client, "request_count", 0))
            + int(getattr(hcot_scorer, "llm_request_count", 0))
        )

    def needs_viz_frame(step_idx: int) -> bool:
        return viz_requested and step_idx < max_steps and step_idx % viz_every == 0

    def rgb_request_device(step_idx: int) -> Optional[str]:
        if step_idx >= max_steps:
            return None
        if detector_requires_rgb(detector, args.detector) and step_idx % perception_every == 0:
            return "cuda" if detector_cuda_rgb else "cpu"
        if needs_viz_frame(step_idx):
            return "cpu"
        return None

    server = IsaacSimServer(
        headless=args.headless,
        width=int(args.isaac_width),
        height=int(args.isaac_height),
        camera_hfov_deg=float(args.camera_hfov_deg),
        mast_height_m=float(args.camera_mast_height_m),
        forward_offset_m=float(args.camera_forward_offset_m),
        camera_pitch_deg=float(args.camera_pitch_deg),
        camera_near_m=float(args.camera_near_m),
        camera_far_m=float(args.camera_far_m),
        enable_depth=bool(getattr(args, "read_depth", False)),
        camera_annotator_device=camera_annotator_device,
        enable_nearfield_depth=bool(getattr(args, "nearfield_depth", False)),
        nearfield_width=int(args.nearfield_width),
        nearfield_height=int(args.nearfield_height),
        nearfield_hfov_deg=float(args.nearfield_hfov_deg),
        nearfield_height_m=float(args.nearfield_height_m),
        nearfield_near_m=float(args.nearfield_near_m),
        nearfield_far_m=float(args.nearfield_far_m),
    )
    try:
        first_rgb_device = rgb_request_device(0)
        obs = server.reset_episode(
            episode["usd_path"],
            start_pose,
            read_rgb=first_rgb_device is not None,
            rgb_device=first_rgb_device,
        )
        print("[sgnav-loop] first Isaac observation received", flush=True)
        viz = SGNavPopupVisualizer(
            enabled=sgnav_viz_enabled,
            save_dir=sgnav_viz_save_dir,
            panel_size=(int(args.sgnav_viz_width), int(args.sgnav_viz_height)),
            save_every_steps=int(args.sgnav_viz_save_every_steps),
            ipc_jpeg_quality=int(args.sgnav_viz_jpeg_quality),
            debug_overlay_layers=bool(args.debug_overlay_layers),
            save_overlay_layer_metadata=bool(args.save_overlay_layer_metadata),
            show_gt_goal_cells=bool(args.show_gt_goal_cells),
            show_room_proposals=bool(args.show_room_proposals),
            show_room_masks=bool(args.show_room_masks),
            show_room_labels=bool(args.show_room_labels),
            show_frontier_member_cells=bool(args.show_frontier_member_cells),
            show_object_nodes=bool(args.show_object_nodes),
            show_candidate_markers=bool(args.show_candidate_markers),
            min_valid_detection_confidence=float(args.min_valid_detection_confidence),
            max_green_like_primitives_before_warning=int(args.max_green_like_primitives_before_warning),
        ) if (sgnav_viz_enabled or sgnav_viz_save_dir) else None
        intr = CameraIntrinsics.from_hfov(int(args.isaac_width), int(args.isaac_height), float(args.camera_hfov_deg))
        nearfield_intr = CameraIntrinsics.from_hfov(
            int(args.nearfield_width),
            int(args.nearfield_height),
            float(args.nearfield_hfov_deg),
        )

        def viz_rgb(current_obs: dict) -> np.ndarray:
            if current_obs.get("has_rgb") and current_obs.get("rgb_device") == "cpu":
                return current_obs["rgb"]
            return server.get_observation(read_rgb=True, read_depth=False, rgb_device="cpu")["rgb"]

        last_decision_mode = "init"
        last_decision_reason = ""
        goal_candidate_count = 0
        sam2_failure_logged = False
        force_perception_step = False
        panorama_frames = 0

        goal_norm = normalize_category(episode["goal_category"])

        def category_matches_goal(category: str) -> bool:
            cat_norm = normalize_category(category)
            return cat_norm == goal_norm or (goal_norm and (goal_norm in cat_norm or cat_norm in goal_norm))

        def summarize_detection(det: Detection2D, step_idx: int) -> dict:
            return {
                "step": int(step_idx),
                "category": normalize_category(det.category),
                "raw_label": str(det.raw_label),
                "confidence": float(det.confidence),
                "bbox_xyxy": [float(v) for v in det.bbox_xyxy],
            }

        def update_mapper_state(current_obs: dict, step_idx: int | None = None):
            nonlocal dynamic_map_info, last_dynamic_occupancy, last_dynamic_free, last_dynamic_navigable, last_dynamic_astar_navigable, last_dynamic_observed
            pose_local = current_obs["pose_world"]
            if not current_obs.get("has_depth"):
                return None
            started_at = time.perf_counter()
            mapper.update(current_obs["depth"], intr, pose_local, current_obs["camera_pose_world"])
            record_mapping_breakdown(getattr(mapper, "last_timing_stats", {}))
            mapper.last_debug_stats["depth_source"] = str(current_obs.get("depth_source", "unknown"))
            mapper.last_debug_stats["camera_frame_sync_updates"] = int(current_obs.get("camera_frame_sync_updates", 0) or 0)
            mapper.last_debug_stats["camera_rendering_time"] = current_obs.get("camera_rendering_time")
            if bool(getattr(args, "nearfield_depth", False)) and current_obs.get("has_nearfield_depth"):
                nearfield_started_at = time.perf_counter()
                nearfield_stats = mapper.update_nearfield_topdown(
                    current_obs["nearfield_depth"],
                    nearfield_intr,
                    pose_local,
                    current_obs["nearfield_camera_pose_world"],
                    radius_m=float(args.nearfield_radius_m),
                    ignore_radius_m=float(args.nearfield_ignore_radius_m),
                    depth_stride_px=int(args.nearfield_depth_stride_px),
                    floor_tolerance_m=float(args.nearfield_floor_tolerance_m),
                    obstacle_min_height_m=float(args.nearfield_obstacle_min_height_m),
                    obstacle_max_height_m=float(args.nearfield_obstacle_max_height_m),
                    splat_point_threshold=int(args.nearfield_splat_point_threshold),
                    free_splat_point_threshold=int(args.nearfield_free_splat_point_threshold),
                )
                record_mapping_timing("nearfield_update_ms", (time.perf_counter() - nearfield_started_at) * 1000.0)
                nearfield_stats["depth_source"] = str(current_obs.get("nearfield_depth_source", "unknown"))
                mapper.last_debug_stats["nearfield"] = nearfield_stats
            if bool(getattr(args, "static_nearfield_map", False)):
                static_nearfield_started_at = time.perf_counter()
                static_nearfield_stats = mapper.update_static_nearfield(
                    static_occupancy,
                    static_navigable,
                    static_map_info,
                    pose_local,
                    radius_m=float(args.static_nearfield_radius_m),
                    static_openings=static_openings,
                )
                record_mapping_timing("static_nearfield_update_ms", (time.perf_counter() - static_nearfield_started_at) * 1000.0)
                mapper.last_debug_stats["static_nearfield"] = static_nearfield_stats
            if bool(getattr(args, "mapping_debug", False)):
                stats = mapper.last_debug_stats
                rel = stats.get("rel_z_m_percentiles", {})
                bands = stats.get("image_bands", {})
                near = stats.get("nearfield", {})
                static_near = stats.get("static_nearfield", {})
                print(
                    "[mapping-debug] step=%s depth=%s sync=%s mode=%s valid=%s rays=%s skip_h=%s free_ray=%s occ_end=%s free_pts=%s obs_pts=%s "
                    "rel_z_p5/50/95=%s/%s/%s top_obs=%s mid_obs=%s bottom_obs=%s ceiling=%s negative=%s "
                    "nearfield=%s/%s/%s static_near=%s/%s/%s"
                    % (
                        "?" if step_idx is None else int(step_idx),
                        stats.get("depth_source", "unknown"),
                        stats.get("camera_frame_sync_updates", 0),
                        stats.get("mapping_mode", "unknown"),
                        stats.get("valid_points", 0),
                        stats.get("ray_count", 0),
                        stats.get("skipped_height_rays", 0),
                        stats.get("free_ray_cells", stats.get("free_splat_cells", 0)),
                        stats.get("occupied_endpoint_cells", stats.get("obstacle_splat_cells", 0)),
                        stats.get("free_band_points", 0),
                        stats.get("obstacle_band_points", 0),
                        rel.get("p5"),
                        rel.get("p50"),
                        rel.get("p95"),
                        (bands.get("top") or {}).get("obstacle_band_points", 0),
                        (bands.get("middle") or {}).get("obstacle_band_points", 0),
                        (bands.get("bottom") or {}).get("obstacle_band_points", 0),
                        stats.get("ceiling_like_points", 0),
                        stats.get("negative_height_points", 0),
                        near.get("reason", "off"),
                        near.get("free_splat_cells", 0),
                        near.get("obstacle_splat_cells", 0),
                        static_near.get("reason", "off"),
                        static_near.get("free_cells", 0),
                        static_near.get("occupied_cells", 0),
                    ),
                    flush=True,
                )
            dynamic_map_info = mapper.grid.map_info
            array_export_started_at = time.perf_counter()
            occupancy_local = mapper.grid.occupied.astype(bool)
            free_local = mapper.grid.free.astype(bool)
            observed_local = mapper.grid.observed.astype(bool)
            record_mapping_timing("array_export_ms", (time.perf_counter() - array_export_started_at) * 1000.0)
            traversible_started_at = time.perf_counter()
            navigable_local = mapper.traversible(unknown_is_obstacle=True)
            record_mapping_timing("traversible_ms", (time.perf_counter() - traversible_started_at) * 1000.0)
            planner_init_started_at = time.perf_counter()
            base_nav_planner_local = GridAStarPlanner(navigable_local, dynamic_map_info.resolution_m, allow_diagonal=True)
            record_mapping_timing("base_planner_init_ms", (time.perf_counter() - planner_init_started_at) * 1000.0)
            snap_started_at = time.perf_counter()
            current_grid_local = base_nav_planner_local.snap_to_free(
                world_xy_to_grid(float(pose_local[0]), float(pose_local[1]), dynamic_map_info)
            )
            record_mapping_timing("snap_to_free_ms", (time.perf_counter() - snap_started_at) * 1000.0)
            clearance_started_at = time.perf_counter()
            astar_navigable_local = apply_dynamic_astar_edge_clearance(
                navigable_local,
                occupancy_local,
                dynamic_map_info.resolution_m,
                float(args.runtime_planning_clearance_m),
                current_grid=current_grid_local,
            )
            record_mapping_timing("astar_clearance_ms", (time.perf_counter() - clearance_started_at) * 1000.0)
            planner_init_started_at = time.perf_counter()
            nav_planner_local = GridAStarPlanner(astar_navigable_local, dynamic_map_info.resolution_m, allow_diagonal=True)
            record_mapping_timing("planner_init_ms", (time.perf_counter() - planner_init_started_at) * 1000.0)
            if current_grid_local is None:
                recovery_started_at = time.perf_counter()
                mapper.update_simple_radius(pose_local, radius_m=max(float(args.robot_radius_m), float(args.online_resolution_m)))
                record_mapping_timing("simple_radius_recovery_ms", (time.perf_counter() - recovery_started_at) * 1000.0)
                array_export_started_at = time.perf_counter()
                occupancy_local = mapper.grid.occupied.astype(bool)
                free_local = mapper.grid.free.astype(bool)
                observed_local = mapper.grid.observed.astype(bool)
                record_mapping_timing("array_export_ms", (time.perf_counter() - array_export_started_at) * 1000.0)
                traversible_started_at = time.perf_counter()
                navigable_local = mapper.traversible(unknown_is_obstacle=True)
                record_mapping_timing("traversible_ms", (time.perf_counter() - traversible_started_at) * 1000.0)
                planner_init_started_at = time.perf_counter()
                base_nav_planner_local = GridAStarPlanner(navigable_local, dynamic_map_info.resolution_m, allow_diagonal=True)
                record_mapping_timing("base_planner_init_ms", (time.perf_counter() - planner_init_started_at) * 1000.0)
                snap_started_at = time.perf_counter()
                current_grid_local = base_nav_planner_local.snap_to_free(
                    world_xy_to_grid(float(pose_local[0]), float(pose_local[1]), dynamic_map_info)
                )
                record_mapping_timing("snap_to_free_ms", (time.perf_counter() - snap_started_at) * 1000.0)
                clearance_started_at = time.perf_counter()
                astar_navigable_local = apply_dynamic_astar_edge_clearance(
                    navigable_local,
                    occupancy_local,
                    dynamic_map_info.resolution_m,
                    float(args.runtime_planning_clearance_m),
                    current_grid=current_grid_local,
                )
                record_mapping_timing("astar_clearance_ms", (time.perf_counter() - clearance_started_at) * 1000.0)
                planner_init_started_at = time.perf_counter()
                nav_planner_local = GridAStarPlanner(astar_navigable_local, dynamic_map_info.resolution_m, allow_diagonal=True)
                record_mapping_timing("planner_init_ms", (time.perf_counter() - planner_init_started_at) * 1000.0)
            last_dynamic_occupancy = occupancy_local
            last_dynamic_free = free_local
            last_dynamic_navigable = navigable_local
            last_dynamic_astar_navigable = astar_navigable_local
            last_dynamic_observed = observed_local
            record_mapping_timing("state_total_ms", (time.perf_counter() - started_at) * 1000.0)
            record_latency("mapping", started_at)
            return {
                "pose": pose_local,
                "map_info": dynamic_map_info,
                "occupancy": occupancy_local,
                "free": free_local,
                "observed": observed_local,
                "navigable": navigable_local,
                "astar_navigable": astar_navigable_local,
                "base_nav_planner": base_nav_planner_local,
                "nav_planner": nav_planner_local,
                "current_grid": current_grid_local,
            }

        def run_detector_update(
            current_obs: dict,
            step_idx: int,
            map_info_local: MapInfo,
            occupancy_local: np.ndarray,
            navigable_local: np.ndarray,
        ) -> dict:
            nonlocal detector_cuda_rgb, logged_detector_rgb_device, segmenter, sam2_failure_logged, last_detections_2d
            if detector is None:
                last_detections_2d = []
                return current_obs
            started_at = time.perf_counter()
            obs_local = current_obs
            detector_rgb = obs_local["rgb"]
            used_cuda_rgb = False
            if detector_cuda_rgb:
                if not (
                    obs_local.get("has_rgb")
                    and obs_local.get("rgb_device") == "cuda"
                    and obs_local.get("rgb_gpu") is not None
                ):
                    obs_local = server.get_observation(read_rgb=True, read_depth=False, rgb_device="cuda")
                if obs_local.get("has_rgb") and obs_local.get("rgb_device") == "cuda" and obs_local.get("rgb_gpu") is not None:
                    detector_rgb = obs_local["rgb_gpu"]
                    used_cuda_rgb = True
                else:
                    detector_cuda_rgb = False
                    detector_rgb = obs_local["rgb"]
            if used_cuda_rgb:
                if not logged_detector_rgb_device:
                    print("[sgnav-loop] detector RGB input: Isaac CUDA annotator -> YOLO tensor", flush=True)
                    logged_detector_rgb_device = True
            elif detector_requires_rgb(detector, args.detector) and not logged_detector_rgb_device:
                print("[sgnav-loop] detector RGB input: CPU fallback", flush=True)
                logged_detector_rgb_device = True
            detections_2d = detector.detect(detector_rgb)
            if int(args.max_detections_per_frame) > 0:
                detections_2d = detections_2d[: int(args.max_detections_per_frame)]
            detections_2d = filter_edge_touching_detections(
                list(detections_2d),
                image_width=int(args.isaac_width),
                image_height=int(args.isaac_height),
                step_idx=int(step_idx),
                reject_edge_touching_bboxes=bool(args.reject_edge_touching_bboxes),
                margin_px=float(args.bbox_edge_margin_px),
                margin_ratio=float(args.bbox_edge_margin_ratio),
                min_confidence=float(args.detector_conf),
                raw_log=raw_detection_debug_log,
            )
            if segmenter is not None and detections_2d:
                try:
                    segment_rgb = obs_local["rgb"] if obs_local.get("rgb_device") == "cpu" else viz_rgb(obs_local)
                    detections_2d = segmenter.segment(segment_rgb, list(detections_2d))
                except Exception as exc:
                    if str(getattr(args, "segmenter", "none")).strip().lower() == "sam2":
                        raise
                    if not sam2_failure_logged:
                        print("[sam2] segmentation failed; continuing with detector boxes only: %s" % exc, flush=True)
                        sam2_failure_logged = True
                    segmenter = None
            last_detections_2d = list(detections_2d)
            for det in detections_2d:
                cat_norm = normalize_category(det.category)
                detection_category_counts[cat_norm] += 1
                if cat_norm == goal_norm:
                    goal_detection_history.append(summarize_detection(det, step_idx))
            if goal_detection_history and goal_detection_history[-1]["step"] == int(step_idx):
                recent = [row for row in goal_detection_history if row["step"] == int(step_idx)]
                print(
                    "[sgnav-loop] detector goal detections step=%s: %s"
                    % (
                        step_idx,
                        ", ".join(
                            "%s raw=%s conf=%.3f"
                            % (row["category"], row["raw_label"], row["confidence"])
                            for row in recent
                        ),
                    ),
                    flush=True,
                )
            if detection_localization == "depth":
                detections_3d = detections_to_3d(
                    detections_2d,
                    obs_local["depth"],
                    intr,
                    obs_local["camera_pose_world"],
                    depth_max_m=float(args.depth_max_m),
                    min_points=int(args.min_depth_points_per_detection),
                )
            elif detection_localization in {"static_map_ray", "map_ray", "rgb_map_ray"}:
                detections_3d = detections_to_3d_static_map_ray(
                    detections_2d,
                    obs_local["camera_pose_world"],
                    int(args.isaac_width),
                    float(args.camera_hfov_deg),
                    map_info_local,
                    occupancy_local,
                    navigable_local,
                    max_range_m=float(args.depth_max_m),
                )
            elif detection_localization == "none":
                detections_3d = []
            else:
                raise ValueError("Unsupported detection localization mode: %s" % detection_localization)
            if str(getattr(args, "sgnav_mode", "legacy")).strip().lower() == "paper" and detection_localization == "depth":
                fused_instances = fused_instance_registry.update(
                    detections_2d,
                    obs_local["depth"],
                    intr,
                    obs_local["camera_pose_world"],
                    step_id=step_idx,
                    depth_max_m=float(args.depth_max_m),
                    min_points=int(args.min_depth_points_per_detection),
                    stride=int(args.depth_stride_px),
                )
                object_memory.update_fused_instances(
                    [instance for instance in fused_instances if int(instance.last_seen_step) == int(step_idx)],
                    step_id=step_idx,
                    map_info=map_info_local,
                )
            else:
                object_memory.update(detections_3d, step_id=step_idx, map_info=map_info_local)
            if paper_mode:
                object_memory.dedupe_goal_candidates(
                    goal_category=episode["goal_category"],
                    merge_radius_m=max(float(args.object_merge_radius_m), 0.75),
                    map_info=map_info_local,
                )
            evaluator.num_detector_calls += 1
            evaluator.num_yolo_calls += 1
            record_latency("perception", started_at)
            return obs_local

        def update_scenegraph_frame(current_obs: dict, step_idx: int, map_state: dict) -> None:
            started_at = time.perf_counter()
            llm_requests_before = total_llm_requests()
            room_map = None if full_room_map is None else observed_room_map(full_room_map, map_state["observed"])
            rgb_for_graph = current_obs["rgb"] if current_obs.get("has_rgb") and current_obs.get("rgb_device") == "cpu" else None
            if rgb_for_graph is None and vllm_needs_cpu_rgb:
                rgb_for_graph = viz_rgb(current_obs)
            scenegraph.update_from_frame(
                object_memory,
                room_map=room_map,
                room_masks=last_room_masks,
                room_semantic_labels=room_semantic_labels,
                room_segmentation_debug=last_room_segmentation_debug,
                room_semantics_debug=last_room_semantics_debug,
                rgb=rgb_for_graph,
                depth=current_obs.get("depth"),
                detections_2d=last_detections_2d,
                map_info=map_state["map_info"],
                occupancy=map_state["occupancy"],
                free=map_state["free"],
                navigable=map_state["free"],
                observed=map_state["observed"],
                pose_world=map_state["pose"],
                camera_pose_world=current_obs.get("camera_pose_world"),
                step_id=step_idx,
            )
            setattr(scenegraph, "room_context_debug", dict(last_room_context_metadata))
            evaluator.num_scenegraph_updates += 1
            record_latency("graph", started_at)
            if total_llm_requests() > llm_requests_before:
                record_latency("llm", started_at)

        def save_roomseg_snapshot_for_frontier_scoring(
            step_idx: int,
            map_state: dict,
            *,
            frontier_map: np.ndarray | None = None,
            selected_frontier_members: Sequence[Sequence[int]] | None = None,
            selected_frontier_center_rc: Sequence[int] | None = None,
        ) -> dict | None:
            if not bool(getattr(args, "save_roomseg_snapshots", False)):
                return None
            return save_roomseg_layer_dump(
                out_dir=str(getattr(args, "roomseg_snapshot_dir", "result/roomseg_snapshots")),
                step=int(step_idx),
                room_debug=_roomseg_debug_for_layer_dump(last_room_segmentation_debug, room_segmenter),
                occupancy_map=map_state["occupancy"],
                observed_free_mask=map_state["free"],
                obstacle_mask=map_state["occupancy"],
                unknown_mask=~np.asarray(map_state["observed"], dtype=bool),
                frontier_map=frontier_map,
                selected_frontier_members=selected_frontier_members,
                selected_frontier_center_rc=selected_frontier_center_rc,
                agent_rc=map_state["current_grid"],
                max_saves=int(getattr(args, "roomseg_snapshot_max_saves", 500)),
                save_npz=True,
                save_png=False,
                save_summary_json=True,
                save_overlay_png=False,
                save_layers_png=False,
                save_navigation_room_masks_png=True,
                npz_keys=ROOMSEG_SNAPSHOT_ARRAY_KEYS,
                include_selected_frontier_sector=False,
            )

        def update_room_context_for_frontier_scoring(step_idx: int, map_state: dict) -> RoomContextResult:
            nonlocal last_room_masks, room_semantic_labels, last_room_segmentation_debug
            nonlocal last_room_semantics_debug, last_room_context_result, last_room_context_metadata
            result = prepare_room_context_for_frontier_scoring(
                step_idx=int(step_idx),
                mapper=mapper,
                object_memory=object_memory,
                room_segmenter=room_segmenter,
                room_labeler=room_labeler,
                map_info=map_state["map_info"],
                previous_room_context=room_context_cache,
                strict_benchmark=bool(getattr(args, "strict_benchmark", False)),
                occupancy=map_state["occupancy"],
                observed_free_mask=map_state["free"],
                obstacle_mask=map_state["occupancy"],
                unknown_mask=~np.asarray(map_state["observed"], dtype=bool),
                allowed_categories=getattr(args, "room_label_allowed_categories", DEFAULT_ROOM_CATEGORIES),
            )
            last_room_context_result = result
            last_room_masks = list(result.room_masks)
            room_semantic_labels = dict(result.room_semantic_labels)
            last_room_segmentation_debug = dict(result.room_segmentation_debug)
            last_room_semantics_debug = dict(result.room_semantics_debug)
            last_room_context_metadata = result.metadata(full_order=False)
            setattr(scenegraph, "room_context_debug", dict(last_room_context_metadata))
            if viz is not None:
                viz.set_room_context(last_room_masks, room_semantic_labels, last_room_segmentation_debug)
            return result

        def save_selected_roomseg_snapshot(
            step_idx: int,
            map_state: dict,
            frontier_layers: Mapping[str, np.ndarray],
            selected_frontier,
        ) -> None:
            nonlocal last_room_segmentation_debug, last_room_context_metadata
            selected_members = getattr(selected_frontier, "members", None) if selected_frontier is not None else None
            selected_center = getattr(selected_frontier, "center_grid", None) if selected_frontier is not None else None
            snapshot = save_roomseg_snapshot_for_frontier_scoring(
                step_idx,
                map_state,
                frontier_map=frontier_layers.get("frontier") if frontier_layers is not None else None,
                selected_frontier_members=selected_members,
                selected_frontier_center_rc=selected_center,
            )
            if snapshot is None:
                return
            snapshot_paths = dict(snapshot.get("paths", {}))
            last_room_segmentation_debug = {
                **dict(last_room_segmentation_debug),
                "roomseg_snapshot_paths": snapshot_paths,
                "roomseg_snapshot_summary": dict(snapshot.get("summary", {})),
            }
            last_room_context_metadata = {
                **dict(last_room_context_metadata),
                "roomseg_snapshot_paths": snapshot_paths,
            }
            setattr(scenegraph, "room_context_debug", dict(last_room_context_metadata))
            if viz is not None:
                viz.set_room_context(last_room_masks, room_semantic_labels, last_room_segmentation_debug)

        def update_roomseg_debug_only(step_idx: int, map_state: dict) -> None:
            nonlocal last_room_masks, room_semantic_labels, last_room_segmentation_debug
            nonlocal last_room_semantics_debug, last_room_context_metadata, last_room_context_result
            if room_segmenter is None:
                last_room_masks = []
                room_semantic_labels = {}
                last_room_context_result = None
                last_room_context_metadata = {
                    **room_context_not_invoked_metadata(),
                    "roomseg_debug_only": True,
                    "room_context_source": "roomseg_debug_only",
                    "room_update_invoked_for_frontier_scoring": False,
                    "room_segmentation_ran": False,
                    "room_labeling_ran": False,
                    "room_segmentation_called_for": "debug_only",
                    "room_vlm_called": False,
                    "scenegraph_updated_after_room_context": False,
                    "frontier_scoring_after_room_context": False,
                    "room_mask_count": 0,
                }
                last_room_segmentation_debug = {
                    "roomseg_debug_only": True,
                    "enabled": False,
                    "source": str(room_map_mode),
                    "algorithm": str(getattr(args, "room_segmentation_config", {}).get("algorithm", room_map_mode)),
                    "room_count": 0,
                    "rooms": [],
                    "reason": "room_segmenter_not_configured",
                }
                if viz is not None:
                    viz.set_room_context(last_room_masks, room_semantic_labels, last_room_segmentation_debug)
                return

            started_at = time.perf_counter()
            update_kwargs = {
                "step": int(step_idx),
                "object_memory": [],
            }
            vertical_profile = getattr(mapper, "vertical_profile", None)
            if vertical_profile is not None:
                update_kwargs["vertical_profile"] = vertical_profile
            roomseg_static_structural = getattr(mapper, "roomseg_static_structural_occupied", None)
            if roomseg_static_structural is not None:
                update_kwargs["roomseg_static_structural_occupied"] = roomseg_static_structural
            roomseg_ray_evidence = getattr(mapper, "roomseg_ray_evidence", None)
            if callable(roomseg_ray_evidence):
                update_kwargs["roomseg_ray_evidence"] = roomseg_ray_evidence()
            try:
                masks = room_segmenter.update(
                    map_state["occupancy"],
                    map_state["free"],
                    map_state["occupancy"],
                    ~np.asarray(map_state["observed"], dtype=bool),
                    **update_kwargs,
                )
            except TypeError:
                update_kwargs.pop("vertical_profile", None)
                update_kwargs.pop("roomseg_static_structural_occupied", None)
                update_kwargs.pop("roomseg_ray_evidence", None)
                masks = room_segmenter.update(
                    map_state["occupancy"],
                    map_state["free"],
                    map_state["occupancy"],
                    ~np.asarray(map_state["observed"], dtype=bool),
                    **update_kwargs,
                )
            last_room_masks = list(masks)
            room_semantic_labels = {}
            debug = dict(getattr(room_segmenter, "last_debug", {}) or {})
            if not debug:
                debug = room_segmentation_debug(
                    last_room_masks,
                    map_state["free"],
                    map_state["occupancy"],
                    ~np.asarray(map_state["observed"], dtype=bool),
                    step=int(step_idx),
                    config=None,
                    source=str(room_map_mode),
                )
            debug = {
                **debug,
                "roomseg_debug_only": True,
                "roomseg_debug_view": "vertical_free_room_domain",
                "source": str(debug.get("source") or room_map_mode),
                "algorithm": str(debug.get("algorithm") or getattr(args, "room_segmentation_config", {}).get("algorithm", room_map_mode)),
                "room_count": int(len(last_room_masks)),
                "room_vlm_called": False,
                "scenegraph_updated_after_room_context": False,
                "frontier_scoring_after_room_context": False,
                "room_segmentation_called_for": "debug_only",
                "room_segmentation_step_index": int(step_idx),
                "room_segmentation_runtime_ms": max(0.0, (time.perf_counter() - started_at) * 1000.0),
            }
            last_room_segmentation_debug = debug
            last_room_semantics_debug = {
                "backend": "disabled_roomseg_debug_only",
                "labels": [],
                "request_count": 0,
                "failure_count": 0,
            }
            last_room_context_result = None
            last_room_context_metadata = {
                **room_context_not_invoked_metadata(),
                "roomseg_debug_only": True,
                "room_context_source": "roomseg_debug_only",
                "room_update_invoked_for_frontier_scoring": False,
                "room_segmentation_ran": True,
                "room_labeling_ran": False,
                "room_context_cache_hit": False,
                "room_label_count": 0,
                "room_label_requests": 0,
                "room_label_cache_hits": 0,
                "room_call_order_trace": ["mapping", "room_segmentation_debug_only", "visualization"],
                "room_segmentation_called_for": "debug_only",
                "room_segmentation_algorithm": str(last_room_segmentation_debug.get("algorithm", "")),
                "room_segmentation_step_index": int(step_idx),
                "room_vlm_called": False,
                "scenegraph_updated_after_room_context": False,
                "frontier_scoring_after_room_context": False,
                "room_mask_count": int(len(last_room_masks)),
            }
            if bool(getattr(args, "debug_roomseg_layers", False)):
                dump = save_roomseg_layer_dump(
                    out_dir=str(getattr(args, "debug_roomseg_dir", "debug/roomseg_layers")),
                    step=int(step_idx),
                    room_debug=_roomseg_debug_for_layer_dump(last_room_segmentation_debug, room_segmenter),
                    occupancy_map=map_state["occupancy"],
                    observed_free_mask=map_state["free"],
                    obstacle_mask=map_state["occupancy"],
                    unknown_mask=~np.asarray(map_state["observed"], dtype=bool),
                    frontier_map=np.zeros_like(map_state["occupancy"], dtype=bool),
                    selected_frontier_members=None,
                    selected_frontier_center_rc=None,
                    agent_rc=map_state["current_grid"],
                    max_saves=int(getattr(args, "debug_roomseg_max_saves", 50)),
                    save_npz=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("save_npz", True)),
                    save_png=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("save_png", True)),
                    save_summary_json=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("save_summary_json", True)),
                    include_selected_frontier_sector=False,
                )
                last_room_segmentation_debug = {
                    **dict(last_room_segmentation_debug),
                    "roomseg_debug_layers": dict(dump.get("paths", {})),
                    "roomseg_debug_summary": dict(dump.get("summary", {})),
                }
                last_room_context_metadata = {
                    **dict(last_room_context_metadata),
                    "roomseg_debug_layers": dict(dump.get("paths", {})),
                    "roomseg_debug_likely_cause": dict(dump.get("summary", {})).get("likely_cause"),
                }
            if viz is not None:
                viz.set_room_context(last_room_masks, room_semantic_labels, last_room_segmentation_debug)

        panorama_steps = max(0, int(getattr(args, "panorama_steps", 0)))
        if panorama_steps > 0:
            print("[sgnav-loop] opening panorama: %d RGB-D views" % panorama_steps, flush=True)
        for pano_idx in range(panorama_steps):
            map_state = update_mapper_state(obs, -panorama_steps + pano_idx)
            if map_state is None or map_state["current_grid"] is None:
                failure_reason = "panorama_agent_off_navigable_map"
                break
            pano_step = -panorama_steps + pano_idx
            if roomseg_debug_only:
                update_roomseg_debug_only(pano_step, map_state)
                panorama_frames += 1
                if viz is not None:
                    viz.update(
                        step=pano_step,
                        rgb=viz_rgb(obs),
                        detections_2d=[],
                        occupancy=map_state["occupancy"],
                        navigable=map_state["free"],
                        observed=map_state["observed"],
                        goal_cells=goal_cells,
                        current_grid=map_state["current_grid"],
                        pose=map_state["pose"],
                        frontiers=[],
                        nav_decision=None,
                        current_path=[],
                        full_path=[],
                        object_memory=object_memory,
                        goal_category=episode["goal_category"],
                        distance_to_goal=evaluator.final_distance_to_goal,
                        path_length=float(evaluator.path_accum.total_m),
                        scenegraph_backend="roomseg_debug_only",
                        score_debug={},
                        failure_reason=failure_reason,
                    )
                if pano_idx + 1 < panorama_steps:
                    yaw_delta = (2.0 * math.pi) / float(panorama_steps)
                    pano_wz = max(1e-3, abs(float(args.panorama_wz_radps)))
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        pano_wz,
                        dt=yaw_delta / pano_wz,
                        render_updates=int(args.panorama_render_updates_per_step),
                        read_rgb=viz_requested,
                        read_depth=True,
                        rgb_device="cpu",
                    )
                continue
            obs = run_detector_update(obs, -panorama_steps + pano_idx, map_state["map_info"], map_state["occupancy"], map_state["navigable"])
            update_scenegraph_frame(obs, -panorama_steps + pano_idx, map_state)
            panorama_frames += 1
            has_goal_detection = any(row["step"] == int(pano_step) for row in goal_detection_history)
            if viz is not None and (has_goal_detection or pano_idx + 1 == panorama_steps):
                viz.update(
                    step=pano_step,
                    rgb=viz_rgb(obs),
                    detections_2d=last_detections_2d,
                    occupancy=map_state["occupancy"],
                    navigable=map_state["free"],
                    observed=map_state["observed"],
                    goal_cells=goal_cells,
                    current_grid=map_state["current_grid"],
                    pose=map_state["pose"],
                    frontiers=last_frontiers,
                    nav_decision=last_nav_decision,
                    current_path=current_path,
                    full_path=full_path,
                    object_memory=object_memory,
                    goal_category=episode["goal_category"],
                    distance_to_goal=evaluator.final_distance_to_goal,
                    path_length=float(evaluator.path_accum.total_m),
                    scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                    score_debug=scenegraph.last_score_debug,
                    failure_reason=failure_reason,
                )
            if pano_idx + 1 < panorama_steps:
                yaw_delta = (2.0 * math.pi) / float(panorama_steps)
                pano_wz = max(1e-3, abs(float(args.panorama_wz_radps)))
                obs = server.step_kinematic_velocity(
                    0.0,
                    0.0,
                    pano_wz,
                    dt=yaw_delta / pano_wz,
                    render_updates=int(args.panorama_render_updates_per_step),
                    read_rgb=detector_requires_rgb(detector, args.detector),
                    read_depth=True,
                    rgb_device="cuda" if detector_cuda_rgb else "cpu",
                )
        for step in range(0 if failure_reason is not None else max_steps):
            map_state = update_mapper_state(obs, step)
            if map_state is None:
                failure_reason = "depth_unavailable_for_online_mapping"
                break
            pose = map_state["pose"]
            dynamic_map_info = map_state["map_info"]
            occupancy = map_state["occupancy"]
            free = map_state["free"]
            observed = map_state["observed"]
            navigable = map_state["navigable"]
            nav_planner = map_state["nav_planner"]
            base_nav_planner = map_state.get("base_nav_planner", nav_planner)
            current_grid = map_state["current_grid"]
            if current_grid is None:
                failure_reason = "agent_off_navigable_map"
                break
            metric_grid = metric_planner.snap_to_free(world_xy_to_grid(float(pose[0]), float(pose[1]), static_map_info))
            if metric_grid is None:
                failure_reason = "agent_off_static_metric_map"
                break
            evaluator.update_pose(pose, metric_grid, collided=bool(obs.get("collided", False)))
            if roomseg_debug_only:
                update_roomseg_debug_only(step, map_state)
                if viz is not None and step % viz_every == 0:
                    viz.update(
                        step=step,
                        rgb=viz_rgb(obs),
                        detections_2d=[],
                        occupancy=occupancy,
                        navigable=free,
                        observed=observed,
                        goal_cells=goal_cells,
                        current_grid=current_grid,
                        pose=pose,
                        frontiers=[],
                        nav_decision=None,
                        current_path=[],
                        full_path=[],
                        object_memory=object_memory,
                        goal_category=episode["goal_category"],
                        distance_to_goal=evaluator.final_distance_to_goal,
                        path_length=float(evaluator.path_accum.total_m),
                        scenegraph_backend="roomseg_debug_only",
                        score_debug={},
                        failure_reason=failure_reason,
                    )
                if step + 1 < max_steps:
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        float(args.panorama_wz_radps),
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=viz_requested and (step + 1) % viz_every == 0,
                        read_depth=True,
                        rgb_device="cpu",
                    )
                    continue
                break
            if evaluator.final_distance_to_goal <= success_distance:
                gt_success_region_reached = True
                if explore_until_no_frontiers:
                    gt_success_ignored_steps += 1
            if success_region_can_finish(
                evaluator.final_distance_to_goal,
                success_distance,
                require_sgnav_stop=bool(args.require_sgnav_stop),
                policy_stop_confirmed=False,
                ignore_goal_success=explore_until_no_frontiers,
            ):
                stop_called = True
                break
            if evaluator.final_distance_to_goal <= success_distance and bool(args.require_sgnav_stop):
                gt_success_without_sgnav_stop_steps += 1
                stop_blocked_reason = "sgnav_stop_required"
                if not logged_gt_success_without_sgnav_stop:
                    print(
                        "[sgnav-loop] inside GT success radius, but strict SG-Nav STOP is not confirmed; continuing perception/replanning",
                        flush=True,
                    )
                    logged_gt_success_without_sgnav_stop = True

            perception_due = detector is not None and (step % perception_every == 0 or force_perception_step)
            if perception_due:
                obs = run_detector_update(obs, step, dynamic_map_info, occupancy, navigable)
                force_perception_step = False
            if step % perception_every == 0 or perception_due:
                update_scenegraph_frame(obs, step, map_state)

            needs_replan = not current_path or step % replan_every == 0
            if current_path:
                suffix = trim_path_to_nearest(current_path, current_grid) if paper_mode else trim_path_to_current(current_path, current_grid)
                if suffix:
                    current_path = suffix
                else:
                    needs_replan = True

            if needs_replan:
                planning_started_at = time.perf_counter()
                llm_requests_before = total_llm_requests()
                astar_traversible = np.asarray(map_state.get("astar_navigable", navigable), dtype=bool)
                frontier_traversible = navigable.astype(bool)
                frontier_free = free.astype(bool) & frontier_traversible
                distance_traversible = frontier_traversible.copy()
                rr, cc = int(current_grid[0]), int(current_grid[1])
                if 0 <= rr < distance_traversible.shape[0] and 0 <= cc < distance_traversible.shape[1]:
                    distance_traversible[rr, cc] = True
                static_only_nearfield = getattr(mapper, "static_nearfield_mask", None)
                depth_free_mask = getattr(mapper, "depth_free_mask", None)
                if bool(getattr(args, "static_nearfield_map", False)) and static_only_nearfield is not None and depth_free_mask is not None:
                    static_only_nearfield = np.asarray(static_only_nearfield).astype(bool) & ~np.asarray(depth_free_mask).astype(bool)
                else:
                    static_only_nearfield = None
                frontier_layers = frontier_debug_layers(
                    frontier_free,
                    observed=observed,
                    occupancy=occupancy,
                    obstacle_dilation_radius_cells=int(args.frontier_obstacle_dilation_radius_cells),
                    unknown_dilation_radius_cells=int(args.frontier_unknown_dilation_radius_cells),
                    exclude_mask=static_only_nearfield,
                    unknown_source=str(args.frontier_unknown_source),
                )
                if bool(getattr(args, "frontier_debug_dump", False)):
                    assert frontier_free.shape == observed.shape == occupancy.shape == distance_traversible.shape
                    assert int(np.count_nonzero(frontier_layers["frontier"] & ~frontier_free)) == 0
                frontiers = list(last_frontiers)
                locked_goal_used = False
                candidate_override = None
                if paper_mode and long_term_goal.exists() and long_term_goal.mode == "frontier":
                    candidate_override = decision_policy.choose_navigation_target(
                        object_memory,
                        episode["goal_category"],
                        current_grid,
                        [],
                        nav_planner,
                        dynamic_map_info,
                        pose,
                        allow_frontier=False,
                        current_step=step,
                    )
                    if candidate_override.mode not in {"candidate", "reperception", "stop"}:
                        candidate_override = None
                if paper_mode and long_term_goal.exists() and candidate_override is None:
                    nav_decision = long_term_goal.to_navigation_decision()
                    locked_goal_used = True
                    if nav_decision.mode == "frontier":
                        last_frontier_commitment_reason = "continue_committed_frontier"
                        last_frontier_commitment_metadata = {
                            **dict(last_frontier_commitment_metadata),
                            "frontier_commitment_reason": "continue_committed_frontier",
                            "long_term_goal_locked": True,
                            "long_term_goal_selected_step": int(long_term_goal.selected_step),
                        }
                else:
                    if candidate_override is not None:
                        long_term_goal.clear("candidate_override")
                    last_frontier_raw_cells = int(np.count_nonzero(frontier_layers["frontier"]))
                    frontiers = extract_frontiers(
                        free=frontier_free,
                        observed=observed,
                        traversible=distance_traversible,
                        map_info=dynamic_map_info,
                        agent_grid=current_grid,
                        min_cluster_size=int(args.frontier_min_cluster_size),
                        min_distance_m=float(args.frontier_min_distance_m),
                        max_count=int(args.frontier_max_count),
                        occupancy=occupancy,
                        obstacle_dilation_radius_cells=int(args.frontier_obstacle_dilation_radius_cells),
                        unknown_dilation_radius_cells=int(args.frontier_unknown_dilation_radius_cells),
                        exclude_mask=static_only_nearfield,
                        unknown_source=str(args.frontier_unknown_source),
                        cluster_distance_mode=str(args.frontier_cluster_distance_mode),
                        allow_near_frontier_fallback=bool(args.frontier_allow_near_fallback),
                        require_reachable=not bool(frontier_mask_probe or explore_until_no_frontiers),
                    )
                    last_frontier_clusters = len(frontiers)
                    last_frontiers = list(frontiers)
                    room_context_result = None
                    candidate_preview = None
                    if paper_mode and frontiers and candidate_override is None and not frontier_mask_probe:
                        candidate_preview = decision_policy.select_goal_candidate(
                            object_memory,
                            episode["goal_category"],
                            pose,
                            current_step=step,
                        )
                        if bool(args.score_frontiers_before_candidate) or candidate_preview is None:
                            room_context_result = update_room_context_for_frontier_scoring(step, map_state)
                            update_scenegraph_frame(obs, step, map_state)
                    nav_decision = candidate_override or decision_policy.choose_navigation_target(
                        object_memory,
                        episode["goal_category"],
                        current_grid,
                        frontiers,
                        base_nav_planner if bool(frontier_mask_probe or explore_until_no_frontiers) else nav_planner,
                        dynamic_map_info,
                        pose,
                        allow_frontier=True,
                        current_step=step,
                    )
                    if nav_decision.mode == "none" and nav_decision.reason == "no_frontiers" and last_frontier_raw_cells > 0:
                        nav_decision.reason = "no_selectable_frontiers"
                        nav_decision.metadata = {
                            **dict(nav_decision.metadata or {}),
                            "frontier_selectable_failure_reason": "raw_frontiers_filtered_out",
                            "frontier_raw_cells": int(last_frontier_raw_cells),
                            "frontier_clusters": int(last_frontier_clusters),
                            "frontier_min_distance_m": float(args.frontier_min_distance_m),
                            "frontier_allow_near_fallback": bool(args.frontier_allow_near_fallback),
                            "frontier_cluster_distance_mode": str(args.frontier_cluster_distance_mode),
                        }
                    if room_context_result is not None:
                        last_room_context_metadata = room_context_result.metadata(full_order=True)
                        frontier_room_contexts = frontier_room_contexts_for_debug(
                            frontiers=frontiers,
                            room_debug=last_room_segmentation_debug,
                            room_masks=last_room_masks,
                            room_semantic_labels=room_semantic_labels,
                            observed_free=free,
                            unknown=~np.asarray(observed, dtype=bool),
                            agent_grid=current_grid,
                            resolution_m=float(dynamic_map_info.resolution_m),
                            config=dict(getattr(args, "room_segmentation_config", {}).get("frontier_room_context", {}) or {}),
                        )
                        selected_frontier = (
                            nav_decision.frontier_decision.selected_frontier
                            if nav_decision.frontier_decision is not None
                            else None
                        )
                        selected_index = (
                            nav_decision.frontier_decision.selected_index
                            if nav_decision.frontier_decision is not None
                            else None
                        )
                        selected_room_context = None
                        if selected_index is not None:
                            for item in frontier_room_contexts:
                                if int(item.get("frontier_id", -1)) == int(selected_index):
                                    selected_room_context = dict(item.get("room_context") or {})
                                    break
                        last_room_context_metadata = {
                            **dict(last_room_context_metadata),
                            "frontier_room_contexts": frontier_room_contexts,
                            "selected_frontier_room_context": selected_room_context,
                        }
                        setattr(scenegraph, "room_context_debug", dict(last_room_context_metadata))
                        nav_decision.metadata = {
                            **dict(nav_decision.metadata or {}),
                            **dict(last_room_context_metadata),
                        }
                        save_selected_roomseg_snapshot(
                            step,
                            map_state,
                            frontier_layers,
                            selected_frontier,
                        )
                        if bool(getattr(args, "debug_roomseg_layers", False)):
                            selected_members = getattr(selected_frontier, "members", None) if selected_frontier is not None else None
                            selected_center = getattr(selected_frontier, "center_grid", None) if selected_frontier is not None else None
                            dump = save_roomseg_layer_dump(
                                out_dir=str(getattr(args, "debug_roomseg_dir", "debug/roomseg_layers")),
                                step=int(step),
                                room_debug=_roomseg_debug_for_layer_dump(last_room_segmentation_debug, room_segmenter),
                                occupancy_map=occupancy,
                                observed_free_mask=free,
                                obstacle_mask=occupancy,
                                unknown_mask=~np.asarray(observed, dtype=bool),
                                frontier_map=frontier_layers["frontier"],
                                selected_frontier_members=selected_members,
                                selected_frontier_center_rc=selected_center,
                                agent_rc=current_grid,
                                max_saves=int(getattr(args, "debug_roomseg_max_saves", 50)),
                                save_npz=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("save_npz", True)),
                                save_png=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("save_png", True)),
                                save_summary_json=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("save_summary_json", True)),
                                include_selected_frontier_sector=bool(dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {}).get("include_selected_frontier_sector", True)),
                            )
                            last_room_segmentation_debug = {
                                **dict(last_room_segmentation_debug),
                                "roomseg_debug_layers": dict(dump.get("paths", {})),
                                "roomseg_debug_summary": dict(dump.get("summary", {})),
                            }
                            last_room_context_metadata["roomseg_debug_layers"] = dict(dump.get("paths", {}))
                            last_room_context_metadata["roomseg_debug_likely_cause"] = dict(dump.get("summary", {})).get("likely_cause")
                            nav_decision.metadata = {
                                **dict(nav_decision.metadata or {}),
                                "roomseg_debug_layers": dict(dump.get("paths", {})),
                                "roomseg_debug_likely_cause": dict(dump.get("summary", {})).get("likely_cause"),
                            }
                            setattr(scenegraph, "room_context_debug", dict(last_room_context_metadata))
                            if viz is not None:
                                viz.set_room_context(last_room_masks, room_semantic_labels, last_room_segmentation_debug)
                if frontier_commitment is not None and nav_decision.mode == "frontier" and not locked_goal_used:
                    frontier_decision = nav_decision.frontier_decision
                    proposed_frontier = frontier_decision.selected_frontier if frontier_decision is not None else None
                    proposed_score = 0.0
                    scores_by_index = []
                    if frontier_decision is not None:
                        scores_by_index = list(frontier_decision.total_scores)
                        if frontier_decision.selected_index is not None and frontier_decision.selected_index < len(scores_by_index):
                            proposed_score = float(scores_by_index[int(frontier_decision.selected_index)])
                    commit_decision = frontier_commitment.select(
                        frontiers,
                        proposed_frontier,
                        proposed_score,
                        current_grid,
                        step,
                        planner=None if bool(frontier_mask_probe or explore_until_no_frontiers) else nav_planner,
                        target_cells=nav_decision.target_cells,
                        scores_by_index=scores_by_index,
                    )
                    last_frontier_commitment_metadata = dict(commit_decision.metadata)
                    last_frontier_commitment_metadata["frontier_commitment_reason"] = commit_decision.reason
                    last_frontier_commitment_reason = commit_decision.reason
                    nav_decision.target_cells = list(commit_decision.target_cells)
                    nav_decision.reason = commit_decision.reason
                    if frontier_decision is not None:
                        frontier_decision.selected_frontier = commit_decision.frontier
                        if commit_decision.frontier is not None:
                            try:
                                frontier_decision.selected_index = frontiers.index(commit_decision.frontier)
                            except ValueError:
                                frontier_decision.selected_index = None
                        else:
                            frontier_decision.selected_index = None
                    nav_decision.metadata = {
                        **dict(nav_decision.metadata or {}),
                        "frontier_commitment": last_frontier_commitment_metadata,
                    }
                if nav_decision.mode == "frontier":
                    nav_meta = dict(nav_decision.metadata or {})
                    frontier_target_mode = nav_meta.get("frontier_target_mode")
                    frontier_center_grid = nav_meta.get("frontier_center_grid")
                    frontier_actual_target_grid = nav_meta.get("frontier_actual_target_grid")
                    frontier_unreachable_recovery = bool(nav_meta.get("frontier_unreachable_recovery", False))
                    frontier_unreachable_reason = nav_meta.get("frontier_unreachable_reason")
                    if nav_decision.frontier_decision is not None and not nav_decision.target_cells:
                        selected_frontier = nav_decision.frontier_decision.selected_frontier
                        if frontier_commitment is not None and selected_frontier is not None:
                            frontier_commitment.blacklist_frontier(
                                selected_frontier,
                                step,
                                str(frontier_unreachable_reason or "frontier_unreachable"),
                            )
                        frontier_blacklisted = True
                        long_term_goal.invalidate(str(frontier_unreachable_reason or "frontier_unreachable"))
                        current_path = []
                        force_perception_step = True
                        last_frontier_commitment_reason = str(frontier_unreachable_reason or "frontier_unreachable")
                        last_frontier_commitment_metadata = {
                            **dict(last_frontier_commitment_metadata),
                            "frontier_commitment_reason": last_frontier_commitment_reason,
                            "frontier_blacklisted": True,
                        }
                        continue
                if paper_mode and not locked_goal_used and nav_decision.mode in {"frontier", "candidate"} and nav_decision.target_cells:
                    long_term_goal.set_from(nav_decision, step)
                last_nav_decision = nav_decision
                nav_target_key = (
                    str(nav_decision.mode),
                    tuple(int(v) for v in nav_decision.target_cells[0]) if nav_decision.target_cells else None,
                    int(len(nav_decision.target_cells or [])),
                    last_frontier_commitment_metadata.get("active_frontier_id"),
                )
                if nav_target_key != nav_execution_progress_key:
                    nav_execution_progress_key = nav_target_key
                    nav_execution_best_distance_m = float("inf")
                    nav_execution_no_progress_steps = 0
                if not locked_goal_used:
                    evaluator.num_frontier_decisions += 1
                last_decision_mode = nav_decision.mode
                last_decision_reason = nav_decision.reason
                if bool(getattr(args, "debug_graph_dump", False)):
                    save_graph_debug_dump(
                        args.debug_graph_dump_dir,
                        step=step,
                        goal=episode["goal_category"],
                        scenegraph=scenegraph,
                        frontiers=frontiers,
                        frontier_decision=nav_decision.frontier_decision,
                        nav_decision=nav_decision,
                        commitment_metadata=last_frontier_commitment_metadata,
                    )
                if bool(getattr(args, "frontier_debug_dump", False)):
                    selected_frontier_cell = None
                    if nav_decision.frontier_decision is not None and nav_decision.frontier_decision.selected_frontier is not None:
                        selected_frontier_cell = nav_decision.frontier_decision.selected_frontier.center_grid
                    save_frontier_debug_snapshot(
                        args.frontier_debug_dir,
                        step,
                        free=frontier_free,
                        occupied=occupancy,
                        observed=observed,
                        unknown=frontier_layers["unknown"],
                        unknown_dilated=frontier_layers["unknown_dilated"],
                        frontier=frontier_layers["frontier"],
                        traversible=distance_traversible,
                        dist_map=astar_distance_map(
                            distance_traversible,
                            current_grid,
                            dynamic_map_info.resolution_m,
                            allow_diagonal=True,
                        ),
                        agent_grid=current_grid,
                        clusters=frontiers,
                        selected_frontier=selected_frontier_cell,
                        candidate_centers=candidate_center_payload(
                            object_memory,
                            episode["goal_category"],
                            selected_candidate_id=(
                                int(nav_decision.selected_candidate.node_id)
                                if nav_decision.selected_candidate is not None
                                else None
                            ),
                        ),
                        candidate_target_cells=nav_decision.target_cells if nav_decision.mode == "candidate" else [],
                        selected_candidate=(
                            nav_decision.selected_candidate.center_grid
                            if nav_decision.selected_candidate is not None
                            else None
                        ),
                        decision_mode=nav_decision.mode,
                    )
                if nav_decision.selected_candidate is not None:
                    last_selected_candidate = nav_decision.selected_candidate
                    candidate_id = int(last_selected_candidate.node_id)
                    if logged_selected_candidate_id != candidate_id:
                        logged_selected_candidate_id = candidate_id
                        print(
                            "[sgnav-loop] selected candidate id=%d category=%s raw=%s conf=%.3f observed=%d center=%s"
                            % (
                                candidate_id,
                                normalize_category(last_selected_candidate.category),
                                str(last_selected_candidate.raw_label),
                                float(last_selected_candidate.confidence),
                                int(last_selected_candidate.observed_count),
                                tuple(float(v) for v in last_selected_candidate.center_world),
                            ),
                            flush=True,
                        )
                goal_candidate_count = len(
                    [
                        node
                        for node in object_memory.nodes
                        if category_matches_goal(node.category)
                    ]
                )
                if nav_decision.stop:
                    policy_stop_confirmed = True
                    stop_blocked_reason = None
                    if evaluator.final_distance_to_goal <= success_distance:
                        gt_success_region_reached = True
                    if success_region_can_finish(
                        evaluator.final_distance_to_goal,
                        success_distance,
                        require_sgnav_stop=bool(args.require_sgnav_stop),
                        policy_stop_confirmed=True,
                        ignore_goal_success=explore_until_no_frontiers,
                    ):
                        stop_called = True
                    else:
                        failure_reason = (
                            "goal_success_ignored_for_frontier_exploration"
                            if explore_until_no_frontiers
                            else "sgnav_stop_outside_goal_region"
                        )
                    if viz is not None:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=free,
                            observed=observed,
                            goal_cells=goal_cells,
                            current_grid=current_grid,
                            pose=pose,
                            frontiers=last_frontiers,
                            nav_decision=last_nav_decision,
                            current_path=current_path,
                            full_path=full_path,
                            object_memory=object_memory,
                            goal_category=episode["goal_category"],
                            distance_to_goal=evaluator.final_distance_to_goal,
                            path_length=float(evaluator.path_accum.total_m),
                            scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                            score_debug=scenegraph.last_score_debug,
                            failure_reason=failure_reason,
                        )
                    break
                if nav_decision.mode == "reperception":
                    current_path = []
                    full_path.append(tuple(int(v) for v in current_grid))
                    if viz is not None and step % viz_every == 0:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=free,
                            observed=observed,
                            goal_cells=goal_cells,
                            current_grid=current_grid,
                            pose=pose,
                            frontiers=last_frontiers,
                            nav_decision=last_nav_decision,
                            current_path=current_path,
                            full_path=full_path,
                            object_memory=object_memory,
                            goal_category=episode["goal_category"],
                            distance_to_goal=evaluator.final_distance_to_goal,
                            path_length=float(evaluator.path_accum.total_m),
                            scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                            score_debug=scenegraph.last_score_debug,
                        )
                    force_perception_step = True
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        float(args.reperception_turn_wz_radps),
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=detector_requires_rgb(detector, args.detector),
                        read_depth=True,
                        rgb_device="cuda" if detector_cuda_rgb else "cpu",
                    )
                    continue
                nav_goals = nav_decision.target_cells
                if paper_mode and nav_decision.mode == "frontier" and not nav_goals:
                    reason = str(
                        (nav_decision.metadata or {}).get("frontier_unreachable_reason")
                        or nav_decision.reason
                        or "frontier_empty_target"
                    )
                    if frontier_commitment is not None:
                        frontier_commitment.invalidate_active(step, reason, blacklist=True)
                    long_term_goal.invalidate(reason)
                    current_path = []
                    force_perception_step = True
                    frontier_unreachable_recovery = True
                    frontier_unreachable_reason = reason
                    frontier_blacklisted = True
                    frontier_stop_at_current_grid = [int(current_grid[0]), int(current_grid[1])]
                    last_frontier_commitment_reason = "%s_blacklisted" % reason
                    last_frontier_commitment_metadata = {
                        **dict(last_frontier_commitment_metadata),
                        "frontier_commitment_reason": last_frontier_commitment_reason,
                        "frontier_blacklisted": True,
                        "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
                    }
                    continue
                if not nav_goals and args.allow_gt_goal_fallback:
                    nav_goals = goal_cells
                    last_decision_mode = "gt_goal_fallback"
                    last_decision_reason = "explicit_gt_goal_fallback"
                if not nav_goals:
                    failure_reason = nav_decision.reason or "sgnav_no_target"
                    if viz is not None:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=free,
                            observed=observed,
                            goal_cells=goal_cells,
                            current_grid=current_grid,
                            pose=pose,
                            frontiers=last_frontiers,
                            nav_decision=last_nav_decision,
                            current_path=current_path,
                            full_path=full_path,
                            object_memory=object_memory,
                            goal_category=episode["goal_category"],
                            distance_to_goal=evaluator.final_distance_to_goal,
                            path_length=float(evaluator.path_accum.total_m),
                            scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                            score_debug=scenegraph.last_score_debug,
                            failure_reason=failure_reason,
                    )
                    break
                planning_goals = nav_goals
                if nav_decision.mode == "frontier":
                    radius_goals = planning_target_cells_within_radius(
                        nav_goals,
                        astar_traversible,
                        dynamic_map_info.resolution_m,
                        float(args.frontier_commit_reached_radius_m),
                    )
                    if radius_goals:
                        planning_goals = radius_goals
                        nav_decision.metadata = {
                            **dict(nav_decision.metadata or {}),
                            "frontier_planning_goal_mode": "within_target_radius",
                            "frontier_planning_goal_cells_count": int(len(radius_goals)),
                            "frontier_planning_goal_radius_m": float(args.frontier_commit_reached_radius_m),
                        }
                result = nav_planner.plan(current_grid, planning_goals)
                if not result.path:
                    if paper_mode and nav_decision.mode == "frontier":
                        reason = "frontier_center_unreachable"
                        long_term_goal.invalidate(reason)
                        if frontier_commitment is not None:
                            frontier_commitment.invalidate_active(step, reason, blacklist=True)
                        current_path = []
                        force_perception_step = True
                        frontier_unreachable_recovery = True
                        frontier_unreachable_reason = reason
                        frontier_blacklisted = True
                        frontier_stop_at_current_grid = [int(current_grid[0]), int(current_grid[1])]
                        last_frontier_commitment_reason = "%s_blacklisted" % reason
                        last_frontier_commitment_metadata = {
                            **dict(last_frontier_commitment_metadata),
                            "frontier_commitment_reason": last_frontier_commitment_reason,
                            "frontier_blacklisted": True,
                            "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
                        }
                        continue
                    failure_reason = "astar_no_path"
                    if viz is not None:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=free,
                            observed=observed,
                            goal_cells=goal_cells,
                            current_grid=current_grid,
                            pose=pose,
                            frontiers=last_frontiers,
                            nav_decision=last_nav_decision,
                            current_path=current_path,
                            full_path=full_path,
                            object_memory=object_memory,
                            goal_category=episode["goal_category"],
                            distance_to_goal=evaluator.final_distance_to_goal,
                            path_length=float(evaluator.path_accum.total_m),
                            scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                            score_debug=scenegraph.last_score_debug,
                            failure_reason=failure_reason,
                        )
                    break
                if frontier_mask_probe and nav_decision.mode == "frontier" and len(result.path) <= 1:
                    reason = "frontier_trivial_path"
                    selected_frontier = (
                        nav_decision.frontier_decision.selected_frontier
                        if nav_decision.frontier_decision is not None
                        else None
                    )
                    if frontier_commitment is not None:
                        if selected_frontier is not None:
                            frontier_commitment.blacklist_frontier(selected_frontier, step, reason)
                        frontier_commitment.invalidate_active(step, reason, blacklist=True)
                    long_term_goal.invalidate(reason)
                    current_path = []
                    force_perception_step = True
                    frontier_unreachable_recovery = True
                    frontier_unreachable_reason = reason
                    frontier_blacklisted = True
                    frontier_stop_at_current_grid = [int(current_grid[0]), int(current_grid[1])]
                    last_frontier_commitment_reason = "%s_blacklisted" % reason
                    last_frontier_commitment_metadata = {
                        **dict(last_frontier_commitment_metadata),
                        "frontier_commitment_reason": last_frontier_commitment_reason,
                        "frontier_blacklisted": True,
                        "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
                    }
                    continue
                if (
                    frontier_mask_probe
                    and nav_decision.mode == "frontier"
                    and nav_decision.frontier_decision is not None
                    and nav_decision.frontier_decision.selected_frontier is not None
                ):
                    selected_frontier = nav_decision.frontier_decision.selected_frontier
                    selected_center = tuple(int(v) for v in selected_frontier.center_grid)
                    selected_target = tuple(int(v) for v in (nav_decision.target_cells[0] if nav_decision.target_cells else selected_center))
                    snapshot_key = (selected_center, selected_target)
                    if snapshot_key != last_roomseg_snapshot_frontier_key:
                        update_room_context_for_frontier_scoring(step, map_state)
                        save_selected_roomseg_snapshot(
                            step,
                            map_state,
                            frontier_layers,
                            selected_frontier,
                        )
                        last_roomseg_snapshot_frontier_key = snapshot_key
                current_path = result.path
                full_path.extend(result.path)
                record_latency("planning", planning_started_at)
                if total_llm_requests() > llm_requests_before:
                    record_latency("llm", planning_started_at)

            if viz is not None and step % viz_every == 0:
                viz.update(
                    step=step,
                    rgb=viz_rgb(obs),
                    detections_2d=last_detections_2d,
                    occupancy=occupancy,
                    navigable=free,
                    observed=observed,
                    goal_cells=goal_cells,
                    current_grid=current_grid,
                    pose=pose,
                    frontiers=last_frontiers,
                    nav_decision=last_nav_decision,
                    current_path=current_path,
                    full_path=full_path,
                    object_memory=object_memory,
                    goal_category=episode["goal_category"],
                    distance_to_goal=evaluator.final_distance_to_goal,
                    path_length=float(evaluator.path_accum.total_m),
                    scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                    score_debug=scenegraph.last_score_debug,
                )

            target_reached, target_distance_m = navigation_target_reached(
                current_grid,
                last_nav_decision,
                dynamic_map_info.resolution_m,
                float(args.frontier_commit_reached_radius_m),
            )
            if target_reached:
                reason = "target_radius_reached"
                target_radius_reached_steps += 1
                if frontier_commitment is not None and last_decision_mode == "frontier":
                    frontier_commitment.invalidate_active(step, reason, blacklist=True)
                if paper_mode:
                    long_term_goal.clear("reached")
                current_path = []
                force_perception_step = True
                full_path.append(tuple(int(v) for v in current_grid))
                last_decision_reason = reason
                if last_nav_decision is not None:
                    last_nav_decision.reason = reason
                    last_nav_decision.metadata = {
                        **dict(last_nav_decision.metadata or {}),
                        "target_radius_reached": True,
                        "target_reached_distance_m": float(target_distance_m),
                        "target_reached_radius_m": float(args.frontier_commit_reached_radius_m),
                    }
                last_frontier_commitment_reason = reason
                last_frontier_commitment_metadata = {
                    **dict(last_frontier_commitment_metadata),
                    "frontier_commitment_reason": reason,
                    "target_radius_reached": True,
                    "target_reached_distance_m": float(target_distance_m),
                    "target_reached_radius_m": float(args.frontier_commit_reached_radius_m),
                }
                obs = server.step_kinematic_velocity(
                    0.0,
                    0.0,
                    0.0,
                    dt=float(args.control_dt),
                    render_updates=int(args.render_updates_per_step),
                    read_rgb=detector_requires_rgb(detector, args.detector),
                    read_depth=True,
                    rgb_device="cuda" if detector_cuda_rgb else "cpu",
                )
                continue

            if (
                last_nav_decision is not None
                and last_nav_decision.mode == "frontier"
                and current_path
                and np.isfinite(float(target_distance_m))
            ):
                progress_delta = float(args.frontier_commit_progress_min_delta_m)
                if float(target_distance_m) + progress_delta < float(nav_execution_best_distance_m):
                    nav_execution_best_distance_m = float(target_distance_m)
                    nav_execution_no_progress_steps = 0
                else:
                    nav_execution_no_progress_steps += 1
                if nav_execution_no_progress_steps >= int(args.frontier_commit_no_progress_steps):
                    reason = "frontier_no_progress_during_execution"
                    execution_best_distance_m = float(nav_execution_best_distance_m)
                    selected_frontier = (
                        last_nav_decision.frontier_decision.selected_frontier
                        if last_nav_decision.frontier_decision is not None
                        else None
                    )
                    if frontier_commitment is not None:
                        if selected_frontier is not None:
                            frontier_commitment.blacklist_frontier(selected_frontier, step, reason)
                        frontier_commitment.invalidate_active(step, reason, blacklist=True)
                    long_term_goal.invalidate(reason)
                    frontier_stop_at_current_grid = [int(current_grid[0]), int(current_grid[1])]
                    frontier_blacklisted = True
                    frontier_unreachable_recovery = True
                    frontier_unreachable_reason = reason
                    last_decision_reason = reason
                    current_path = []
                    full_path.append(tuple(int(v) for v in current_grid))
                    force_perception_step = True
                    nav_execution_progress_key = None
                    nav_execution_best_distance_m = float("inf")
                    nav_execution_no_progress_steps = 0
                    last_frontier_commitment_reason = "%s_blacklisted" % reason
                    last_frontier_commitment_metadata = {
                        **dict(last_frontier_commitment_metadata),
                        "frontier_commitment_reason": last_frontier_commitment_reason,
                        "frontier_blacklisted": True,
                        "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
                        "frontier_execution_no_progress_steps": int(args.frontier_commit_no_progress_steps),
                        "frontier_execution_best_distance_m": float(execution_best_distance_m),
                        "frontier_execution_distance_m": float(target_distance_m),
                    }
                    if last_nav_decision is not None:
                        last_nav_decision.reason = reason
                        last_nav_decision.metadata = {
                            **dict(last_nav_decision.metadata or {}),
                            "frontier_execution_no_progress": True,
                            "frontier_execution_distance_m": float(target_distance_m),
                            "frontier_execution_no_progress_steps": int(args.frontier_commit_no_progress_steps),
                        }
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        0.0,
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=detector_requires_rgb(detector, args.detector),
                        read_depth=True,
                        rgb_device="cuda" if detector_cuda_rgb else "cpu",
                    )
                    continue

            path_world = path_cells_to_world(current_path[1: min(len(current_path), 20)], dynamic_map_info)
            if not path_world:
                if evaluator.final_distance_to_goal <= success_distance:
                    gt_success_region_reached = True
                    if explore_until_no_frontiers:
                        gt_success_ignored_steps += 1
                if success_region_can_finish(
                    evaluator.final_distance_to_goal,
                    success_distance,
                    require_sgnav_stop=bool(args.require_sgnav_stop),
                    policy_stop_confirmed=policy_stop_confirmed,
                    ignore_goal_success=explore_until_no_frontiers,
                ):
                    stop_called = True
                    if viz is not None:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=free,
                            observed=observed,
                            goal_cells=goal_cells,
                            current_grid=current_grid,
                            pose=pose,
                            frontiers=last_frontiers,
                            nav_decision=last_nav_decision,
                            current_path=current_path,
                            full_path=full_path,
                            object_memory=object_memory,
                            goal_category=episode["goal_category"],
                            distance_to_goal=evaluator.final_distance_to_goal,
                            path_length=float(evaluator.path_accum.total_m),
                            scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                            score_debug=scenegraph.last_score_debug,
                            failure_reason=failure_reason,
                        )
                    break
                if evaluator.final_distance_to_goal <= success_distance and bool(args.require_sgnav_stop):
                    gt_success_without_sgnav_stop_steps += 1
                    stop_blocked_reason = "sgnav_stop_required"
                    if not logged_gt_success_without_sgnav_stop:
                        print(
                            "[sgnav-loop] local path ended inside GT success radius, but SG-Nav STOP is not confirmed; continuing",
                            flush=True,
                        )
                        logged_gt_success_without_sgnav_stop = True
                if last_decision_mode in {"candidate", "frontier"}:
                    if paper_mode:
                        long_term_goal.clear("reached")
                    current_path = []
                    force_perception_step = True
                    full_path.append(tuple(int(v) for v in current_grid))
                    if viz is not None:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=free,
                            observed=observed,
                            goal_cells=goal_cells,
                            current_grid=current_grid,
                            pose=pose,
                            frontiers=last_frontiers,
                            nav_decision=last_nav_decision,
                            current_path=current_path,
                            full_path=full_path,
                            object_memory=object_memory,
                            goal_category=episode["goal_category"],
                            distance_to_goal=evaluator.final_distance_to_goal,
                            path_length=float(evaluator.path_accum.total_m),
                            scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                            score_debug=scenegraph.last_score_debug,
                        )
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        float(args.reperception_turn_wz_radps),
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=detector_requires_rgb(detector, args.detector),
                        read_depth=True,
                        rgb_device="cuda" if detector_cuda_rgb else "cpu",
                    )
                    continue
                else:
                    failure_reason = "path_exhausted_without_success"
                if viz is not None:
                    viz.update(
                        step=step,
                        rgb=viz_rgb(obs),
                        detections_2d=last_detections_2d,
                        occupancy=occupancy,
                        navigable=free,
                        observed=observed,
                        goal_cells=goal_cells,
                        current_grid=current_grid,
                        pose=pose,
                        frontiers=last_frontiers,
                        nav_decision=last_nav_decision,
                        current_path=current_path,
                        full_path=full_path,
                        object_memory=object_memory,
                        goal_category=episode["goal_category"],
                        distance_to_goal=evaluator.final_distance_to_goal,
                        path_length=float(evaluator.path_accum.total_m),
                        scenegraph_backend="original" if scenegraph.scenegraph is not None else "fallback",
                        score_debug=scenegraph.last_score_debug,
                        failure_reason=failure_reason,
                    )
                break
            cmd = follower.compute_cmd(pose, path_world)
            cmd, blocked_by_guard = guard_kinematic_cmd(
                tuple(float(v) for v in pose),
                cmd,
                float(args.control_dt),
                navigable,
                dynamic_map_info,
                float(args.camera_forward_offset_m),
            )
            if blocked_by_guard:
                if paper_mode and last_decision_mode == "frontier":
                    reason = "frontier_unreachable_stop_at_current_pose"
                    if frontier_commitment is not None:
                        frontier_commitment.invalidate_active(step, reason, blacklist=True)
                    long_term_goal.invalidate(reason)
                    frontier_stop_at_current_grid = [int(current_grid[0]), int(current_grid[1])]
                    frontier_blacklisted = True
                    frontier_unreachable_recovery = True
                    frontier_unreachable_reason = reason
                    last_frontier_commitment_reason = "%s_blacklisted" % reason
                    last_frontier_commitment_metadata = {
                        **dict(last_frontier_commitment_metadata),
                        "frontier_commitment_reason": last_frontier_commitment_reason,
                        "frontier_blacklisted": True,
                        "frontier_stop_at_current_grid": frontier_stop_at_current_grid,
                    }
                    current_path = []
                    full_path.append(tuple(int(v) for v in current_grid))
                    force_perception_step = True
                    obs = server.step_kinematic_velocity(
                        0.0,
                        0.0,
                        0.0,
                        dt=float(args.control_dt),
                        render_updates=int(args.render_updates_per_step),
                        read_rgb=detector_requires_rgb(detector, args.detector),
                        read_depth=True,
                        rgb_device="cuda" if detector_cuda_rgb else "cpu",
                    )
                    continue
                failure_reason = "kinematic_collision_guard"
                break
            next_step = step + 1
            next_rgb_device = rgb_request_device(next_step)
            obs = server.step_kinematic_velocity(
                cmd[0],
                cmd[1],
                cmd[2],
                dt=float(args.control_dt),
                render_updates=int(args.render_updates_per_step),
                read_rgb=next_rgb_device is not None,
                read_depth=True,
                rgb_device=next_rgb_device,
            )
        else:
            if failure_reason is None:
                failure_reason = "max_control_steps"

        row = evaluator.finish(stop_called=stop_called, planner=args.planner, detector=args.detector, failure_reason=failure_reason)
        row["sim_backend"] = "isaac"
        row["closed_loop"] = True
        row["control_mode"] = "kinematic_holonomic"
        row["stop_called"] = bool(stop_called)
        row["policy_stop_confirmed"] = bool(policy_stop_confirmed)
        row["success_requires_sgnav_stop"] = bool(args.require_sgnav_stop)
        row["gt_success_region_reached"] = bool(gt_success_region_reached)
        row["gt_success_without_sgnav_stop_steps"] = int(gt_success_without_sgnav_stop_steps)
        row["explore_until_no_frontiers"] = bool(explore_until_no_frontiers)
        row["goal_success_ignored_steps"] = int(gt_success_ignored_steps)
        row["target_radius_reached_steps"] = int(target_radius_reached_steps)
        row["target_reached_radius_m"] = float(args.frontier_commit_reached_radius_m)
        row["stop_blocked_reason"] = stop_blocked_reason
        row["seeded_object_memory_count"] = int(seeded)
        row["object_memory_count"] = int(len(object_memory.nodes))
        row["goal_candidate_count"] = int(goal_candidate_count)
        row["sgnav_decision_mode"] = last_decision_mode
        row["sgnav_decision_reason"] = last_decision_reason
        row["scenegraph_backend"] = "original" if scenegraph.scenegraph is not None else "fallback"
        row["sgnav_state"] = getattr(decision_policy, "state", last_decision_mode)
        decision_metadata = dict(getattr(last_nav_decision, "metadata", {}) or {})
        row["sgnav_decision_metadata"] = decision_metadata
        row["target_cells_count"] = int(len(getattr(last_nav_decision, "target_cells", []) or []))
        row["selected_frontier_index"] = (
            int(last_nav_decision.frontier_decision.selected_index)
            if last_nav_decision is not None
            and last_nav_decision.frontier_decision is not None
            and last_nav_decision.frontier_decision.selected_index is not None
            else None
        )
        row["candidate_credibility"] = decision_metadata.get("candidate_credibility")
        row["candidate_reperception_steps"] = decision_metadata.get("candidate_reperception_steps")
        row["candidate_rejected"] = bool(decision_metadata.get("candidate_rejected", False))
        row["candidate_accepted"] = bool(decision_metadata.get("candidate_accepted", False))
        row["detector_confidence_threshold"] = float(args.detector_conf)
        row["min_valid_detection_confidence"] = float(args.min_valid_detection_confidence)
        row["candidate_start_min_confidence"] = float(args.candidate_start_min_confidence)
        row["candidate_start_min_hits"] = int(args.candidate_start_min_hits)
        row["candidate_standoff_max_cells"] = int(args.candidate_standoff_max_cells)
        row["detected_category_counts"] = {
            str(key): int(value)
            for key, value in sorted(detection_category_counts.items(), key=lambda item: (-item[1], item[0]))
        }
        row["goal_detection_history"] = list(goal_detection_history)
        object_memory_gnn_snapshot = fused_instance_registry.gnn_snapshot()
        row["object_memory_gnn_snapshot"] = object_memory_gnn_snapshot
        registry_raw_detections = list(object_memory_gnn_snapshot.get("raw_detections") or [])
        pre_registry_rejections = [item for item in raw_detection_debug_log if item.get("reject_reason")]
        row["raw_detection_log"] = (
            pre_registry_rejections + registry_raw_detections
            if registry_raw_detections
            else list(raw_detection_debug_log)
        )
        row["raw_rejected_detections_count"] = int(
            len([item for item in row["raw_detection_log"] if item.get("reject_reason")])
        )
        row["object_memory_tracks"] = object_memory.to_dicts()
        row["selected_candidate"] = last_selected_candidate.to_dict() if last_selected_candidate is not None else None
        row["object_memory_goal_candidates"] = [
            node.to_dict()
            for node in object_memory.nodes
            if category_matches_goal(node.category)
        ]
        row["candidate_pair_distances_m"] = goal_candidate_pair_distances(object_memory, episode["goal_category"])
        row["candidate_duplicate_warning"] = any(float(pair["dist"]) < 0.75 for pair in row["candidate_pair_distances_m"])
        row["panorama_frames"] = int(panorama_frames)
        row["requested_perception_every_steps"] = int(requested_perception_every)
        row["effective_perception_every_steps"] = int(perception_every)
        row["open_vocab_detector_every_frame"] = bool(str(getattr(args, "detector", "")).strip().lower() in {"yolo_world", "grounding_dino"} and perception_every == 1)
        row["yolo_world_every_frame"] = bool(str(getattr(args, "detector", "")).strip().lower() == "yolo_world" and perception_every == 1)
        row["graph_object_nodes"] = int(len(getattr(scenegraph, "runtime_nodes", {})))
        row["graph_group_nodes"] = int(len(getattr(scenegraph, "runtime_groups", [])))
        row["graph_room_nodes"] = int(len(getattr(scenegraph, "runtime_rooms", {})))
        row["graph_edges"] = int(len(getattr(scenegraph, "runtime_edges", [])))
        paper_graph = getattr(scenegraph, "paper_graph", None)
        row["sgnav_mode"] = str(getattr(args, "sgnav_mode", "legacy"))
        row["long_term_goal"] = {
            "mode": str(long_term_goal.mode),
            "target_cells_count": int(len(long_term_goal.target_cells)),
            "center_grid": list(long_term_goal.center_grid) if long_term_goal.center_grid is not None else None,
            "selected_step": int(long_term_goal.selected_step),
            "invalid_reason": str(long_term_goal.invalid_reason),
            "reached": bool(long_term_goal.reached),
        }
        row["frontier_target_mode"] = frontier_target_mode
        row["frontier_center_grid"] = frontier_center_grid
        row["frontier_actual_target_grid"] = frontier_actual_target_grid
        row["frontier_unreachable_recovery"] = bool(frontier_unreachable_recovery)
        row["frontier_unreachable_reason"] = frontier_unreachable_reason
        row["frontier_stop_at_current_grid"] = frontier_stop_at_current_grid
        row["frontier_blacklisted"] = bool(frontier_blacklisted)
        row["active_long_term_goal_mode"] = str(long_term_goal.mode)
        row["active_long_term_goal_age"] = (
            max(0, int(evaluator.num_steps) - int(long_term_goal.selected_step))
            if int(long_term_goal.selected_step) >= 0
            else None
        )
        row["paper_num_object_nodes"] = int(len(getattr(paper_graph, "object_nodes", {}) or {}))
        row["paper_num_room_nodes"] = int(len(getattr(paper_graph, "room_nodes", {}) or {}))
        row["paper_num_group_nodes"] = int(len(getattr(paper_graph, "group_nodes", {}) or {}))
        row["paper_num_object_edges"] = int(len(getattr(paper_graph, "object_edges", []) or []))
        row["paper_frontier_interpolation"] = dict(getattr(scenegraph, "last_score_debug", {}) or {})
        row["vllm_frontier_scoring"] = bool(getattr(args, "vllm_frontier_scoring", False))
        row["vllm_image_scoring"] = bool(getattr(args, "vllm_image_scoring", True))
        row["score_frontiers_before_candidate"] = bool(getattr(args, "score_frontiers_before_candidate", False))
        vllm_scorer = getattr(scenegraph, "vllm_scorer", None)
        row["vllm_last_used_image"] = bool(getattr(vllm_scorer, "last_used_image", False))
        row["vllm_num_requests"] = int(getattr(vllm_scorer, "request_count", 0))
        row["vllm_cache_hits"] = int(getattr(vllm_scorer, "cache_hit_count", 0))
        row["vllm_last_skip_reason"] = getattr(vllm_scorer, "last_skip_reason", None)
        row["vllm_last_request_frontiers"] = int(getattr(vllm_scorer, "last_request_frontiers", 0))
        row["vllm_last_response_chars"] = int(getattr(vllm_scorer, "last_response_chars", 0))
        row["vllm_disabled_reason"] = getattr(vllm_scorer, "disabled_reason", None)
        paper_llm_client = getattr(scenegraph, "paper_llm_client", None)
        hcot_scorer = getattr(scenegraph, "hcot_scorer", None)
        row["paper_llm_enabled"] = paper_llm_client is not None
        row["paper_llm_requests"] = int(getattr(paper_llm_client, "request_count", 0))
        row["hcot_llm_enabled"] = getattr(hcot_scorer, "llm_client", None) is not None
        row["hcot_llm_attempts"] = int(getattr(hcot_scorer, "llm_request_count", 0))
        row["hcot_llm_failures"] = int(getattr(hcot_scorer, "llm_failure_count", 0))
        row["hcot_llm_fallbacks"] = int(getattr(hcot_scorer, "llm_fallback_count", 0))
        row["hcot_llm_fallback_count"] = int(getattr(hcot_scorer, "llm_fallback_count", 0))
        row["hcot_llm_last_error"] = getattr(hcot_scorer, "last_error", None)
        row["hc_p_num_subgraphs_total"] = row["paper_frontier_interpolation"].get("num_subgraphs_total")
        row["hc_p_num_subgraphs_scored"] = row["paper_frontier_interpolation"].get("num_subgraphs_scored")
        row["detection_localization"] = detection_localization
        row["read_depth"] = bool(getattr(args, "read_depth", False))
        row["mapping_source"] = (
            "depth_ray_online+static_nearfield"
            if bool(getattr(args, "static_nearfield_map", False))
            else "depth_ray_online"
        )
        row["online_map_resolution_m"] = float(dynamic_map_info.resolution_m)
        row["robot_radius_m"] = float(args.robot_radius_m)
        row["robot_width_m"] = float(args.robot_radius_m) * 2.0
        row["online_effective_obstacle_inflation_m"] = float(args.robot_radius_m)
        row["online_extra_inflation_radius_m_requested"] = float(args.online_inflation_radius_m)
        row["online_observed_cells"] = int(np.count_nonzero(last_dynamic_observed))
        row["online_raw_free_cells"] = int(np.count_nonzero(last_dynamic_free))
        row["online_free_cells"] = int(np.count_nonzero(last_dynamic_navigable))
        row["online_astar_extra_clearance_m"] = float(args.runtime_planning_clearance_m)
        row["online_astar_clearance_source"] = "raw_occupied"
        row["online_astar_free_cells"] = int(np.count_nonzero(last_dynamic_astar_navigable))
        row["online_occupied_cells"] = int(np.count_nonzero(last_dynamic_occupancy))
        row["online_mapper_debug"] = dict(getattr(mapper, "last_debug_stats", {}))
        row["nearfield_depth"] = bool(getattr(args, "nearfield_depth", False))
        row["nearfield_mapper_debug"] = dict(getattr(mapper, "last_nearfield_debug_stats", {}))
        row["static_nearfield_map"] = bool(getattr(args, "static_nearfield_map", False))
        row["static_nearfield_radius_m"] = float(getattr(args, "static_nearfield_radius_m", 1.0))
        row["static_nearfield_mapper_debug"] = dict(getattr(mapper, "last_static_nearfield_debug_stats", {}))
        row["frontier_raw_cells"] = int(last_frontier_raw_cells)
        row["frontier_clusters"] = int(last_frontier_clusters)
        row["frontier_unknown_source"] = str(args.frontier_unknown_source)
        row["frontier_cluster_distance_mode"] = str(args.frontier_cluster_distance_mode)
        row["frontier_allow_near_fallback"] = bool(args.frontier_allow_near_fallback)
        row["frontier_min_distance_m"] = float(args.frontier_min_distance_m)
        row["frontier_selectable_failure_reason"] = decision_metadata.get("frontier_selectable_failure_reason")
        row["frontier_debug_dump"] = bool(args.frontier_debug_dump)
        row["frontier_debug_dir"] = str(args.frontier_debug_dir)
        row["frontier_commitment_enabled"] = bool(args.frontier_commitment_enabled)
        row["frontier_commitment_reason"] = str(last_frontier_commitment_reason)
        row["frontier_commitment"] = dict(last_frontier_commitment_metadata)
        row["active_frontier_id"] = last_frontier_commitment_metadata.get("active_frontier_id")
        row["active_frontier_age"] = last_frontier_commitment_metadata.get("active_frontier_age")
        row["active_frontier_distance_m"] = last_frontier_commitment_metadata.get("active_frontier_distance_m")
        row["frontier_scenegraph_score_norm"] = str(args.frontier_scenegraph_score_norm)
        row["room_map_mode"] = str(room_map_mode)
        row["roomseg_debug_only"] = bool(roomseg_debug_only)
        room_context_row = dict(last_room_context_metadata)
        row["room_context"] = room_context_row
        for key in (
            "room_context_source",
            "room_update_invoked_for_frontier_scoring",
            "room_segmentation_ran",
            "room_labeling_ran",
            "room_context_cache_hit",
            "room_label_count",
            "room_label_requests",
            "room_label_cache_hits",
            "room_call_order_trace",
            "room_segmentation_called_for",
            "room_segmentation_algorithm",
            "room_segmentation_step_index",
            "room_vlm_called",
            "scenegraph_updated_after_room_context",
            "frontier_scoring_after_room_context",
        ):
            row[key] = room_context_row.get(key)
        row["room_segmentation"] = dict(last_room_segmentation_debug)
        row["room_semantics"] = dict(last_room_semantics_debug)
        row["room_mask_count"] = int(room_context_row.get("room_mask_count", len([room for room in last_room_masks if not getattr(room, "stale", False)])) or 0)
        row["room_vlm_backend"] = str(getattr(room_labeler, "backend", "unavailable") if room_labeler is not None else "unavailable")
        row["room_vlm_requests"] = int(getattr(room_labeler, "request_count", 0) if room_labeler is not None else 0)
        row["room_vlm_failures"] = int(getattr(room_labeler, "failure_count", 0) if room_labeler is not None else 0)
        row["room_vlm_invalid_json"] = bool(getattr(room_labeler, "failure_count", 0) if room_labeler is not None else 0)
        row["reperception_state"] = build_reperception_state_payload(decision_metadata)
        row["stop_state"] = build_stop_state_payload(last_nav_decision, row)
        row["debug_graph_dump"] = bool(args.debug_graph_dump)
        row["debug_graph_dump_dir"] = str(args.debug_graph_dump_dir)
        row["segmenter"] = str(getattr(args, "segmenter", "none") or "none")
        row["camera_annotator_device"] = camera_annotator_device
        row["detector_cuda_rgb"] = bool(detector_cuda_rgb)
        row["isaac_visual_robot_proxy"] = bool(obs.get("visual_robot_proxy", False)) if isinstance(obs, dict) else False
        row["isaac_robot_pose_sync_failures"] = int(obs.get("robot_pose_sync_failures", 0)) if isinstance(obs, dict) else 0
        row["isaac_camera_pose_sync_failures"] = int(obs.get("camera_pose_sync_failures", 0)) if isinstance(obs, dict) else 0
        row["isaac_nearfield_camera_pose_sync_failures"] = (
            int(obs.get("nearfield_camera_pose_sync_failures", 0)) if isinstance(obs, dict) else 0
        )
        row["sgnav_viz_every_steps"] = int(viz_every)
        row["max_vx_mps"] = float(args.max_vx_mps)
        row["max_vy_mps"] = float(args.max_vy_mps)
        row["max_wz_radps"] = float(args.max_wz_radps)
        row["perception_latency_ms"] = float(latency_totals_ms["perception"])
        row["mapping_latency_ms"] = float(latency_totals_ms["mapping"])
        row["graph_latency_ms"] = float(latency_totals_ms["graph"])
        row["llm_latency_ms"] = float(latency_totals_ms["llm"])
        row["planning_latency_ms"] = float(latency_totals_ms["planning"])
        row["latency_counts"] = {key: int(value) for key, value in latency_counts.items()}
        row["mapping_latency_breakdown_ms"] = {
            str(key): float(value) for key, value in sorted(mapping_breakdown_totals_ms.items())
        }
        row["mapping_latency_breakdown_counts"] = {
            str(key): int(value) for key, value in sorted(mapping_breakdown_counts.items())
        }
        row["mapping_latency_breakdown_avg_ms"] = {
            str(key): float(mapping_breakdown_totals_ms[key] / max(1, mapping_breakdown_counts.get(key, 0)))
            for key in sorted(mapping_breakdown_totals_ms)
        }
        for key, value in row["mapping_latency_breakdown_ms"].items():
            row["mapping_%s" % str(key)] = float(value)
        if args.save_debug_video or args.debug_map:
            debug_map = args.debug_map or str(Path(args.output).with_suffix(".png"))
            start = world_xy_to_grid(float(start_pose[0]), float(start_pose[1]), dynamic_map_info)
            save_map_png(debug_map, last_dynamic_occupancy, last_dynamic_navigable, start=start, goals=goal_cells, path_cells=full_path)
        row = complete_result_row(row, args)
        if roomseg_debug_only:
            row["metric_valid"] = False
            row["detector_backend"] = "none"
            row["segmenter_backend"] = "none"
            row["llm_backend"] = "disabled_roomseg_debug_only"
            row["policy_name"] = "roomseg_debug_only"
            row["sgnav_decision_mode"] = "roomseg_debug_only"
            row["sgnav_decision_reason"] = "roomseg_debug_only_no_frontier_or_policy_decision"
        row = make_jsonable(row)
        summary_row = final_log_row(row)
        JsonlEpisodeLogger(args.output).log(row)
        print(json.dumps(summary_row, ensure_ascii=False), flush=True)
        args._row_already_logged = True
        if args.hold_open:
            print("[isaac-loop] episode finished; hold-open enabled, close the Isaac window or press Ctrl+C", flush=True)
            while True:
                server.app.update()
        return row
    except BaseException:
        import traceback

        print("[sgnav-loop] exception before Isaac shutdown:", flush=True)
        traceback.print_exc()
        raise
    finally:
        detector = getattr(args, "_detector_instance", None)
        if detector is not None and hasattr(detector, "close"):
            detector.close()
            args._detector_instance = None
            args._detector_key = None
        if viz is not None:
            viz.close()
        if not args.hold_open:
            server.close()


def run_episode_map_sim(episode: dict, args) -> dict:
    scene_dir, map_info, occupancy, navigable = load_preprocessed_for_episode(episode)
    navigable = apply_episode_planning_clearance(
        scene_dir,
        map_info,
        navigable,
        episode,
        runtime_planning_clearance_m=getattr(args, "runtime_planning_clearance_m", 0.0),
    )
    ensure_detector_loaded(args, scene_dir)
    planner = GridAStarPlanner(navigable, map_info.resolution_m, allow_diagonal=True)
    start = (int(episode["start_grid"][0]), int(episode["start_grid"][1]))
    goals = [(int(r), int(c)) for r, c in episode["goal_regions_grid"]]
    planning_started_at = time.perf_counter()
    result = planner.plan(start, goals)
    planning_latency_ms = max(0.0, (time.perf_counter() - planning_started_at) * 1000.0)
    evaluator = EpisodeEvaluator(episode, planner)
    env = MapSimHabitatLikeEnv(args.episode_file, args.episode_index)
    env.reset()

    object_memory = ObjectMemory(min_valid_confidence=float(args.min_valid_detection_confidence))
    scenegraph = SGNavSceneGraphAdapter(
        args.sgnav_repo,
        use_original=args.use_original_scenegraph,
        semantic_priors_path=getattr(args, "semantic_priors_path", None),
        llm_config={
            "enabled": bool(getattr(args, "llm_enabled", False)),
            "base_url": getattr(args, "llm_base_url", None),
            "model": getattr(args, "llm_model", None),
            "api_key": getattr(args, "llm_api_key", None),
            "timeout_s": float(getattr(args, "llm_timeout_s", 30.0)),
            "temperature": float(getattr(args, "llm_temperature", 0.0)),
            "max_tokens": int(getattr(args, "llm_max_tokens", 512)),
            "max_hcot_subgraphs_per_decision": int(getattr(args, "max_hcot_subgraphs_per_decision", 8)),
        },
        sgnav_mode=str(getattr(args, "sgnav_mode", "legacy")),
    )
    scenegraph.reset(episode["goal_category"])
    scenegraph.update(object_memory)
    decision = SGNavDecision(scenegraph)
    _ = decision  # kept to make the SG-Nav decision adapter part of the run loop surface.

    failure_reason = None
    stop_called = False
    if not result.path:
        failure_reason = "astar_no_path"
        evaluator.final_distance_to_goal = math.inf
    else:
        for cell in result.path:
            x, y = grid_to_world_xy(cell[0], cell[1], map_info)
            pose = [x, y, episode["start_pose_world"][2], episode["start_pose_world"][3]]
            env.set_pose(pose)
            evaluator.update_pose(pose, cell, collided=False)
        stop_called = True

    row = evaluator.finish(stop_called=stop_called, planner=args.planner, detector=args.detector, failure_reason=failure_reason)
    row["sim_backend"] = "map"
    row["map_source"] = "static_preprocessed"
    row["policy_name"] = "%s_static_map_baseline" % str(args.planner)
    row["object_memory_count"] = 0
    row["goal_candidate_count"] = 0
    row["frontier_count"] = 0
    row["selected_frontier"] = None
    row["detector_confidence_threshold"] = float(args.detector_conf)
    row["min_valid_detection_confidence"] = float(args.min_valid_detection_confidence)
    row["planning_latency_ms"] = float(planning_latency_ms)
    row = complete_result_row(row, args)
    if args.save_debug_video or args.debug_map:
        debug_map = args.debug_map or str(Path(args.output).with_suffix(".png"))
        save_map_png(debug_map, occupancy, navigable, start=start, goals=goals, path_cells=result.path)
    return row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="isaac_bench/configs/isaac_bench.yaml")
    parser.add_argument("--episode-file", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--success-distance-m", type=float, default=None)
    parser.add_argument("--planner", default=None, choices=["astar", "nav2"])
    parser.add_argument("--detector", default=None, choices=["dry_run", "yolo_world", "grounding_dino", "none"])
    parser.add_argument("--yolo-world-model", default=None)
    parser.add_argument("--grounding-dino-checkpoint", default=None)
    parser.add_argument("--grounding-dino-config", default=None)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=None)
    parser.add_argument("--grounding-dino-device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--detector-conf", type=float, default=None)
    parser.add_argument("--min-valid-detection-confidence", type=float, default=None)
    parser.add_argument("--detector-iou", type=float, default=None)
    parser.add_argument("--reject-edge-touching-bboxes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bbox-edge-margin-px", type=float, default=None)
    parser.add_argument("--bbox-edge-margin-ratio", type=float, default=None)
    parser.add_argument("--headless", nargs="?", const=True, default=None, type=str_to_bool)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--sim-backend", default=None, choices=["map", "isaac"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--debug-map", default=None)
    parser.add_argument("--save-debug-video", action="store_true")
    parser.add_argument("--strict-benchmark", nargs="?", const=True, default=None, type=str_to_bool)
    parser.add_argument("--no-strict-benchmark", dest="strict_benchmark", action="store_false")
    parser.add_argument("--allow-debug-fallbacks", action="store_true", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--ablation-name", default=None)
    parser.add_argument("--sgnav-repo", default=None)
    parser.add_argument("--sgnav-mode", default=None, choices=["legacy", "paper"])
    parser.add_argument("--use-original-scenegraph", action="store_true", default=None)
    parser.add_argument("--max-control-steps", type=int, default=None)
    parser.add_argument(
        "--explore-until-no-frontiers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Debug/probe mode: ignore goal success radius and keep selecting frontiers until none remain or max-control-steps is reached.",
    )
    parser.add_argument("--control-dt", type=float, default=None)
    parser.add_argument("--replan-every-steps", type=int, default=None)
    parser.add_argument("--perception-every-steps", type=int, default=None)
    parser.add_argument("--observed-radius-m", type=float, default=None)
    parser.add_argument("--lookahead-m", type=float, default=None)
    parser.add_argument("--max-vx-mps", type=float, default=None)
    parser.add_argument("--max-vy-mps", type=float, default=None)
    parser.add_argument("--max-wz-radps", type=float, default=None)
    parser.add_argument("--render-updates-per-step", type=int, default=None)
    parser.add_argument("--isaac-width", type=int, default=None)
    parser.add_argument("--isaac-height", type=int, default=None)
    parser.add_argument("--camera-hfov-deg", type=float, default=None)
    parser.add_argument("--camera-mast-height-m", type=float, default=None)
    parser.add_argument("--camera-forward-offset-m", type=float, default=None)
    parser.add_argument("--camera-pitch-deg", type=float, default=None)
    parser.add_argument("--camera-near-m", type=float, default=None)
    parser.add_argument("--camera-far-m", type=float, default=None)
    parser.add_argument("--camera-annotator-device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--read-depth", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--nearfield-depth", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--nearfield-width", type=int, default=None)
    parser.add_argument("--nearfield-height", type=int, default=None)
    parser.add_argument("--nearfield-hfov-deg", type=float, default=None)
    parser.add_argument("--nearfield-height-m", type=float, default=None)
    parser.add_argument("--nearfield-near-m", type=float, default=None)
    parser.add_argument("--nearfield-far-m", type=float, default=None)
    parser.add_argument("--nearfield-radius-m", type=float, default=None)
    parser.add_argument("--nearfield-ignore-radius-m", type=float, default=None)
    parser.add_argument("--nearfield-depth-stride-px", type=int, default=None)
    parser.add_argument("--nearfield-floor-tolerance-m", type=float, default=None)
    parser.add_argument("--nearfield-obstacle-min-height-m", type=float, default=None)
    parser.add_argument("--nearfield-obstacle-max-height-m", type=float, default=None)
    parser.add_argument("--nearfield-splat-point-threshold", type=int, default=None)
    parser.add_argument("--nearfield-free-splat-point-threshold", type=int, default=None)
    parser.add_argument("--static-nearfield-map", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--static-nearfield-radius-m", type=float, default=None)
    parser.add_argument("--panorama-steps", type=int, default=None)
    parser.add_argument("--panorama-wz-radps", type=float, default=None)
    parser.add_argument("--panorama-render-updates-per-step", type=int, default=None)
    parser.add_argument("--depth-max-m", type=float, default=None)
    parser.add_argument("--depth-min-m", type=float, default=None)
    parser.add_argument("--depth-stride-px", type=int, default=None)
    parser.add_argument("--online-map-size-m", type=float, default=None)
    parser.add_argument("--online-resolution-m", type=float, default=None)
    parser.add_argument("--obstacle-min-height-m", type=float, default=None)
    parser.add_argument("--obstacle-max-height-m", type=float, default=None)
    parser.add_argument("--free-min-height-m", type=float, default=None)
    parser.add_argument("--free-max-height-m", type=float, default=None)
    parser.add_argument("--splat-point-threshold", type=int, default=None)
    parser.add_argument("--free-splat-point-threshold", type=int, default=None)
    parser.add_argument("--mapping-debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-min-cluster-size", type=int, default=None, help="Deprecated; original SG-Nav FBE uses per-cell frontiers without clustering.")
    parser.add_argument("--frontier-min-distance-m", type=float, default=None)
    parser.add_argument("--frontier-max-count", type=int, default=None)
    parser.add_argument("--frontier-obstacle-dilation-radius-cells", type=int, default=None)
    parser.add_argument("--frontier-unknown-dilation-radius-cells", type=int, default=None)
    parser.add_argument("--frontier-unknown-source", "--frontier_unknown_source", default=None, choices=["observed", "implicit"])
    parser.add_argument(
        "--frontier-cluster-distance-mode",
        "--frontier_cluster_distance_mode",
        default=None,
        choices=["mean", "min", "center"],
    )
    parser.add_argument("--frontier-allow-near-fallback", "--frontier_allow_near_fallback", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-debug-dump", "--frontier_debug_dump", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-debug-dir", "--frontier_debug_dir", default=None)
    parser.add_argument("--frontier-commitment-enabled", "--frontier_commitment_enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontier-commit-match-radius-m", "--frontier_commit_match_radius_m", type=float, default=None)
    parser.add_argument("--frontier-commit-reached-radius-m", "--frontier_commit_reached_radius_m", type=float, default=None)
    parser.add_argument("--frontier-commit-min-steps", "--frontier_commit_min_steps", type=int, default=None)
    parser.add_argument("--frontier-commit-max-steps", "--frontier_commit_max_steps", type=int, default=None)
    parser.add_argument("--frontier-commit-switch-margin", "--frontier_commit_switch_margin", type=float, default=None)
    parser.add_argument("--frontier-commit-switch-ratio", "--frontier_commit_switch_ratio", type=float, default=None)
    parser.add_argument("--frontier-commit-no-progress-steps", "--frontier_commit_no_progress_steps", type=int, default=None)
    parser.add_argument("--frontier-commit-progress-min-delta-m", "--frontier_commit_progress_min_delta_m", type=float, default=None)
    parser.add_argument("--frontier-blacklist-ttl-steps", "--frontier_blacklist_ttl_steps", type=int, default=None)
    parser.add_argument("--robot-radius-m", type=float, default=None)
    parser.add_argument(
        "--online-inflation-radius-m",
        type=float,
        default=None,
        help="Deprecated compatibility option; online traversal inflation is footprint-only.",
    )
    parser.add_argument("--room-map-mode", default=None)
    parser.add_argument(
        "--roomseg-debug-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run only online mapping plus the current room segmentation backend; skip detector, scene graph, frontier scoring, and local planning.",
    )
    parser.add_argument("--debug-roomseg-layers", action="store_true", default=None)
    parser.add_argument("--roomseg-debug-layers", dest="debug_roomseg_layers", action="store_true", default=None)
    parser.add_argument(
        "--save-roomseg-snapshots",
        action="store_true",
        default=None,
        help="Save lightweight per-roomseg npz, summary, and unlabeled navigation mask images without enabling debug layer dumps.",
    )
    parser.add_argument("--debug-roomseg-dir", default=None)
    parser.add_argument("--roomseg-debug-dir", dest="debug_roomseg_dir", default=None)
    parser.add_argument("--roomseg-snapshot-dir", default=None)
    parser.add_argument("--debug-roomseg-max-saves", type=int, default=None)
    parser.add_argument("--roomseg-snapshot-max-saves", type=int, default=None)
    parser.add_argument("--roomseg-roomseg-depth-stride-px", type=int, default=None)
    parser.add_argument("--roomseg-disable-corridor-cuts", action="store_true", default=False)
    parser.add_argument("--roomseg-disable-doorway-cuts", action="store_true", default=False)
    parser.add_argument("--roomseg-disable-wall-completion", action="store_true", default=False)
    parser.add_argument(
        "--roomseg-backend",
        default=None,
        choices=[
            "rose2_source_form",
            "rose2_source_form_v2",
            "rose2_source_faithful_v1",
            "rose2_source_external",
            "rose2_source_external_runner",
            "legacy_rose2_style_debug",
            ONLINE_ROSE_STYLE_BACKEND,
            ONLINE_WATERSHED_ROOMSEG_BACKEND,
            VERTICAL_FREE_ROOMSEG_BACKEND,
            VERTICAL_FREE_ROOMSEG_ALGORITHM,
            VERTICAL_FREE_GAP_CLOSURE_BACKEND,
        ],
    )
    parser.add_argument("--debug-rose2-source", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rose2-source-work-dir", default=None)
    parser.add_argument("--rose2-compare-legacy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--roomseg-finalization-mode",
        default=None,
        choices=[
            "no_merge",
            "proposal_only",
            "premerge_proposals",
            "doorway_constrained_merge",
            "no_merge_until_source_backend_verified",
            "no_merge_until_geometry_verified",
        ],
    )
    parser.add_argument("--enable-roomseg-nav-free-overlay", dest="roomseg_nav_free_overlay", action="store_true", default=None)
    parser.add_argument("--disable-roomseg-nav-free-overlay", dest="roomseg_nav_free_overlay", action="store_false")
    parser.add_argument("--enable-frontier-room-known-free-side", dest="frontier_room_known_free_side", action="store_true", default=None)
    parser.add_argument("--disable-frontier-room-known-free-side", dest="frontier_room_known_free_side", action="store_false")
    parser.add_argument("--roomseg-wall-gating-fix", action="store_true", default=None)
    parser.add_argument("--room-label-backend", default=None, choices=["vlm", "deterministic_debug", "unavailable"])
    parser.add_argument("--room-label-min-confidence", type=float, default=None)
    parser.add_argument("--room-label-ambiguity-margin", type=float, default=None)
    parser.add_argument("--room-label-min-reliable-objects", type=int, default=None)
    parser.add_argument("--room-label-unknown-category", default=None)
    parser.add_argument("--max-room-objects-in-prompt", type=int, default=None)
    parser.add_argument("--detection-localization", default=None, choices=["static_map_ray", "map_ray", "rgb_map_ray", "depth", "none"])
    parser.add_argument("--min-depth-points-per-detection", type=int, default=None)
    parser.add_argument("--segmenter", default=None, choices=["none", "auto", "sam2"])
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-cfg", default=None)
    parser.add_argument("--sam2-device", default=None)
    parser.add_argument("--max-detections-per-frame", type=int, default=None)
    parser.add_argument("--object-merge-radius-m", type=float, default=None)
    parser.add_argument("--instance-merge-distance-m", type=float, default=None)
    parser.add_argument("--instance-merge-iou-3d", type=float, default=None)
    parser.add_argument("--partial-class-weight", type=float, default=None)
    parser.add_argument("--min-geometry-confidence", type=float, default=None)
    parser.add_argument("--partial-stability-min-observations", type=int, default=None)
    parser.add_argument("--mask-iou-association-threshold", type=float, default=None)
    parser.add_argument("--mask-containment-track-match-threshold", type=float, default=None)
    parser.add_argument("--footprint-iou-association-threshold", type=float, default=None)
    parser.add_argument("--child-containment-threshold", type=float, default=None)
    parser.add_argument("--child-area-ratio-threshold", type=float, default=None)
    parser.add_argument("--frontier-distance-weight", type=float, default=None)
    parser.add_argument("--frontier-scenegraph-score-norm", "--frontier_scenegraph_score_norm", default=None, choices=["none", "minmax", "zscore"])
    parser.add_argument("--frontier-selection-mode", "--frontier_selection_mode", default=None, choices=["sgnav", "nearest", "random"])
    parser.add_argument("--frontier-random-seed", "--frontier_random_seed", type=int, default=None)
    parser.add_argument("--semantic-priors-path", "--semantic_priors_path", default=None)
    parser.add_argument("--debug-graph-dump", "--debug_graph_dump", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debug-graph-dump-dir", "--debug_graph_dump_dir", default=None)
    parser.add_argument("--runtime-planning-clearance-m", type=float, default=None)
    parser.add_argument("--candidate-min-detector-hits", type=int, default=None)
    parser.add_argument("--candidate-start-min-confidence", type=float, default=None)
    parser.add_argument("--candidate-start-min-hits", "--candidate_start_min_hits", type=int, default=None)
    parser.add_argument("--candidate-recent-max-age-steps", "--candidate_recent_max_age_steps", type=int, default=None)
    parser.add_argument("--candidate-match-substring", "--candidate_match_substring", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--candidate-accept-requires-reperception",
        "--candidate_accept_requires_reperception",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--candidate-reject-ttl-steps", "--candidate_reject_ttl_steps", type=int, default=None)
    parser.add_argument("--candidate-accept-threshold", "--candidate_accept_threshold", type=float, default=None)
    parser.add_argument("--candidate-stop-distance-m", type=float, default=None)
    parser.add_argument("--candidate-standoff-min-m", type=float, default=None)
    parser.add_argument("--candidate-standoff-max-m", type=float, default=None)
    parser.add_argument("--candidate-standoff-max-cells", "--candidate_standoff_max_cells", type=int, default=None)
    parser.add_argument("--candidate-standoff-ideal-m", "--candidate_standoff_ideal_m", type=float, default=None)
    parser.add_argument("--reperception-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reperception-min-observations", type=int, default=None)
    parser.add_argument("--reperception-max-steps", type=int, default=None)
    parser.add_argument("--reperception-same-goal-radius-m", type=float, default=None)
    parser.add_argument("--reperception-turn-wz-radps", type=float, default=None)
    parser.add_argument("--stop-verification-steps", type=int, default=None)
    parser.add_argument("--stop-verification-min-hits", type=int, default=None)
    parser.add_argument("--found-goal-stop-distance-m", type=float, default=None)
    parser.add_argument("--require-sgnav-stop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--llm-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-key", default=None)
    parser.add_argument("--llm-timeout-s", type=float, default=None)
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument("--max-hcot-subgraphs-per-decision", type=int, default=None)
    parser.add_argument("--vllm-frontier-scoring", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--vllm-timeout-s", type=float, default=None)
    parser.add_argument("--vllm-temperature", type=float, default=None)
    parser.add_argument("--vllm-max-frontiers", type=int, default=None)
    parser.add_argument("--vllm-image-scoring", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--vllm-image-max-width", type=int, default=None)
    parser.add_argument("--vllm-image-jpeg-quality", type=int, default=None)
    parser.add_argument("--score-frontiers-before-candidate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed-gt-object-memory", action="store_true", default=None)
    parser.add_argument("--no-seed-gt-object-memory", dest="seed_gt_object_memory", action="store_false")
    parser.add_argument("--allow-gt-goal-fallback", action="store_true", default=None)
    parser.add_argument("--no-allow-gt-goal-fallback", dest="allow_gt_goal_fallback", action="store_false")
    parser.add_argument("--sgnav-viz", action="store_true", default=None)
    parser.add_argument("--no-sgnav-viz", dest="sgnav_viz", action="store_false")
    parser.add_argument("--sgnav-viz-every-steps", type=int, default=None)
    parser.add_argument("--sgnav-viz-save-dir", default=None)
    parser.add_argument("--sgnav-viz-save-every-steps", type=int, default=None)
    parser.add_argument("--sgnav-viz-width", type=int, default=None)
    parser.add_argument("--sgnav-viz-height", type=int, default=None)
    parser.add_argument("--sgnav-viz-jpeg-quality", type=int, default=None)
    parser.add_argument("--debug-overlay-layers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-overlay-layer-metadata", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-gt-goal-cells", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-room-proposals", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-room-masks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-room-labels", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-frontier-member-cells", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-object-nodes", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-candidate-markers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-green-like-primitives-before-warning", type=int, default=None)
    parser.add_argument("--hold-open", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    args.planner = args.planner or get_nested(cfg, "repo.planner", "astar")
    args.detector = args.detector or get_nested(cfg, "repo.detector", "dry_run")
    args.yolo_world_model = args.yolo_world_model or get_nested(cfg, "paths.yolo_world_model", get_nested(cfg, "perception.yolo_world_model", "data/models/yolov8l-worldv2.pt"))
    args.grounding_dino_checkpoint = args.grounding_dino_checkpoint or os.environ.get("GROUNDING_DINO_CHECKPOINT") or get_nested(
        cfg,
        "paths.grounding_dino_checkpoint",
        get_nested(cfg, "perception.grounding_dino.checkpoint", "data/models/groundingdino_swinb_cogcoor.pth"),
    )
    args.grounding_dino_config = args.grounding_dino_config or os.environ.get("GROUNDING_DINO_CONFIG") or get_nested(
        cfg,
        "paths.grounding_dino_config",
        get_nested(cfg, "perception.grounding_dino.config", ""),
    )
    args.grounding_dino_text_threshold = float(
        args.grounding_dino_text_threshold
        if args.grounding_dino_text_threshold is not None
        else get_nested(cfg, "perception.grounding_dino.text_threshold", 0.25)
    )
    args.grounding_dino_device = str(
        args.grounding_dino_device
        if args.grounding_dino_device is not None
        else get_nested(cfg, "perception.grounding_dino.device", "cuda")
    )
    args.min_valid_detection_confidence = float(
        args.min_valid_detection_confidence
        if args.min_valid_detection_confidence is not None
        else get_nested(cfg, "perception.min_valid_detection_confidence", MIN_VALID_DETECTION_CONFIDENCE)
    )
    args.detector_conf = max(
        float(args.detector_conf if args.detector_conf is not None else get_nested(cfg, "perception.confidence_threshold", 0.7)),
        float(args.min_valid_detection_confidence),
    )
    args.detector_iou = float(args.detector_iou if args.detector_iou is not None else get_nested(cfg, "perception.nms_iou_threshold", 0.5))
    args.reject_edge_touching_bboxes = bool(
        args.reject_edge_touching_bboxes
        if args.reject_edge_touching_bboxes is not None
        else get_nested(cfg, "perception.yolo_world.reject_edge_touching_bboxes", True)
    )
    args.bbox_edge_margin_px = float(
        args.bbox_edge_margin_px
        if args.bbox_edge_margin_px is not None
        else get_nested(cfg, "perception.yolo_world.bbox_edge_margin_px", 2)
    )
    args.bbox_edge_margin_ratio = float(
        args.bbox_edge_margin_ratio
        if args.bbox_edge_margin_ratio is not None
        else get_nested(cfg, "perception.yolo_world.bbox_edge_margin_ratio", 0.0)
    )
    args.partial_class_weight = float(
        args.partial_class_weight
        if args.partial_class_weight is not None
        else get_nested(cfg, "object_memory.partial_class_weight", 0.25)
    )
    args.min_geometry_confidence = float(
        args.min_geometry_confidence
        if args.min_geometry_confidence is not None
        else get_nested(cfg, "object_memory.min_geometry_confidence", 0.50)
    )
    args.partial_stability_min_observations = int(
        args.partial_stability_min_observations
        if args.partial_stability_min_observations is not None
        else get_nested(cfg, "object_memory.partial_stability_min_observations", 3)
    )
    args.mask_iou_association_threshold = float(
        args.mask_iou_association_threshold
        if args.mask_iou_association_threshold is not None
        else get_nested(cfg, "object_memory.mask_iou_association_threshold", get_nested(cfg, "object_memory.mask_iou_track_match_threshold", 0.25))
    )
    args.mask_containment_track_match_threshold = float(
        args.mask_containment_track_match_threshold
        if args.mask_containment_track_match_threshold is not None
        else get_nested(cfg, "object_memory.mask_containment_track_match_threshold", 0.60)
    )
    args.footprint_iou_association_threshold = float(
        args.footprint_iou_association_threshold
        if args.footprint_iou_association_threshold is not None
        else get_nested(cfg, "object_memory.footprint_iou_association_threshold", get_nested(cfg, "object_memory.footprint_iou_track_match_threshold", 0.20))
    )
    args.child_containment_threshold = float(
        args.child_containment_threshold
        if args.child_containment_threshold is not None
        else get_nested(cfg, "object_memory.child_containment_threshold", 0.70)
    )
    args.child_area_ratio_threshold = float(
        args.child_area_ratio_threshold
        if args.child_area_ratio_threshold is not None
        else get_nested(cfg, "object_memory.child_area_ratio_threshold", get_nested(cfg, "object_memory.child_object_area_ratio_max", 0.35))
    )
    args.headless = bool(get_nested(cfg, "isaac.headless", True) if args.headless is None else args.headless)
    viz_cfg = get_nested(cfg, "visualization.sgnav_popup", "auto")
    if args.sgnav_viz is None:
        if str(viz_cfg).strip().lower() == "auto":
            args.sgnav_viz = not args.headless
        else:
            args.sgnav_viz = bool(str_to_bool(viz_cfg))
    args.sgnav_viz_every_steps = int(args.sgnav_viz_every_steps or get_nested(cfg, "visualization.sgnav_popup_every_steps", 20))
    args.sgnav_viz_save_dir = args.sgnav_viz_save_dir or get_nested(cfg, "visualization.sgnav_popup_save_dir", None)
    args.sgnav_viz_save_every_steps = int(args.sgnav_viz_save_every_steps or get_nested(cfg, "visualization.sgnav_popup_save_every_steps", 10))
    args.sgnav_viz_width = int(args.sgnav_viz_width or get_nested(cfg, "visualization.sgnav_popup_width", 1440))
    args.sgnav_viz_height = int(args.sgnav_viz_height or get_nested(cfg, "visualization.sgnav_popup_height", 900))
    args.sgnav_viz_jpeg_quality = int(args.sgnav_viz_jpeg_quality or get_nested(cfg, "visualization.sgnav_popup_jpeg_quality", 75))
    args.debug_overlay_layers = bool(
        args.debug_overlay_layers
        if args.debug_overlay_layers is not None
        else get_nested(cfg, "visualization.debug_overlay_layers", True)
    )
    args.save_overlay_layer_metadata = bool(
        args.save_overlay_layer_metadata
        if args.save_overlay_layer_metadata is not None
        else get_nested(cfg, "visualization.save_overlay_layer_metadata", True)
    )
    args.show_gt_goal_cells = bool(
        args.show_gt_goal_cells
        if args.show_gt_goal_cells is not None
        else get_nested(cfg, "visualization.show_gt_goal_cells", False)
    )
    args.show_room_proposals = bool(
        args.show_room_proposals
        if args.show_room_proposals is not None
        else get_nested(cfg, "visualization.show_room_proposals", True)
    )
    args.show_room_masks = bool(
        args.show_room_masks
        if args.show_room_masks is not None
        else get_nested(cfg, "visualization.show_room_masks", True)
    )
    args.show_room_labels = bool(
        args.show_room_labels
        if args.show_room_labels is not None
        else get_nested(cfg, "visualization.show_room_labels", True)
    )
    args.show_frontier_member_cells = bool(
        args.show_frontier_member_cells
        if args.show_frontier_member_cells is not None
        else get_nested(cfg, "visualization.show_frontier_member_cells", True)
    )
    args.show_object_nodes = bool(
        args.show_object_nodes
        if args.show_object_nodes is not None
        else get_nested(cfg, "visualization.show_object_nodes", True)
    )
    args.show_candidate_markers = bool(
        args.show_candidate_markers
        if args.show_candidate_markers is not None
        else get_nested(cfg, "visualization.show_candidate_markers", True)
    )
    args.max_green_like_primitives_before_warning = int(
        args.max_green_like_primitives_before_warning
        if args.max_green_like_primitives_before_warning is not None
        else get_nested(cfg, "visualization.max_green_like_primitives_before_warning", 200)
    )
    args.sgnav_mode = str(args.sgnav_mode or get_nested(cfg, "sgnav.mode", "legacy")).strip().lower()
    args.strict_benchmark = bool(
        args.strict_benchmark
        if args.strict_benchmark is not None
        else get_nested(cfg, "benchmark.strict_benchmark", True)
    )
    args.allow_debug_fallbacks = bool(
        args.allow_debug_fallbacks
        if args.allow_debug_fallbacks is not None
        else get_nested(cfg, "benchmark.allow_debug_fallbacks", False)
    )
    args.policy = args.policy or get_nested(cfg, "benchmark.policy", None)
    args.ablation_name = args.ablation_name or get_nested(cfg, "benchmark.ablation_name", None)
    args.sim_backend = args.sim_backend or get_nested(cfg, "repo.backend", "map")
    args.output = args.output or str(Path(get_nested(cfg, "project.output_dir", "data/isaac_bench_runs")) / "run_one_episode" / "results.jsonl")
    args.sgnav_repo = args.sgnav_repo or get_nested(cfg, "paths.sgnav_repo", "/home/echo/SG-Nav")
    if args.use_original_scenegraph is None:
        args.use_original_scenegraph = bool(get_nested(cfg, "repo.use_original_scenegraph", False))
    args.max_control_steps = int(args.max_control_steps or get_nested(cfg, "episodes.max_steps", 500))
    args.control_dt = float(args.control_dt or get_nested(cfg, "isaac.control_dt", 0.2))
    args.replan_every_steps = int(args.replan_every_steps or get_nested(cfg, "astar.replan_every_steps", 5))
    args.perception_every_steps = int(args.perception_every_steps or get_nested(cfg, "isaac.perception_every_steps", 10))
    args.observed_radius_m = float(args.observed_radius_m or get_nested(cfg, "isaac.observed_radius_m", 3.0))
    args.lookahead_m = float(args.lookahead_m or get_nested(cfg, "astar.waypoint_lookahead_m", 0.15))
    args.max_vx_mps = float(args.max_vx_mps if args.max_vx_mps is not None else get_nested(cfg, "robot.max_vx_mps", 0.15))
    args.max_vy_mps = float(args.max_vy_mps if args.max_vy_mps is not None else get_nested(cfg, "robot.max_vy_mps", 0.15))
    args.max_wz_radps = float(args.max_wz_radps if args.max_wz_radps is not None else get_nested(cfg, "robot.max_wz_radps", 0.35))
    args.render_updates_per_step = int(args.render_updates_per_step or get_nested(cfg, "isaac.render_updates_per_step", 2))
    args.isaac_width = int(args.isaac_width or get_nested(cfg, "camera.width", 640))
    args.isaac_height = int(args.isaac_height or get_nested(cfg, "camera.height", 480))
    args.camera_hfov_deg = float(args.camera_hfov_deg or get_nested(cfg, "camera.hfov_deg", 110.0))
    args.camera_mast_height_m = float(args.camera_mast_height_m or get_nested(cfg, "camera.mast_height_m", 1.35))
    args.camera_forward_offset_m = float(args.camera_forward_offset_m if args.camera_forward_offset_m is not None else get_nested(cfg, "camera.forward_offset_m", 0.0))
    args.camera_pitch_deg = float(args.camera_pitch_deg if args.camera_pitch_deg is not None else get_nested(cfg, "camera.pitch_deg", 0.0))
    args.camera_near_m = float(args.camera_near_m if args.camera_near_m is not None else get_nested(cfg, "camera.near_m", 0.02))
    args.camera_far_m = float(args.camera_far_m if args.camera_far_m is not None else get_nested(cfg, "camera.far_m", 5.0))
    args.camera_annotator_device = str(args.camera_annotator_device or get_nested(cfg, "isaac.camera_annotator_device", "cuda")).strip().lower()
    args.read_depth = bool(args.read_depth if args.read_depth is not None else get_nested(cfg, "isaac.read_depth", False))
    args.nearfield_depth = bool(args.nearfield_depth if args.nearfield_depth is not None else get_nested(cfg, "nearfield_depth.enabled", False))
    args.nearfield_width = int(args.nearfield_width if args.nearfield_width is not None else get_nested(cfg, "nearfield_depth.width", 192))
    args.nearfield_height = int(args.nearfield_height if args.nearfield_height is not None else get_nested(cfg, "nearfield_depth.height", 192))
    args.nearfield_hfov_deg = float(args.nearfield_hfov_deg if args.nearfield_hfov_deg is not None else get_nested(cfg, "nearfield_depth.hfov_deg", 115.0))
    args.nearfield_height_m = float(args.nearfield_height_m if args.nearfield_height_m is not None else get_nested(cfg, "nearfield_depth.height_m", 1.15))
    args.nearfield_near_m = float(args.nearfield_near_m if args.nearfield_near_m is not None else get_nested(cfg, "nearfield_depth.near_m", 0.02))
    args.nearfield_far_m = float(args.nearfield_far_m if args.nearfield_far_m is not None else get_nested(cfg, "nearfield_depth.far_m", 1.8))
    args.nearfield_radius_m = float(args.nearfield_radius_m if args.nearfield_radius_m is not None else get_nested(cfg, "nearfield_depth.radius_m", 1.2))
    args.nearfield_ignore_radius_m = float(args.nearfield_ignore_radius_m if args.nearfield_ignore_radius_m is not None else get_nested(cfg, "nearfield_depth.ignore_radius_m", 0.14))
    args.nearfield_depth_stride_px = int(args.nearfield_depth_stride_px if args.nearfield_depth_stride_px is not None else get_nested(cfg, "nearfield_depth.depth_stride_px", 3))
    args.nearfield_floor_tolerance_m = float(args.nearfield_floor_tolerance_m if args.nearfield_floor_tolerance_m is not None else get_nested(cfg, "nearfield_depth.floor_tolerance_m", 0.12))
    args.nearfield_obstacle_min_height_m = float(args.nearfield_obstacle_min_height_m if args.nearfield_obstacle_min_height_m is not None else get_nested(cfg, "nearfield_depth.obstacle_min_height_m", 0.18))
    args.nearfield_obstacle_max_height_m = float(args.nearfield_obstacle_max_height_m if args.nearfield_obstacle_max_height_m is not None else get_nested(cfg, "nearfield_depth.obstacle_max_height_m", 0.90))
    args.nearfield_splat_point_threshold = int(args.nearfield_splat_point_threshold if args.nearfield_splat_point_threshold is not None else get_nested(cfg, "nearfield_depth.splat_point_threshold", 2))
    args.nearfield_free_splat_point_threshold = int(args.nearfield_free_splat_point_threshold if args.nearfield_free_splat_point_threshold is not None else get_nested(cfg, "nearfield_depth.free_splat_point_threshold", 1))
    args.static_nearfield_map = bool(
        args.static_nearfield_map
        if args.static_nearfield_map is not None
        else get_nested(cfg, "nearfield_static_map.enabled", False)
    )
    args.static_nearfield_radius_m = float(
        args.static_nearfield_radius_m
        if args.static_nearfield_radius_m is not None
        else get_nested(cfg, "nearfield_static_map.radius_m", 1.0)
    )
    args.panorama_steps = int(args.panorama_steps if args.panorama_steps is not None else get_nested(cfg, "sgnav.panorama_steps", 8))
    args.panorama_wz_radps = float(args.panorama_wz_radps if args.panorama_wz_radps is not None else get_nested(cfg, "sgnav.panorama_wz_radps", 0.8))
    args.panorama_render_updates_per_step = int(args.panorama_render_updates_per_step if args.panorama_render_updates_per_step is not None else get_nested(cfg, "sgnav.panorama_render_updates_per_step", get_nested(cfg, "isaac.render_updates_per_step", 1)))
    args.depth_max_m = float(args.depth_max_m if args.depth_max_m is not None else get_nested(cfg, "mapping.depth_max_m", 5.0))
    args.depth_min_m = float(args.depth_min_m if args.depth_min_m is not None else get_nested(cfg, "mapping.depth_min_m", 0.20))
    args.depth_stride_px = int(args.depth_stride_px if args.depth_stride_px is not None else get_nested(cfg, "mapping.depth_stride_px", 8))
    args.online_map_size_m = float(args.online_map_size_m if args.online_map_size_m is not None else get_nested(cfg, "mapping.map_size_m", 40.0))
    args.online_resolution_m = float(args.online_resolution_m if args.online_resolution_m is not None else get_nested(cfg, "mapping.online_resolution_m", get_nested(cfg, "scene_preprocess.map_resolution_m", 0.05)))
    args.obstacle_min_height_m = float(args.obstacle_min_height_m if args.obstacle_min_height_m is not None else get_nested(cfg, "mapping.obstacle_min_height_m", 0.20))
    args.obstacle_max_height_m = float(args.obstacle_max_height_m if args.obstacle_max_height_m is not None else get_nested(cfg, "mapping.obstacle_max_height_m", 1.50))
    args.free_min_height_m = float(args.free_min_height_m if args.free_min_height_m is not None else get_nested(cfg, "mapping.free_min_height_m", -1.50))
    args.free_max_height_m = float(args.free_max_height_m if args.free_max_height_m is not None else get_nested(cfg, "mapping.free_max_height_m", 0.10))
    args.splat_point_threshold = int(args.splat_point_threshold if args.splat_point_threshold is not None else get_nested(cfg, "mapping.splat_point_threshold", 6))
    args.free_splat_point_threshold = int(
        args.free_splat_point_threshold
        if args.free_splat_point_threshold is not None
        else get_nested(cfg, "mapping.free_splat_point_threshold", 1)
    )
    args.mapping_debug = bool(args.mapping_debug if args.mapping_debug is not None else get_nested(cfg, "mapping.debug", False))
    args.frontier_min_cluster_size = int(args.frontier_min_cluster_size if args.frontier_min_cluster_size is not None else get_nested(cfg, "mapping.frontier_min_cluster_size", 1))
    args.frontier_min_distance_m = float(args.frontier_min_distance_m if args.frontier_min_distance_m is not None else get_nested(cfg, "mapping.frontier_min_distance_m", 1.0))
    args.frontier_max_count = int(args.frontier_max_count if args.frontier_max_count is not None else get_nested(cfg, "mapping.frontier_max_count", 0))
    args.frontier_obstacle_dilation_radius_cells = int(
        args.frontier_obstacle_dilation_radius_cells
        if args.frontier_obstacle_dilation_radius_cells is not None
        else get_nested(cfg, "mapping.frontier_obstacle_dilation_radius_cells", 4)
    )
    args.frontier_unknown_dilation_radius_cells = int(
        args.frontier_unknown_dilation_radius_cells
        if args.frontier_unknown_dilation_radius_cells is not None
        else get_nested(cfg, "mapping.frontier_unknown_dilation_radius_cells", 1)
    )
    args.frontier_unknown_source = str(args.frontier_unknown_source or get_nested(cfg, "mapping.frontier_unknown_source", "observed"))
    if args.frontier_unknown_source not in {"observed", "implicit"}:
        raise ValueError("mapping.frontier_unknown_source must be 'observed' or 'implicit'")
    args.frontier_cluster_distance_mode = str(
        args.frontier_cluster_distance_mode or get_nested(cfg, "mapping.frontier_cluster_distance_mode", "mean")
    )
    if args.frontier_cluster_distance_mode not in {"mean", "min", "center"}:
        raise ValueError("mapping.frontier_cluster_distance_mode must be 'mean', 'min', or 'center'")
    args.frontier_allow_near_fallback = bool(
        args.frontier_allow_near_fallback
        if args.frontier_allow_near_fallback is not None
        else get_nested(cfg, "mapping.frontier_allow_near_fallback", False)
    )
    args.frontier_debug_dump = bool(
        args.frontier_debug_dump
        if args.frontier_debug_dump is not None
        else get_nested(cfg, "mapping.frontier_debug_dump", False)
    )
    args.frontier_debug_dir = str(args.frontier_debug_dir or get_nested(cfg, "mapping.frontier_debug_dir", "data/debug_frontier"))
    args.frontier_commitment_enabled = bool(
        args.frontier_commitment_enabled
        if args.frontier_commitment_enabled is not None
        else get_nested(cfg, "sgnav.frontier_commitment_enabled", True)
    )
    args.frontier_commit_match_radius_m = float(
        args.frontier_commit_match_radius_m
        if args.frontier_commit_match_radius_m is not None
        else get_nested(cfg, "sgnav.frontier_commit_match_radius_m", 0.75)
    )
    args.frontier_commit_reached_radius_m = float(
        args.frontier_commit_reached_radius_m
        if args.frontier_commit_reached_radius_m is not None
        else get_nested(cfg, "sgnav.frontier_commit_reached_radius_m", 0.20)
    )
    args.frontier_commit_min_steps = int(
        args.frontier_commit_min_steps
        if args.frontier_commit_min_steps is not None
        else get_nested(cfg, "sgnav.frontier_commit_min_steps", 12)
    )
    args.frontier_commit_max_steps = int(
        args.frontier_commit_max_steps
        if args.frontier_commit_max_steps is not None
        else get_nested(cfg, "sgnav.frontier_commit_max_steps", 120)
    )
    args.frontier_commit_switch_margin = float(
        args.frontier_commit_switch_margin
        if args.frontier_commit_switch_margin is not None
        else get_nested(cfg, "sgnav.frontier_commit_switch_margin", 0.25)
    )
    args.frontier_commit_switch_ratio = float(
        args.frontier_commit_switch_ratio
        if args.frontier_commit_switch_ratio is not None
        else get_nested(cfg, "sgnav.frontier_commit_switch_ratio", 1.15)
    )
    args.frontier_commit_no_progress_steps = int(
        args.frontier_commit_no_progress_steps
        if args.frontier_commit_no_progress_steps is not None
        else get_nested(cfg, "sgnav.frontier_commit_no_progress_steps", 25)
    )
    args.frontier_commit_progress_min_delta_m = float(
        args.frontier_commit_progress_min_delta_m
        if args.frontier_commit_progress_min_delta_m is not None
        else get_nested(cfg, "sgnav.frontier_commit_progress_min_delta_m", 0.10)
    )
    args.frontier_blacklist_ttl_steps = int(
        args.frontier_blacklist_ttl_steps
        if args.frontier_blacklist_ttl_steps is not None
        else get_nested(cfg, "sgnav.frontier_blacklist_ttl_steps", 100)
    )
    default_robot_radius_m = get_nested(cfg, "robot.footprint_radius_m", None)
    if default_robot_radius_m is None:
        default_robot_radius_m = 0.5 * float(get_nested(cfg, "robot.footprint_width_m", 0.28))
    args.robot_radius_m = float(args.robot_radius_m if args.robot_radius_m is not None else default_robot_radius_m)
    args.online_inflation_radius_m = float(args.online_inflation_radius_m if args.online_inflation_radius_m is not None else get_nested(cfg, "mapping.inflation_radius_m", 0.0))
    args.room_map_mode = str(args.room_map_mode or get_nested(cfg, "mapping.room_map_mode", VERTICAL_FREE_GAP_CLOSURE_CONTEXT))
    args.room_segmentation_config = dict(get_nested(cfg, "mapping.room_segmentation", {}) or {})
    roomseg_debug_layers_cfg = dict(args.room_segmentation_config.get("debug_layers", {}) or {})
    roomseg_overlay_cfg = dict(args.room_segmentation_config.get("navigation_free_context_overlay", {}) or {})
    roomseg_frontier_context_cfg = dict(args.room_segmentation_config.get("frontier_room_context", {}) or {})
    roomseg_wall_gating_fix_cfg = dict(args.room_segmentation_config.get("wall_gating_fix", {}) or {})
    roomseg_online_cfg = dict(args.room_segmentation_config.get("online_roomseg", {}) or {})
    roomseg_watershed_cfg = dict(args.room_segmentation_config.get("online_watershed_roomseg", {}) or {})
    if args.roomseg_backend is not None:
        args.room_segmentation_config["backend"] = str(args.roomseg_backend)
    if args.debug_rose2_source is not None:
        args.room_segmentation_config["debug_rose2_source"] = bool(args.debug_rose2_source)
    if args.rose2_source_work_dir is not None:
        args.room_segmentation_config["rose2_source_work_dir"] = str(args.rose2_source_work_dir)
    if args.rose2_compare_legacy is not None:
        args.room_segmentation_config["rose2_compare_legacy"] = bool(args.rose2_compare_legacy)
    if args.roomseg_finalization_mode is not None:
        args.room_segmentation_config["finalization_mode"] = str(args.roomseg_finalization_mode)
    if args.debug_roomseg_layers is not None:
        roomseg_debug_layers_cfg["enabled"] = bool(args.debug_roomseg_layers)
    if args.debug_roomseg_dir is not None:
        roomseg_debug_layers_cfg["output_dir"] = str(args.debug_roomseg_dir)
    if args.debug_roomseg_max_saves is not None:
        roomseg_debug_layers_cfg["max_saves"] = int(args.debug_roomseg_max_saves)
    if args.roomseg_nav_free_overlay is not None:
        roomseg_overlay_cfg["enabled"] = bool(args.roomseg_nav_free_overlay)
    if args.frontier_room_known_free_side is not None:
        roomseg_frontier_context_cfg["use_known_free_side"] = bool(args.frontier_room_known_free_side)
        roomseg_frontier_context_cfg["enabled"] = bool(args.frontier_room_known_free_side)
    if args.roomseg_wall_gating_fix is not None:
        roomseg_wall_gating_fix_cfg["enabled"] = bool(args.roomseg_wall_gating_fix)
    if args.roomseg_roomseg_depth_stride_px is not None:
        top_depth_cfg = dict(args.room_segmentation_config.get("depth", {}) or {})
        top_depth_cfg["roomseg_depth_stride_px"] = int(args.roomseg_roomseg_depth_stride_px)
        args.room_segmentation_config["depth"] = top_depth_cfg
        depth_cfg = dict(roomseg_online_cfg.get("depth", {}) or {})
        depth_cfg["roomseg_depth_stride_px"] = int(args.roomseg_roomseg_depth_stride_px)
        roomseg_online_cfg["depth"] = depth_cfg
        watershed_depth_cfg = dict(roomseg_watershed_cfg.get("depth", {}) or {})
        watershed_depth_cfg["roomseg_depth_stride_px"] = int(args.roomseg_roomseg_depth_stride_px)
        roomseg_watershed_cfg["depth"] = watershed_depth_cfg
    if args.roomseg_disable_corridor_cuts:
        neck_cfg = dict(roomseg_online_cfg.get("corridor_room_neck_cut", {}) or {})
        neck_cfg["enabled"] = False
        roomseg_online_cfg["corridor_room_neck_cut"] = neck_cfg
    if args.roomseg_disable_doorway_cuts:
        doorway_cfg = dict(roomseg_online_cfg.get("doorway_virtual_cut", {}) or {})
        doorway_cfg["enabled"] = False
        roomseg_online_cfg["doorway_virtual_cut"] = doorway_cfg
    if args.roomseg_disable_wall_completion:
        physical_cfg = dict(roomseg_online_cfg.get("physical_wall_completion", {}) or {})
        physical_cfg["enabled"] = False
        roomseg_online_cfg["physical_wall_completion"] = physical_cfg
    args.room_segmentation_config["debug_layers"] = roomseg_debug_layers_cfg
    args.room_segmentation_config["navigation_free_context_overlay"] = roomseg_overlay_cfg
    args.room_segmentation_config["frontier_room_context"] = roomseg_frontier_context_cfg
    args.room_segmentation_config["wall_gating_fix"] = roomseg_wall_gating_fix_cfg
    args.room_segmentation_config["online_roomseg"] = roomseg_online_cfg
    args.room_segmentation_config["online_watershed_roomseg"] = roomseg_watershed_cfg
    args.debug_roomseg_layers = bool(roomseg_debug_layers_cfg.get("enabled", False))
    args.debug_roomseg_dir = str(roomseg_debug_layers_cfg.get("output_dir", "debug/roomseg_layers"))
    args.debug_roomseg_max_saves = int(roomseg_debug_layers_cfg.get("max_saves", 50))
    args.save_roomseg_snapshots = bool(args.save_roomseg_snapshots)
    args.roomseg_snapshot_dir = str(args.roomseg_snapshot_dir or "result/roomseg_snapshots")
    args.roomseg_snapshot_max_saves = int(args.roomseg_snapshot_max_saves if args.roomseg_snapshot_max_saves is not None else 500)
    if args.debug_roomseg_layers:
        for nested_key in ("vertical_free_roomseg", "vertical_free_gap_closure", "online_roomseg", "online_watershed_roomseg"):
            nested_cfg = dict(args.room_segmentation_config.get(nested_key, {}) or {})
            nested_cfg["debug_dump"] = True
            nested_cfg.setdefault("debug_dir", args.debug_roomseg_dir)
            if nested_key == "online_roomseg":
                debug_cfg = dict(nested_cfg.get("debug", {}) or {})
                debug_cfg["save_layers"] = True
                debug_cfg["save_candidate_json"] = True
                nested_cfg["debug"] = debug_cfg
            if nested_key == "online_watershed_roomseg":
                debug_cfg = dict(nested_cfg.get("debug", {}) or {})
                debug_cfg["save_layers"] = True
                debug_cfg["save_json"] = True
                nested_cfg["debug"] = debug_cfg
            args.room_segmentation_config[nested_key] = nested_cfg
    args.roomseg_backend = str(args.room_segmentation_config.get("backend", "rose2_source_external_runner"))
    args.debug_rose2_source = bool(args.room_segmentation_config.get("debug_rose2_source", False))
    args.rose2_source_work_dir = str(args.room_segmentation_config.get("rose2_source_work_dir", "debug/rose2_source"))
    args.rose2_compare_legacy = bool(args.room_segmentation_config.get("rose2_compare_legacy", False))
    args.roomseg_finalization_mode = str(args.room_segmentation_config.get("finalization_mode", "no_merge_until_source_backend_verified"))
    args.roomseg_nav_free_overlay = bool(roomseg_overlay_cfg.get("enabled", False))
    args.frontier_room_known_free_side = bool(roomseg_frontier_context_cfg.get("enabled", True) and roomseg_frontier_context_cfg.get("use_known_free_side", True))
    room_semantics_cfg = dict(get_nested(cfg, "room_semantics", {}) or {})
    for key in (
        "use_premerge_labels_for_open_plan_merge",
        "min_label_reliability_for_functional_split",
        "unknown_allows_functional_split",
        "final_label_after_merge",
    ):
        if key in room_semantics_cfg:
            args.room_segmentation_config.setdefault(key, room_semantics_cfg[key])
    room_node_cfg = dict(get_nested(cfg, "sgnav.scene_graph.room_nodes", {}) or {})
    args.room_label_backend = str(args.room_label_backend or room_node_cfg.get("room_label_backend", "vlm"))
    args.room_label_allowed_categories = list(room_node_cfg.get("allowed_room_categories", DEFAULT_ROOM_CATEGORIES) or DEFAULT_ROOM_CATEGORIES)
    args.room_label_min_confidence = float(
        args.room_label_min_confidence
        if args.room_label_min_confidence is not None
        else room_node_cfg.get("room_label_min_confidence", 0.60)
    )
    args.room_label_ambiguity_margin = float(
        args.room_label_ambiguity_margin
        if args.room_label_ambiguity_margin is not None
        else room_node_cfg.get("room_label_ambiguity_margin", 0.15)
    )
    args.room_label_min_reliable_objects = int(
        args.room_label_min_reliable_objects
        if args.room_label_min_reliable_objects is not None
        else room_node_cfg.get("room_label_min_reliable_objects", 2)
    )
    args.room_label_unknown_category = str(args.room_label_unknown_category or room_node_cfg.get("unknown_category", "unknown"))
    args.max_room_objects_in_prompt = int(
        args.max_room_objects_in_prompt
        if args.max_room_objects_in_prompt is not None
        else room_node_cfg.get("max_room_objects_in_prompt", 25)
    )
    args.detection_localization = str(args.detection_localization or get_nested(cfg, "perception.detection_localization", "static_map_ray"))
    args.min_depth_points_per_detection = int(args.min_depth_points_per_detection if args.min_depth_points_per_detection is not None else get_nested(cfg, "perception.min_depth_points_per_detection", 20))
    args.segmenter = str(args.segmenter or get_nested(cfg, "perception.segmenter", "none"))
    sam2_checkpoint_value = args.sam2_checkpoint if args.sam2_checkpoint is not None else get_nested(cfg, "perception.sam2_checkpoint", "")
    sam2_model_cfg_value = args.sam2_model_cfg if args.sam2_model_cfg is not None else get_nested(cfg, "perception.sam2_model_cfg", "facebook/sam2.1-hiera-tiny")
    args.sam2_checkpoint = "" if sam2_checkpoint_value is None else str(sam2_checkpoint_value)
    args.sam2_model_cfg = "facebook/sam2.1-hiera-tiny" if sam2_model_cfg_value is None else str(sam2_model_cfg_value)
    args.sam2_device = str(args.sam2_device or get_nested(cfg, "perception.sam2_device", "cuda"))
    args.max_detections_per_frame = int(args.max_detections_per_frame if args.max_detections_per_frame is not None else get_nested(cfg, "perception.max_detections_per_frame", 100))
    args.object_merge_radius_m = float(args.object_merge_radius_m if args.object_merge_radius_m is not None else get_nested(cfg, "perception.object_merge_radius_m", 0.5))
    args.instance_merge_distance_m = float(
        args.instance_merge_distance_m
        if args.instance_merge_distance_m is not None
        else get_nested(cfg, "sgnav.perception.instance_merge_distance_m", get_nested(cfg, "perception.object_merge_radius_m", 0.75))
    )
    args.instance_merge_iou_3d = float(
        args.instance_merge_iou_3d
        if args.instance_merge_iou_3d is not None
        else get_nested(cfg, "sgnav.perception.instance_merge_iou_3d", 0.15)
    )
    args.frontier_distance_weight = float(args.frontier_distance_weight if args.frontier_distance_weight is not None else get_nested(cfg, "sgnav.frontier_distance_weight", 0.2))
    args.frontier_scenegraph_score_norm = str(
        args.frontier_scenegraph_score_norm
        if args.frontier_scenegraph_score_norm is not None
        else get_nested(cfg, "sgnav.frontier_scenegraph_score_norm", "minmax")
    )
    args.frontier_selection_mode = str(
        args.frontier_selection_mode
        if args.frontier_selection_mode is not None
        else get_nested(cfg, "sgnav.frontier_selection_mode", "sgnav")
    )
    args.frontier_random_seed = int(
        args.frontier_random_seed
        if args.frontier_random_seed is not None
        else get_nested(cfg, "sgnav.frontier_random_seed", 0)
    )
    args.semantic_priors_path = str(args.semantic_priors_path or get_nested(cfg, "sgnav.semantic_priors_path", "isaac_bench/configs/sgnav_semantic_priors.yaml"))
    args.debug_graph_dump = bool(
        args.debug_graph_dump
        if args.debug_graph_dump is not None
        else get_nested(cfg, "sgnav.debug_graph_dump", False)
    )
    args.debug_graph_dump_dir = str(args.debug_graph_dump_dir or get_nested(cfg, "sgnav.debug_graph_dump_dir", "debug/graphs"))
    args.runtime_planning_clearance_m = float(args.runtime_planning_clearance_m if args.runtime_planning_clearance_m is not None else get_nested(cfg, "astar.runtime_planning_clearance_m", 0.0))
    args.candidate_min_detector_hits = int(args.candidate_min_detector_hits if args.candidate_min_detector_hits is not None else get_nested(cfg, "sgnav.candidate_min_detector_hits", 2))
    args.candidate_start_min_confidence = max(
        float(args.candidate_start_min_confidence if args.candidate_start_min_confidence is not None else get_nested(cfg, "sgnav.candidate_start_min_confidence", MIN_VALID_DETECTION_CONFIDENCE)),
        float(args.min_valid_detection_confidence),
    )
    args.candidate_start_min_hits = int(args.candidate_start_min_hits if args.candidate_start_min_hits is not None else get_nested(cfg, "sgnav.candidate_start_min_hits", 2))
    args.candidate_recent_max_age_steps = int(
        args.candidate_recent_max_age_steps
        if args.candidate_recent_max_age_steps is not None
        else get_nested(cfg, "sgnav.candidate_recent_max_age_steps", 30)
    )
    args.candidate_match_substring = bool(
        args.candidate_match_substring
        if args.candidate_match_substring is not None
        else get_nested(cfg, "sgnav.candidate_match_substring", False)
    )
    args.candidate_accept_requires_reperception = bool(
        args.candidate_accept_requires_reperception
        if args.candidate_accept_requires_reperception is not None
        else get_nested(cfg, "sgnav.candidate_accept_requires_reperception", True)
    )
    args.candidate_reject_ttl_steps = int(
        args.candidate_reject_ttl_steps
        if args.candidate_reject_ttl_steps is not None
        else get_nested(cfg, "sgnav.candidate_reject_ttl_steps", 80)
    )
    args.candidate_accept_threshold = float(
        args.candidate_accept_threshold
        if args.candidate_accept_threshold is not None
        else get_nested(cfg, "sgnav.candidate_accept_threshold", 0.65)
    )
    args.candidate_stop_distance_m = float(args.candidate_stop_distance_m if args.candidate_stop_distance_m is not None else get_nested(cfg, "sgnav.candidate_stop_distance_m", get_nested(cfg, "episodes.success_distance_m", 1.0)))
    args.candidate_standoff_min_m = float(args.candidate_standoff_min_m if args.candidate_standoff_min_m is not None else get_nested(cfg, "sgnav.candidate_standoff_min_m", 0.65))
    args.candidate_standoff_max_m = float(args.candidate_standoff_max_m if args.candidate_standoff_max_m is not None else get_nested(cfg, "sgnav.candidate_standoff_max_m", 1.80))
    args.candidate_standoff_max_cells = int(
        args.candidate_standoff_max_cells
        if args.candidate_standoff_max_cells is not None
        else get_nested(cfg, "sgnav.candidate_standoff_max_cells", 16)
    )
    args.candidate_standoff_ideal_m = float(
        args.candidate_standoff_ideal_m
        if args.candidate_standoff_ideal_m is not None
        else get_nested(cfg, "sgnav.candidate_standoff_ideal_m", 1.0)
    )
    args.reperception_enabled = bool(args.reperception_enabled if args.reperception_enabled is not None else get_nested(cfg, "sgnav.reperception_enabled", True))
    args.reperception_min_observations = int(args.reperception_min_observations if args.reperception_min_observations is not None else get_nested(cfg, "sgnav.reperception_min_observations", 3))
    args.reperception_max_steps = int(args.reperception_max_steps if args.reperception_max_steps is not None else get_nested(cfg, "sgnav.reperception_max_steps", 10))
    args.reperception_same_goal_radius_m = float(args.reperception_same_goal_radius_m if args.reperception_same_goal_radius_m is not None else get_nested(cfg, "sgnav.reperception_same_goal_radius_m", 0.8))
    args.reperception_turn_wz_radps = float(args.reperception_turn_wz_radps if args.reperception_turn_wz_radps is not None else get_nested(cfg, "sgnav.reperception_turn_wz_radps", 0.5))
    args.stop_verification_steps = int(args.stop_verification_steps if args.stop_verification_steps is not None else get_nested(cfg, "sgnav.stop_verification_steps", 4))
    args.stop_verification_min_hits = int(args.stop_verification_min_hits if args.stop_verification_min_hits is not None else get_nested(cfg, "sgnav.stop_verification_min_hits", 2))
    args.found_goal_stop_distance_m = float(args.found_goal_stop_distance_m if args.found_goal_stop_distance_m is not None else get_nested(cfg, "sgnav.found_goal_stop_distance_m", 0.35))
    args.require_sgnav_stop = bool(args.require_sgnav_stop if args.require_sgnav_stop is not None else get_nested(cfg, "episodes.success_requires_stop", True))
    args.llm_enabled = bool(args.llm_enabled if args.llm_enabled is not None else get_nested(cfg, "llm.enabled", True))
    args.llm_base_url = args.llm_base_url or get_nested(cfg, "llm.base_url", "http://127.0.0.1:8000/v1")
    args.llm_model = args.llm_model or get_nested(cfg, "llm.model", "qwen3-vl-8b-instruct")
    args.llm_api_key = args.llm_api_key or get_nested(cfg, "llm.api_key", "EMPTY")
    args.llm_timeout_s = float(args.llm_timeout_s if args.llm_timeout_s is not None else get_nested(cfg, "llm.timeout_s", 30.0))
    args.llm_temperature = float(args.llm_temperature if args.llm_temperature is not None else get_nested(cfg, "llm.temperature", 0.0))
    args.llm_max_tokens = int(args.llm_max_tokens if args.llm_max_tokens is not None else get_nested(cfg, "llm.max_tokens", 512))
    args.max_hcot_subgraphs_per_decision = int(
        args.max_hcot_subgraphs_per_decision
        if args.max_hcot_subgraphs_per_decision is not None
        else get_nested(cfg, "llm.max_hcot_subgraphs_per_decision", 8)
    )
    args.vllm_frontier_scoring = bool(args.vllm_frontier_scoring if args.vllm_frontier_scoring is not None else get_nested(cfg, "vllm.frontier_scoring", False))
    args.vllm_base_url = str(args.vllm_base_url or get_nested(cfg, "vllm.base_url", "http://127.0.0.1:8000/v1"))
    args.vllm_model = str(args.vllm_model or get_nested(cfg, "vllm.model", "qwen3-vl-8b-instruct"))
    args.vllm_timeout_s = float(args.vllm_timeout_s if args.vllm_timeout_s is not None else get_nested(cfg, "vllm.timeout_s", 8.0))
    args.vllm_temperature = float(args.vllm_temperature if args.vllm_temperature is not None else get_nested(cfg, "vllm.temperature", 0.0))
    args.vllm_max_frontiers = int(args.vllm_max_frontiers if args.vllm_max_frontiers is not None else get_nested(cfg, "vllm.max_frontiers", 32))
    args.vllm_image_scoring = bool(args.vllm_image_scoring if args.vllm_image_scoring is not None else get_nested(cfg, "vllm.image_scoring", True))
    args.vllm_image_max_width = int(args.vllm_image_max_width if args.vllm_image_max_width is not None else get_nested(cfg, "vllm.image_max_width", 640))
    args.vllm_image_jpeg_quality = int(args.vllm_image_jpeg_quality if args.vllm_image_jpeg_quality is not None else get_nested(cfg, "vllm.image_jpeg_quality", 75))
    args.score_frontiers_before_candidate = bool(
        args.score_frontiers_before_candidate
        if args.score_frontiers_before_candidate is not None
        else get_nested(cfg, "vllm.score_frontiers_before_candidate", args.vllm_frontier_scoring)
    )
    args.seed_gt_object_memory = bool(get_nested(cfg, "sgnav.seed_gt_object_memory", False) if args.seed_gt_object_memory is None else args.seed_gt_object_memory)
    args.allow_gt_goal_fallback = bool(get_nested(cfg, "sgnav.allow_gt_goal_fallback", False) if args.allow_gt_goal_fallback is None else args.allow_gt_goal_fallback)
    configured_success_distance = get_nested(cfg, "episodes.success_distance_m", None)
    args.success_distance_m = (
        float(args.success_distance_m)
        if args.success_distance_m is not None
        else (float(configured_success_distance) if configured_success_distance is not None else None)
    )
    args.roomseg_debug_only = bool(
        args.roomseg_debug_only
        if args.roomseg_debug_only is not None
        else get_nested(cfg, "debug.roomseg_debug_only", False)
    )
    requested_explore_until_no_frontiers = args.explore_until_no_frontiers
    policy_name = str(args.policy or "")
    debug_frontier_policy = policy_name in {"random_frontier_mask_probe", "nearest_frontier_mask_probe"}
    args.explore_until_no_frontiers = bool(
        requested_explore_until_no_frontiers
        if requested_explore_until_no_frontiers is not None
        else (
            debug_frontier_policy
            or bool(get_nested(cfg, "debug.explore_until_no_frontiers", False))
        )
    )
    if debug_frontier_policy:
        args.strict_benchmark = False
        args.allow_debug_fallbacks = True
        args.llm_enabled = False
        args.vllm_frontier_scoring = False
        args.vllm_image_scoring = False
        args.score_frontiers_before_candidate = False
        args.require_sgnav_stop = False
        args.frontier_allow_near_fallback = True
        if str(getattr(args, "detector", "none") or "none") == "none":
            args.segmenter = "none"
        if str(getattr(args, "room_label_backend", "unavailable") or "unavailable") == "unavailable":
            args.room_label_backend = "unavailable"
    if args.explore_until_no_frontiers:
        args.frontier_allow_near_fallback = True
    if args.roomseg_debug_only:
        args.strict_benchmark = False
        args.allow_debug_fallbacks = True
        args.detector = "none"
        args.segmenter = "none"
        args.llm_enabled = False
        args.vllm_frontier_scoring = False
        args.vllm_image_scoring = False
        args.score_frontiers_before_candidate = False
        args.seed_gt_object_memory = False
        args.allow_gt_goal_fallback = False
        args.require_sgnav_stop = False
        args.room_label_backend = "unavailable"
        args.sgnav_viz_every_steps = 1
        args.debug_roomseg_layers = True
        debug_layers_cfg = dict(getattr(args, "room_segmentation_config", {}).get("debug_layers", {}) or {})
        debug_layers_cfg["enabled"] = True
        debug_layers_cfg.setdefault("output_dir", str(getattr(args, "debug_roomseg_dir", "debug/roomseg_layers")))
        args.room_segmentation_config["debug_layers"] = debug_layers_cfg
        args.debug_roomseg_dir = str(debug_layers_cfg.get("output_dir", getattr(args, "debug_roomseg_dir", "debug/roomseg_layers")))
        for nested_key in ("vertical_free_roomseg", "vertical_free_gap_closure", "online_roomseg", "online_watershed_roomseg"):
            nested_cfg = dict(args.room_segmentation_config.get(nested_key, {}) or {})
            nested_cfg["debug_dump"] = True
            nested_cfg.setdefault("debug_dir", args.debug_roomseg_dir)
            if nested_key == "online_roomseg":
                debug_cfg = dict(nested_cfg.get("debug", {}) or {})
                debug_cfg["save_layers"] = True
                debug_cfg["save_candidate_json"] = True
                nested_cfg["debug"] = debug_cfg
            if nested_key == "online_watershed_roomseg":
                debug_cfg = dict(nested_cfg.get("debug", {}) or {})
                debug_cfg["save_layers"] = True
                debug_cfg["save_json"] = True
                nested_cfg["debug"] = debug_cfg
            args.room_segmentation_config[nested_key] = nested_cfg

    if args.planner == "nav2":
        from isaac_bench.navigation.nav2_client import Nav2NavigateToPoseClient

        Nav2NavigateToPoseClient()
    try:
        validate_strict_benchmark_assets(args)
    except BenchmarkAssetError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 2
    episodes = read_jsonl(args.episode_file)
    episode = apply_success_distance_override(episodes[int(args.episode_index)], args)
    if args.sim_backend == "isaac":
        row = run_episode_isaac_closed_loop(episode, args)
    else:
        row = run_episode_map_sim(episode, args)
    if not getattr(args, "_row_already_logged", False):
        row = make_jsonable(row)
        summary_row = final_log_row(row)
        logger = JsonlEpisodeLogger(args.output)
        logger.log(row)
        print(json.dumps(summary_row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
