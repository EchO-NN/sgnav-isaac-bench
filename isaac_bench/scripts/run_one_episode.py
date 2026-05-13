from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from isaac_bench.config import get_nested, load_config, str_to_bool
from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.dataset.episode_generator import read_jsonl
from isaac_bench.env.habitat_like_env import MapSimHabitatLikeEnv
from isaac_bench.graph.decision import SGNavDecision
from isaac_bench.graph.sgnav_scenegraph_adapter import SGNAV_ROOM_NAMES, SGNavSceneGraphAdapter
from isaac_bench.mapping.coordinate_transform import MapInfo, grid_to_world_xy, is_inside_grid, world_xy_to_grid
from isaac_bench.mapping.frontier import extract_frontiers
from isaac_bench.mapping.online_mapper import OnlineMapper
from isaac_bench.mapping.room_map_from_rooms_json import build_room_index_map, load_rooms
from isaac_bench.metrics.episode_logger import JsonlEpisodeLogger
from isaac_bench.metrics.evaluator import EpisodeEvaluator
from isaac_bench.navigation.astar import GridAStarPlanner
from isaac_bench.navigation.waypoint_follower import HolonomicWaypointFollower
from isaac_bench.perception.detection_types import Detection2D, Detection3D
from isaac_bench.perception.detector_ipc import SubprocessDetector
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


def maybe_build_detector(
    detector_name: str,
    model_path: str,
    categories: List[str],
    conf: float = 0.08,
    iou: float = 0.5,
    allow_ipc_fallback: bool = False,
):
    try:
        detector = build_detector(detector_name, model_path, conf=conf, iou=iou)
    except Exception as exc:
        if detector_name != "yolo_world" or not allow_ipc_fallback:
            raise
        print(
            "[detector-ipc] direct YOLO-World load failed in this process; using external SG-Nav env worker: %s"
            % exc,
            flush=True,
        )
        detector = SubprocessDetector(detector_name, model_path, conf=conf, iou=iou)
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
    conf = float(getattr(args, "detector_conf", 0.08))
    iou = float(getattr(args, "detector_iou", 0.5))
    key = (str(args.detector), str(args.yolo_world_model), conf, iou, bool(allow_ipc_fallback), tuple(categories))
    if getattr(args, "_detector_key", None) == key:
        return
    old_detector = getattr(args, "_detector_instance", None)
    if old_detector is not None and hasattr(old_detector, "close"):
        old_detector.close()
    args._detector_instance = None
    args._detector_key = None
    args._detector_instance = maybe_build_detector(
        args.detector,
        args.yolo_world_model,
        categories,
        conf=conf,
        iou=iou,
        allow_ipc_fallback=allow_ipc_fallback,
    )
    args._detector_key = key


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


def trim_path_to_current(path: List[Tuple[int, int]], current_grid: Tuple[int, int]) -> List[Tuple[int, int]]:
    for idx, cell in enumerate(path):
        if cell == current_grid:
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
    return [det for det in detections if float(det.confidence) >= threshold]


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

    scene_dir, static_map_info, _static_occupancy, static_navigable = load_preprocessed_for_episode(episode)
    static_navigable = apply_episode_planning_clearance(
        scene_dir,
        static_map_info,
        static_navigable,
        episode,
        runtime_planning_clearance_m=getattr(args, "runtime_planning_clearance_m", 0.0),
    )
    ensure_detector_loaded(args, scene_dir, allow_ipc_fallback=True)
    detector = getattr(args, "_detector_instance", None)
    if detector is not None:
        ensure_segmenter_loaded(args)
        segmenter = getattr(args, "_segmenter_instance", None)
    else:
        segmenter = None
    camera_annotator_device = str(getattr(args, "camera_annotator_device", "cuda")).strip().lower()
    if camera_annotator_device not in {"cpu", "cuda"}:
        camera_annotator_device = "cpu"

    metric_planner = GridAStarPlanner(static_navigable, static_map_info.resolution_m, allow_diagonal=True)
    evaluator = EpisodeEvaluator(episode, metric_planner)
    object_memory = ObjectMemory(merge_radius_m=float(args.object_merge_radius_m))
    if args.seed_gt_object_memory:
        print("[sgnav-loop] seed_gt_object_memory ignored for depth-online mapping", flush=True)
    seeded = 0
    scenegraph = SGNavSceneGraphAdapter(args.sgnav_repo, use_original=args.use_original_scenegraph)
    scenegraph.reset(episode["goal_category"])
    full_room_map = None
    scenegraph.update(object_memory, room_map=full_room_map)
    evaluator.num_scenegraph_updates += 1
    decision_policy = SGNavDecision(
        scenegraph,
        frontier_distance_weight=float(args.frontier_distance_weight),
        candidate_min_hits=int(args.candidate_min_detector_hits),
        candidate_start_min_confidence=float(args.candidate_start_min_confidence),
        candidate_stop_distance_m=float(args.candidate_stop_distance_m),
        candidate_standoff_min_m=float(args.candidate_standoff_min_m),
        candidate_standoff_max_m=float(args.candidate_standoff_max_m),
    )
    follower = HolonomicWaypointFollower(
        max_vx=float(args.max_vx_mps),
        max_vy=float(args.max_vy_mps),
        max_wz=float(args.max_wz_radps),
        lookahead_m=float(args.lookahead_m),
    )
    mapper = OnlineMapper(
        size_m=float(args.online_map_size_m),
        resolution_m=float(args.online_resolution_m),
        depth_max_m=float(args.depth_max_m),
        depth_min_m=float(args.depth_min_m),
        depth_stride_px=int(args.depth_stride_px),
        obstacle_min_height_m=float(args.obstacle_min_height_m),
        obstacle_max_height_m=float(args.obstacle_max_height_m),
        robot_radius_m=float(args.robot_radius_m),
        inflation_radius_m=float(args.online_inflation_radius_m),
    )
    static_goal_cells = [(int(r), int(c)) for r, c in episode["goal_regions_grid"]]
    start_pose = tuple(float(v) for v in episode["start_pose_world"])
    mapper.reset((float(start_pose[0]), float(start_pose[1])))
    dynamic_map_info = mapper.grid.map_info
    goal_cells = []
    for goal_r, goal_c in static_goal_cells:
        gx, gy = grid_to_world_xy(goal_r, goal_c, static_map_info)
        dyn_goal = world_xy_to_grid(gx, gy, dynamic_map_info)
        if is_inside_grid(dyn_goal[0], dyn_goal[1], dynamic_map_info):
            goal_cells.append(dyn_goal)
    max_steps = int(args.max_control_steps)
    replan_every = max(1, int(args.replan_every_steps))
    perception_every = max(1, int(args.perception_every_steps))
    success_distance = float(episode.get("success_distance_m", 1.0))
    current_path: List[Tuple[int, int]] = []
    full_path: List[Tuple[int, int]] = []
    failure_reason = None
    stop_called = False
    last_detections_2d: List[Detection2D] = []
    last_frontiers = []
    last_nav_decision = None
    last_dynamic_occupancy = mapper.grid.occupied.astype(bool)
    last_dynamic_navigable = mapper.traversible(unknown_is_obstacle=True)
    last_dynamic_observed = mapper.grid.observed.astype(bool)
    sgnav_viz_enabled = bool(getattr(args, "sgnav_viz", False))
    sgnav_viz_save_dir = getattr(args, "sgnav_viz_save_dir", None)
    detection_localization = str(getattr(args, "detection_localization", "static_map_ray")).strip().lower()
    if detection_localization in {"static_map_ray", "map_ray", "rgb_map_ray"}:
        print("[sgnav-loop] static-map detection localization disabled; using depth projection", flush=True)
        detection_localization = "depth"
    args.read_depth = True
    viz = None
    viz_requested = bool(sgnav_viz_enabled or sgnav_viz_save_dir)
    viz_every = max(1, int(getattr(args, "sgnav_viz_every_steps", 1)))
    detector_cuda_rgb = detector_can_use_cuda_rgb(detector, args.detector, camera_annotator_device)
    logged_detector_rgb_device = False

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
        ) if (sgnav_viz_enabled or sgnav_viz_save_dir) else None
        intr = CameraIntrinsics.from_hfov(int(args.isaac_width), int(args.isaac_height), float(args.camera_hfov_deg))

        def viz_rgb(current_obs: dict) -> np.ndarray:
            if current_obs.get("has_rgb") and current_obs.get("rgb_device") == "cpu":
                return current_obs["rgb"]
            return server.get_observation(read_rgb=True, read_depth=False, rgb_device="cpu")["rgb"]

        last_decision_mode = "init"
        last_decision_reason = ""
        goal_candidate_count = 0
        sam2_failure_logged = False
        for step in range(max_steps):
            pose = obs["pose_world"]
            if not obs.get("has_depth"):
                failure_reason = "depth_unavailable_for_online_mapping"
                break
            mapper.update(obs["depth"], intr, pose, obs["camera_pose_world"])
            dynamic_map_info = mapper.grid.map_info
            occupancy = mapper.grid.occupied.astype(bool)
            observed = mapper.grid.observed.astype(bool)
            navigable = mapper.traversible(unknown_is_obstacle=True)
            last_dynamic_occupancy = occupancy
            last_dynamic_navigable = navigable
            last_dynamic_observed = observed
            nav_planner = GridAStarPlanner(navigable, dynamic_map_info.resolution_m, allow_diagonal=True)
            current_grid = nav_planner.snap_to_free(world_xy_to_grid(float(pose[0]), float(pose[1]), dynamic_map_info))
            if current_grid is None:
                mapper.update_simple_radius(pose, radius_m=max(float(args.robot_radius_m) * 2.0, 0.8))
                occupancy = mapper.grid.occupied.astype(bool)
                observed = mapper.grid.observed.astype(bool)
                navigable = mapper.traversible(unknown_is_obstacle=True)
                last_dynamic_occupancy = occupancy
                last_dynamic_navigable = navigable
                last_dynamic_observed = observed
                nav_planner = GridAStarPlanner(navigable, dynamic_map_info.resolution_m, allow_diagonal=True)
                current_grid = nav_planner.snap_to_free(world_xy_to_grid(float(pose[0]), float(pose[1]), dynamic_map_info))
            if current_grid is None:
                failure_reason = "agent_off_navigable_map"
                break
            metric_grid = metric_planner.snap_to_free(world_xy_to_grid(float(pose[0]), float(pose[1]), static_map_info))
            if metric_grid is None:
                failure_reason = "agent_off_static_metric_map"
                break
            evaluator.update_pose(pose, metric_grid, collided=bool(obs.get("collided", False)))
            if evaluator.final_distance_to_goal <= success_distance:
                stop_called = True
                break

            if detector is not None and step % perception_every == 0:
                detector_rgb = obs["rgb"]
                used_cuda_rgb = False
                if detector_cuda_rgb:
                    if not (obs.get("has_rgb") and obs.get("rgb_device") == "cuda" and obs.get("rgb_gpu") is not None):
                        obs = server.get_observation(read_rgb=True, read_depth=False, rgb_device="cuda")
                    if obs.get("has_rgb") and obs.get("rgb_device") == "cuda" and obs.get("rgb_gpu") is not None:
                        detector_rgb = obs["rgb_gpu"]
                        used_cuda_rgb = True
                    else:
                        detector_cuda_rgb = False
                        detector_rgb = obs["rgb"]
                if used_cuda_rgb:
                    if not logged_detector_rgb_device:
                        print("[sgnav-loop] detector RGB input: Isaac CUDA annotator -> YOLO tensor", flush=True)
                        logged_detector_rgb_device = True
                elif detector_requires_rgb(detector, args.detector) and not logged_detector_rgb_device:
                    print("[sgnav-loop] detector RGB input: CPU fallback", flush=True)
                    logged_detector_rgb_device = True
                detections_2d = detector.detect(detector_rgb)
                detections_2d = filter_detections_by_confidence(detections_2d, float(args.detector_conf))
                if int(args.max_detections_per_frame) > 0:
                    detections_2d = detections_2d[: int(args.max_detections_per_frame)]
                if segmenter is not None and detections_2d:
                    try:
                        segment_rgb = obs["rgb"] if obs.get("rgb_device") == "cpu" else viz_rgb(obs)
                        detections_2d = segmenter.segment(segment_rgb, list(detections_2d))
                    except Exception as exc:
                        if str(getattr(args, "segmenter", "none")).strip().lower() == "sam2":
                            raise
                        if not sam2_failure_logged:
                            print("[sam2] segmentation failed; continuing with YOLO boxes only: %s" % exc, flush=True)
                            sam2_failure_logged = True
                        segmenter = None
                last_detections_2d = list(detections_2d)
                if detection_localization == "depth":
                    detections_3d = detections_to_3d(
                        detections_2d,
                        obs["depth"],
                        intr,
                        obs["camera_pose_world"],
                        depth_max_m=float(args.depth_max_m),
                        min_points=int(args.min_depth_points_per_detection),
                    )
                elif detection_localization in {"static_map_ray", "map_ray", "rgb_map_ray"}:
                    detections_3d = detections_to_3d_static_map_ray(
                        detections_2d,
                        obs["camera_pose_world"],
                        int(args.isaac_width),
                        float(args.camera_hfov_deg),
                        dynamic_map_info,
                        occupancy,
                        navigable,
                        max_range_m=float(args.depth_max_m),
                    )
                elif detection_localization == "none":
                    detections_3d = []
                else:
                    raise ValueError("Unsupported detection localization mode: %s" % detection_localization)
                object_memory.update(detections_3d, step_id=step, map_info=dynamic_map_info)
                evaluator.num_yolo_calls += 1
            if step % perception_every == 0:
                scenegraph.update(object_memory, room_map=None if full_room_map is None else observed_room_map(full_room_map, observed))
                evaluator.num_scenegraph_updates += 1

            needs_replan = not current_path or step % replan_every == 0
            if current_path:
                suffix = trim_path_to_current(current_path, current_grid)
                if suffix:
                    current_path = suffix
                else:
                    needs_replan = True

            if needs_replan:
                frontiers = extract_frontiers(
                    free=navigable,
                    observed=observed,
                    traversible=navigable,
                    map_info=dynamic_map_info,
                    agent_grid=current_grid,
                    min_cluster_size=3,
                    min_distance_m=0.5,
                    max_count=32,
                )
                last_frontiers = list(frontiers)
                nav_decision = decision_policy.choose_navigation_target(
                    object_memory,
                    episode["goal_category"],
                    current_grid,
                    frontiers,
                    nav_planner,
                    dynamic_map_info,
                    pose,
                    allow_frontier=True,
                )
                last_nav_decision = nav_decision
                evaluator.num_frontier_decisions += 1
                last_decision_mode = nav_decision.mode
                last_decision_reason = nav_decision.reason
                goal_norm = normalize_category(episode["goal_category"])
                goal_candidate_count = len(
                    [
                        node
                        for node in object_memory.nodes
                        if normalize_category(node.category) == goal_norm
                        or goal_norm in normalize_category(node.category)
                        or normalize_category(node.category) in goal_norm
                    ]
                )
                if nav_decision.stop:
                    if evaluator.final_distance_to_goal <= success_distance:
                        stop_called = True
                    else:
                        failure_reason = "sgnav_stop_outside_goal_region"
                    if viz is not None:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=navigable,
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
                nav_goals = nav_decision.target_cells
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
                            navigable=navigable,
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
                result = nav_planner.plan(current_grid, nav_goals)
                if not result.path:
                    failure_reason = "astar_no_path"
                    if viz is not None:
                        viz.update(
                            step=step,
                            rgb=viz_rgb(obs),
                            detections_2d=last_detections_2d,
                            occupancy=occupancy,
                            navigable=navigable,
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
                current_path = result.path
                full_path.extend(result.path)

            if viz is not None and step % viz_every == 0:
                viz.update(
                    step=step,
                    rgb=viz_rgb(obs),
                    detections_2d=last_detections_2d,
                    occupancy=occupancy,
                    navigable=navigable,
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

            path_world = path_cells_to_world(current_path[1: min(len(current_path), 20)], dynamic_map_info)
            if not path_world:
                if evaluator.final_distance_to_goal <= success_distance:
                    stop_called = True
                elif last_decision_mode == "candidate":
                    failure_reason = "candidate_standoff_reached_without_success"
                elif last_decision_mode == "frontier":
                    failure_reason = "frontier_reached_without_success"
                else:
                    failure_reason = "path_exhausted_without_success"
                if viz is not None:
                    viz.update(
                        step=step,
                        rgb=viz_rgb(obs),
                        detections_2d=last_detections_2d,
                        occupancy=occupancy,
                        navigable=navigable,
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
            failure_reason = "max_control_steps"

        row = evaluator.finish(stop_called=stop_called, planner=args.planner, detector=args.detector, failure_reason=failure_reason)
        row["sim_backend"] = "isaac"
        row["closed_loop"] = True
        row["control_mode"] = "kinematic_holonomic"
        row["seeded_object_memory_count"] = int(seeded)
        row["object_memory_count"] = int(len(object_memory.nodes))
        row["goal_candidate_count"] = int(goal_candidate_count)
        row["sgnav_decision_mode"] = last_decision_mode
        row["sgnav_decision_reason"] = last_decision_reason
        row["scenegraph_backend"] = "original" if scenegraph.scenegraph is not None else "fallback"
        row["detection_localization"] = detection_localization
        row["read_depth"] = bool(getattr(args, "read_depth", False))
        row["mapping_source"] = "depth_online"
        row["online_map_resolution_m"] = float(dynamic_map_info.resolution_m)
        row["online_observed_cells"] = int(np.count_nonzero(last_dynamic_observed))
        row["online_free_cells"] = int(np.count_nonzero(last_dynamic_navigable))
        row["online_occupied_cells"] = int(np.count_nonzero(last_dynamic_occupancy))
        row["segmenter"] = str(getattr(args, "segmenter", "none") or "none")
        row["camera_annotator_device"] = camera_annotator_device
        row["detector_cuda_rgb"] = bool(detector_cuda_rgb)
        row["sgnav_viz_every_steps"] = int(viz_every)
        row["max_vx_mps"] = float(args.max_vx_mps)
        row["max_vy_mps"] = float(args.max_vy_mps)
        row["max_wz_radps"] = float(args.max_wz_radps)
        if args.save_debug_video or args.debug_map:
            debug_map = args.debug_map or str(Path(args.output).with_suffix(".png"))
            start = world_xy_to_grid(float(start_pose[0]), float(start_pose[1]), dynamic_map_info)
            save_map_png(debug_map, last_dynamic_occupancy, last_dynamic_navigable, start=start, goals=goal_cells, path_cells=full_path)
        JsonlEpisodeLogger(args.output).log(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
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
    result = planner.plan(start, goals)
    evaluator = EpisodeEvaluator(episode, planner)
    env = MapSimHabitatLikeEnv(args.episode_file, args.episode_index)
    env.reset()

    object_memory = ObjectMemory()
    scenegraph = SGNavSceneGraphAdapter(args.sgnav_repo, use_original=args.use_original_scenegraph)
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
    if args.save_debug_video or args.debug_map:
        debug_map = args.debug_map or str(Path(args.output).with_suffix(".png"))
        save_map_png(debug_map, occupancy, navigable, start=start, goals=goals, path_cells=result.path)
    return row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="isaac_bench/configs/isaac_bench.yaml")
    parser.add_argument("--episode-file", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--planner", default=None, choices=["astar", "nav2"])
    parser.add_argument("--detector", default=None, choices=["dry_run", "yolo_world", "none"])
    parser.add_argument("--yolo-world-model", default=None)
    parser.add_argument("--detector-conf", type=float, default=None)
    parser.add_argument("--detector-iou", type=float, default=None)
    parser.add_argument("--headless", nargs="?", const=True, default=None, type=str_to_bool)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--sim-backend", default=None, choices=["map", "isaac"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--debug-map", default=None)
    parser.add_argument("--save-debug-video", action="store_true")
    parser.add_argument("--sgnav-repo", default=None)
    parser.add_argument("--use-original-scenegraph", action="store_true", default=None)
    parser.add_argument("--max-control-steps", type=int, default=None)
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
    parser.add_argument("--depth-max-m", type=float, default=None)
    parser.add_argument("--depth-min-m", type=float, default=None)
    parser.add_argument("--depth-stride-px", type=int, default=None)
    parser.add_argument("--online-map-size-m", type=float, default=None)
    parser.add_argument("--online-resolution-m", type=float, default=None)
    parser.add_argument("--obstacle-min-height-m", type=float, default=None)
    parser.add_argument("--obstacle-max-height-m", type=float, default=None)
    parser.add_argument("--robot-radius-m", type=float, default=None)
    parser.add_argument("--online-inflation-radius-m", type=float, default=None)
    parser.add_argument("--detection-localization", default=None, choices=["static_map_ray", "map_ray", "rgb_map_ray", "depth", "none"])
    parser.add_argument("--min-depth-points-per-detection", type=int, default=None)
    parser.add_argument("--segmenter", default=None, choices=["none", "auto", "sam2"])
    parser.add_argument("--sam2-checkpoint", default=None)
    parser.add_argument("--sam2-model-cfg", default=None)
    parser.add_argument("--sam2-device", default=None)
    parser.add_argument("--max-detections-per-frame", type=int, default=None)
    parser.add_argument("--object-merge-radius-m", type=float, default=None)
    parser.add_argument("--frontier-distance-weight", type=float, default=None)
    parser.add_argument("--runtime-planning-clearance-m", type=float, default=None)
    parser.add_argument("--candidate-min-detector-hits", type=int, default=None)
    parser.add_argument("--candidate-start-min-confidence", type=float, default=None)
    parser.add_argument("--candidate-stop-distance-m", type=float, default=None)
    parser.add_argument("--candidate-standoff-min-m", type=float, default=None)
    parser.add_argument("--candidate-standoff-max-m", type=float, default=None)
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
    parser.add_argument("--hold-open", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    args.planner = args.planner or get_nested(cfg, "repo.planner", "astar")
    args.detector = args.detector or get_nested(cfg, "repo.detector", "dry_run")
    args.yolo_world_model = args.yolo_world_model or get_nested(cfg, "paths.yolo_world_model", get_nested(cfg, "perception.yolo_world_model", "data/models/yolov8s-worldv2.pt"))
    args.detector_conf = float(args.detector_conf if args.detector_conf is not None else get_nested(cfg, "perception.confidence_threshold", 0.08))
    args.detector_iou = float(args.detector_iou if args.detector_iou is not None else get_nested(cfg, "perception.nms_iou_threshold", 0.5))
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
    args.sgnav_viz_width = int(args.sgnav_viz_width or get_nested(cfg, "visualization.sgnav_popup_width", 960))
    args.sgnav_viz_height = int(args.sgnav_viz_height or get_nested(cfg, "visualization.sgnav_popup_height", 540))
    args.sgnav_viz_jpeg_quality = int(args.sgnav_viz_jpeg_quality or get_nested(cfg, "visualization.sgnav_popup_jpeg_quality", 75))
    args.sim_backend = args.sim_backend or "map"
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
    args.camera_far_m = float(args.camera_far_m if args.camera_far_m is not None else get_nested(cfg, "camera.far_m", 80.0))
    args.camera_annotator_device = str(args.camera_annotator_device or get_nested(cfg, "isaac.camera_annotator_device", "cuda")).strip().lower()
    args.read_depth = bool(args.read_depth if args.read_depth is not None else get_nested(cfg, "isaac.read_depth", False))
    args.depth_max_m = float(args.depth_max_m if args.depth_max_m is not None else get_nested(cfg, "mapping.depth_max_m", 6.0))
    args.depth_min_m = float(args.depth_min_m if args.depth_min_m is not None else get_nested(cfg, "mapping.depth_min_m", 0.20))
    args.depth_stride_px = int(args.depth_stride_px if args.depth_stride_px is not None else get_nested(cfg, "mapping.depth_stride_px", 8))
    args.online_map_size_m = float(args.online_map_size_m if args.online_map_size_m is not None else get_nested(cfg, "mapping.map_size_m", 40.0))
    args.online_resolution_m = float(args.online_resolution_m if args.online_resolution_m is not None else get_nested(cfg, "mapping.online_resolution_m", get_nested(cfg, "scene_preprocess.map_resolution_m", 0.05)))
    args.obstacle_min_height_m = float(args.obstacle_min_height_m if args.obstacle_min_height_m is not None else get_nested(cfg, "mapping.obstacle_min_height_m", 0.05))
    args.obstacle_max_height_m = float(args.obstacle_max_height_m if args.obstacle_max_height_m is not None else get_nested(cfg, "mapping.obstacle_max_height_m", 1.50))
    args.robot_radius_m = float(args.robot_radius_m if args.robot_radius_m is not None else get_nested(cfg, "robot.footprint_radius_m", 0.28))
    args.online_inflation_radius_m = float(args.online_inflation_radius_m if args.online_inflation_radius_m is not None else get_nested(cfg, "mapping.inflation_radius_m", 0.0))
    args.detection_localization = str(args.detection_localization or get_nested(cfg, "perception.detection_localization", "static_map_ray"))
    args.min_depth_points_per_detection = int(args.min_depth_points_per_detection if args.min_depth_points_per_detection is not None else get_nested(cfg, "perception.min_depth_points_per_detection", 20))
    args.segmenter = str(args.segmenter or get_nested(cfg, "perception.segmenter", "none"))
    args.sam2_checkpoint = str(args.sam2_checkpoint if args.sam2_checkpoint is not None else get_nested(cfg, "perception.sam2_checkpoint", ""))
    args.sam2_model_cfg = str(args.sam2_model_cfg if args.sam2_model_cfg is not None else get_nested(cfg, "perception.sam2_model_cfg", "facebook/sam2.1-hiera-tiny"))
    args.sam2_device = str(args.sam2_device or get_nested(cfg, "perception.sam2_device", "cuda"))
    args.max_detections_per_frame = int(args.max_detections_per_frame if args.max_detections_per_frame is not None else get_nested(cfg, "perception.max_detections_per_frame", 100))
    args.object_merge_radius_m = float(args.object_merge_radius_m if args.object_merge_radius_m is not None else get_nested(cfg, "perception.object_merge_radius_m", 0.5))
    args.frontier_distance_weight = float(args.frontier_distance_weight if args.frontier_distance_weight is not None else get_nested(cfg, "sgnav.frontier_distance_weight", 2.0))
    args.runtime_planning_clearance_m = float(args.runtime_planning_clearance_m if args.runtime_planning_clearance_m is not None else get_nested(cfg, "astar.runtime_planning_clearance_m", 0.0))
    args.candidate_min_detector_hits = int(args.candidate_min_detector_hits if args.candidate_min_detector_hits is not None else get_nested(cfg, "sgnav.candidate_min_detector_hits", 2))
    args.candidate_start_min_confidence = float(args.candidate_start_min_confidence if args.candidate_start_min_confidence is not None else get_nested(cfg, "sgnav.candidate_start_min_confidence", 0.20))
    args.candidate_stop_distance_m = float(args.candidate_stop_distance_m if args.candidate_stop_distance_m is not None else get_nested(cfg, "sgnav.candidate_stop_distance_m", get_nested(cfg, "episodes.success_distance_m", 1.0)))
    args.candidate_standoff_min_m = float(args.candidate_standoff_min_m if args.candidate_standoff_min_m is not None else get_nested(cfg, "sgnav.candidate_standoff_min_m", 0.65))
    args.candidate_standoff_max_m = float(args.candidate_standoff_max_m if args.candidate_standoff_max_m is not None else get_nested(cfg, "sgnav.candidate_standoff_max_m", 1.80))
    args.seed_gt_object_memory = bool(get_nested(cfg, "sgnav.seed_gt_object_memory", False) if args.seed_gt_object_memory is None else args.seed_gt_object_memory)
    args.allow_gt_goal_fallback = bool(get_nested(cfg, "sgnav.allow_gt_goal_fallback", False) if args.allow_gt_goal_fallback is None else args.allow_gt_goal_fallback)

    if args.planner == "nav2":
        from isaac_bench.navigation.nav2_client import Nav2NavigateToPoseClient

        Nav2NavigateToPoseClient()
    episodes = read_jsonl(args.episode_file)
    episode = episodes[int(args.episode_index)]
    if args.sim_backend == "isaac":
        row = run_episode_isaac_closed_loop(episode, args)
    else:
        row = run_episode_map_sim(episode, args)
    if not getattr(args, "_row_already_logged", False):
        logger = JsonlEpisodeLogger(args.output)
        logger.log(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
