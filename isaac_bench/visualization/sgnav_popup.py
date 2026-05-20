from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from isaac_bench.config import repo_root
from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.graph.decision import NavigationDecision
from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.perception.detection_types import MIN_VALID_DETECTION_CONFIDENCE, Detection2D, detection_confidence_is_valid
from isaac_bench.perception.object_memory import ObjectMemory


GridCell = Tuple[int, int]


def _is_green_like(color: Tuple[int, int, int]) -> bool:
    r, g, b = [int(v) for v in color]
    return bool(g >= 150 and g > r + 25 and g >= b)


def _cell_in_crop(cell: GridCell, r0: int, r1: int, c0: int, c1: int) -> bool:
    row, col = int(cell[0]), int(cell[1])
    return bool(r0 <= row < r1 and c0 <= col < c1)


class SGNavPopupVisualizer:
    def __init__(
        self,
        enabled: bool = True,
        window_name: str = "SG-Nav Isaac Debug",
        save_dir: Optional[str] = None,
        panel_size: Tuple[int, int] = (1440, 900),
        save_every_steps: int = 10,
        ipc_jpeg_quality: int = 75,
        debug_overlay_layers: bool = True,
        save_overlay_layer_metadata: bool = True,
        show_gt_goal_cells: bool = False,
        show_room_proposals: bool = True,
        show_room_masks: bool = True,
        show_room_labels: bool = True,
        show_rose_occupancy_map: bool = True,
        show_frontier_member_cells: bool = True,
        show_object_nodes: bool = True,
        show_candidate_markers: bool = True,
        min_valid_detection_confidence: float = MIN_VALID_DETECTION_CONFIDENCE,
        max_green_like_primitives_before_warning: int = 200,
    ) -> None:
        self.enabled = bool(enabled)
        self.window_name = window_name
        self.save_dir = Path(save_dir) if save_dir else None
        self.panel_size = (int(panel_size[0]), int(panel_size[1]))
        self.save_every_steps = max(1, int(save_every_steps))
        self.ipc_jpeg_quality = max(30, min(95, int(ipc_jpeg_quality)))
        self.debug_overlay_layers = bool(debug_overlay_layers)
        self.save_overlay_layer_metadata = bool(save_overlay_layer_metadata)
        self.show_gt_goal_cells = bool(show_gt_goal_cells)
        self.show_room_proposals = bool(show_room_proposals)
        self.show_room_masks = bool(show_room_masks)
        self.show_room_labels = bool(show_room_labels)
        self.show_rose_occupancy_map = bool(show_rose_occupancy_map)
        self.show_frontier_member_cells = bool(show_frontier_member_cells)
        self.show_object_nodes = bool(show_object_nodes)
        self.show_candidate_markers = bool(show_candidate_markers)
        self.min_valid_detection_confidence = float(min_valid_detection_confidence)
        self.max_green_like_primitives_before_warning = max(0, int(max_green_like_primitives_before_warning))
        self._last_overlay_layers: List[dict] = []
        self._room_masks: List[object] = []
        self._room_semantic_labels: dict[str, object] = {}
        self._room_segmentation_debug: dict = {}
        self._proc: Optional[subprocess.Popen[str]] = None
        self._ipc_dir = Path(tempfile.gettempdir()) / ("sgnav_viz_%d" % os.getpid())
        self._frame_path = self._ipc_dir / "latest.jpg"
        self._font = ImageFont.load_default()
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def set_room_context(
        self,
        room_masks: Sequence[object],
        room_semantic_labels: Optional[Mapping[str, object]] = None,
        room_segmentation_debug: Optional[Mapping[str, object]] = None,
    ) -> None:
        self._room_masks = list(room_masks or [])
        self._room_semantic_labels = dict(room_semantic_labels or {})
        self._room_segmentation_debug = dict(room_segmentation_debug or {})

    def _try_open_window(self) -> None:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print("[sgnav-viz] no DISPLAY/WAYLAND_DISPLAY; popup disabled", file=sys.stderr, flush=True)
            self.enabled = False
            return
        try:
            root = repo_root()
            python_executable = self._default_python_executable()
            cmd = [
                python_executable,
                "-m",
                "isaac_bench.visualization.sgnav_popup_worker",
                "--window-name",
                self.window_name,
                "--width",
                str(self.panel_size[0]),
                "--height",
                str(self.panel_size[1]),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root)
            env["PYTHONUNBUFFERED"] = "1"
            if os.environ.get("ISAAC_BENCH_KEEP_POPUP_LD_LIBRARY_PATH", "").lower() not in {"1", "true", "yes"}:
                env.pop("LD_LIBRARY_PATH", None)
            print("[sgnav-viz] starting popup worker with %s" % python_executable, flush=True)
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
            )
            self._read_ready(timeout_s=10.0)
        except Exception as exc:
            print("[sgnav-viz] OpenCV popup unavailable: %s" % exc, file=sys.stderr, flush=True)
            self.enabled = False
            self.close()

    @staticmethod
    def _default_python_executable() -> str:
        explicit = os.environ.get("SG_NAV_PYTHON")
        if explicit:
            return explicit
        current = Path(sys.executable)
        if current.exists() and "sgnav-isaac" in str(current):
            return str(current)
        env_root = Path(os.environ.get("SG_NAV_ENV", "/home/echo/SG-Nav/.mamba/envs/sg-nav"))
        candidate = env_root / "bin" / "python"
        if candidate.exists():
            return str(candidate)
        return sys.executable

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin:
                    proc.stdin.write(json.dumps({"type": "close"}) + "\n")
                    proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream:
                    stream.close()
            except Exception:
                pass

    def update(
        self,
        *,
        step: int,
        rgb: np.ndarray,
        detections_2d: Sequence[Detection2D],
        occupancy: np.ndarray,
        navigable: np.ndarray,
        observed: np.ndarray,
        goal_cells: Sequence[GridCell],
        current_grid: GridCell,
        pose: Sequence[float],
        frontiers: Sequence[FrontierCluster],
        nav_decision: Optional[NavigationDecision],
        current_path: Sequence[GridCell],
        full_path: Sequence[GridCell],
        object_memory: ObjectMemory,
        goal_category: str,
        distance_to_goal: float,
        path_length: float,
        scenegraph_backend: str,
        score_debug: Optional[dict] = None,
        failure_reason: Optional[str] = None,
    ) -> np.ndarray:
        panel = self.render(
            step=step,
            rgb=rgb,
            detections_2d=detections_2d,
            occupancy=occupancy,
            navigable=navigable,
            observed=observed,
            goal_cells=goal_cells,
            current_grid=current_grid,
            pose=pose,
            frontiers=frontiers,
            nav_decision=nav_decision,
            current_path=current_path,
            full_path=full_path,
            object_memory=object_memory,
            goal_category=goal_category,
            distance_to_goal=distance_to_goal,
            path_length=path_length,
            scenegraph_backend=scenegraph_backend,
            score_debug=score_debug,
            failure_reason=failure_reason,
        )
        if self.save_dir and int(step) % self.save_every_steps == 0:
            Image.fromarray(panel).save(self.save_dir / ("sgnav_step_%06d.jpg" % int(step)), format="JPEG", quality=85)
            if self.save_overlay_layer_metadata:
                meta_path = self.save_dir / ("sgnav_step_%06d.layers.json" % int(step))
                meta_path.write_text(
                    json.dumps(self.overlay_layer_metadata(step), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        if self.enabled:
            self._send_frame(panel)
        return panel

    def overlay_layer_metadata(self, frame_id: int = 0) -> dict:
        layers = [dict(item) for item in self._last_overlay_layers]
        green_like = sum(int(item.get("primitive_count", 0)) for item in layers if bool(item.get("green_like", False)))
        payload = {
            "frame_id": int(frame_id),
            "layers": layers,
            "green_like_primitive_count": int(green_like),
            "max_green_like_primitives_before_warning": int(self.max_green_like_primitives_before_warning),
            "green_like_warning": bool(
                self.max_green_like_primitives_before_warning > 0
                and green_like > self.max_green_like_primitives_before_warning
            ),
        }
        if isinstance(self._room_segmentation_debug, Mapping):
            if self._room_segmentation_debug.get("roomseg_debug_layers"):
                payload["roomseg_debug_layers"] = dict(self._room_segmentation_debug.get("roomseg_debug_layers") or {})
                summary = dict(self._room_segmentation_debug.get("roomseg_debug_summary") or {})
                payload["roomseg_debug_summary"] = {
                    "counts": dict(summary.get("counts") or {}),
                    "likely_cause": summary.get("likely_cause"),
                }
            overlay = self._room_segmentation_debug.get("navigation_free_context_overlay")
            if isinstance(overlay, Mapping):
                payload["context_overlay"] = {
                    "enabled": bool(overlay.get("nav_free_overlay_enabled", False)),
                    "absorbed_cells": int(overlay.get("absorbed_cells", 0) or 0),
                    "remaining_unlabeled_nav_free_cells": int(overlay.get("remaining_unlabeled_nav_free_cells", 0) or 0),
                    "used_for_frontier_room_assignment": True,
                    "used_for_room_nodes": bool(overlay.get("nav_free_overlay_enabled", False)),
                }
        return payload

    def _read_ready(self, timeout_s: float) -> None:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("popup worker did not start")
        ready, _, _ = select.select([self._proc.stdout], [], [], float(timeout_s))
        if not ready:
            raise RuntimeError("timed out waiting for popup worker")
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("popup worker exited before ready (code=%s)" % self._proc.poll())
        response = json.loads(line)
        if response.get("type") != "ready":
            raise RuntimeError("popup worker protocol error: %s" % response)

    def _send_frame(self, panel: np.ndarray) -> None:
        if self._proc is None:
            self._try_open_window()
            if not self.enabled or self._proc is None:
                return
        proc = self._proc
        if proc is None or proc.stdin is None:
            self.enabled = False
            return
        if proc.poll() is not None:
            print("[sgnav-viz] popup worker exited; continuing without popup", file=sys.stderr, flush=True)
            self.enabled = False
            return
        try:
            self._ipc_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = self._ipc_dir / "latest.tmp.jpg"
            Image.fromarray(panel).save(tmp_path, format="JPEG", quality=self.ipc_jpeg_quality)
            os.replace(tmp_path, self._frame_path)
            proc.stdin.write(json.dumps({"type": "frame_path", "path": str(self._frame_path)}) + "\n")
            proc.stdin.flush()
        except Exception as exc:
            print("[sgnav-viz] popup IPC failed; continuing without popup: %s" % exc, file=sys.stderr, flush=True)
            self.enabled = False
            self.close()

    def render(
        self,
        *,
        step: int,
        rgb: np.ndarray,
        detections_2d: Sequence[Detection2D],
        occupancy: np.ndarray,
        navigable: np.ndarray,
        observed: np.ndarray,
        goal_cells: Sequence[GridCell],
        current_grid: GridCell,
        pose: Sequence[float],
        frontiers: Sequence[FrontierCluster],
        nav_decision: Optional[NavigationDecision],
        current_path: Sequence[GridCell],
        full_path: Sequence[GridCell],
        object_memory: ObjectMemory,
        goal_category: str,
        distance_to_goal: float,
        path_length: float,
        scenegraph_backend: str,
        score_debug: Optional[dict] = None,
        failure_reason: Optional[str] = None,
    ) -> np.ndarray:
        panel_w, panel_h = self.panel_size
        right_w = max(1, int(panel_w * 0.60))
        left_w = panel_w - right_w
        rgb_h = int(panel_h * 0.62)
        text_h = panel_h - rgb_h
        valid_detections_2d = self._valid_detections(detections_2d)

        panel = Image.new("RGB", (panel_w, panel_h), (18, 20, 24))
        rgb_panel = self._render_rgb(
            rgb,
            valid_detections_2d,
            (left_w, rgb_h),
            goal_category=goal_category,
            nav_decision=nav_decision,
        )
        map_panel = self._render_map(
            occupancy=occupancy,
            navigable=navigable,
            observed=observed,
            goal_cells=goal_cells,
            current_grid=current_grid,
            pose=pose,
            frontiers=frontiers,
            nav_decision=nav_decision,
            current_path=current_path,
            full_path=full_path,
            object_memory=object_memory,
            goal_category=goal_category,
            size=(right_w, panel_h),
        )
        text_panel = self._render_text(
            step=step,
            detections_2d=valid_detections_2d,
            frontiers=frontiers,
            nav_decision=nav_decision,
            object_memory=object_memory,
            goal_category=goal_category,
            distance_to_goal=distance_to_goal,
            path_length=path_length,
            scenegraph_backend=scenegraph_backend,
            score_debug=score_debug or {},
            failure_reason=failure_reason,
            size=(left_w, text_h),
        )
        panel.paste(rgb_panel, (0, 0))
        panel.paste(text_panel, (0, rgb_h))
        panel.paste(map_panel, (left_w, 0))
        draw = ImageDraw.Draw(panel)
        draw.line([(left_w, 0), (left_w, panel_h)], fill=(70, 74, 80), width=2)
        draw.line([(0, rgb_h), (left_w, rgb_h)], fill=(70, 74, 80), width=2)
        return np.asarray(panel, dtype=np.uint8)

    def _valid_detections(self, detections: Sequence[Detection2D]) -> List[Detection2D]:
        return [
            det
            for det in detections
            if detection_confidence_is_valid(float(det.confidence), self.min_valid_detection_confidence)
        ]

    def _render_rgb(
        self,
        rgb: np.ndarray,
        detections: Sequence[Detection2D],
        size: Tuple[int, int],
        goal_category: str = "",
        nav_decision: Optional[NavigationDecision] = None,
    ) -> Image.Image:
        width, height = size
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] < 3:
            arr = np.zeros((height, width, 3), dtype=np.uint8)
        src_h, src_w = arr.shape[:2]
        image = Image.new("RGB", (width, height), (12, 14, 18))
        scale = min(width / max(src_w, 1), height / max(src_h, 1))
        render_w = max(1, int(round(src_w * scale)))
        render_h = max(1, int(round(src_h * scale)))
        offset_x = (width - render_w) // 2
        offset_y = (height - render_h) // 2
        rgb_image = Image.fromarray(arr[:, :, :3]).resize((render_w, render_h), Image.BILINEAR)
        image.paste(rgb_image, (offset_x, offset_y))
        draw = ImageDraw.Draw(image)
        sx, sy = render_w / max(src_w, 1), render_h / max(src_h, 1)
        valid_detections = self._valid_detections(detections)
        for det in valid_detections[:30]:
            x1, y1, x2, y2 = det.bbox_xyxy
            box = [
                int(offset_x + x1 * sx),
                int(offset_y + y1 * sy),
                int(offset_x + x2 * sx),
                int(offset_y + y2 * sy),
            ]
            color = self._bbox_color(det, goal_category, nav_decision)
            draw.rectangle(box, outline=color, width=2)
            label = "%s %.2f" % (det.category, float(det.confidence))
            self._label(draw, (box[0], max(0, box[1] - 14)), label, color)
        self._label(draw, (8, 8), "RGB / detector detections: %d  green=normal red=goal" % len(valid_detections), (255, 255, 255))
        return image

    def _render_map(
        self,
        *,
        occupancy: np.ndarray,
        navigable: np.ndarray,
        observed: np.ndarray,
        goal_cells: Sequence[GridCell],
        current_grid: GridCell,
        pose: Sequence[float],
        frontiers: Sequence[FrontierCluster],
        nav_decision: Optional[NavigationDecision],
        current_path: Sequence[GridCell],
        full_path: Sequence[GridCell],
        object_memory: ObjectMemory,
        goal_category: str,
        size: Tuple[int, int],
    ) -> Image.Image:
        width, height = size
        rose_panel_enabled = bool(self.show_rose_occupancy_map)
        rose_h = 0
        divider_h = 0
        map_h_available = int(height)
        if rose_panel_enabled and height >= 180:
            rose_h = min(max(96, int(round(height * 0.28))), max(1, height // 2))
            divider_h = 2
            map_h_available = max(1, int(height) - rose_h - divider_h)
        h, w = occupancy.shape
        base = np.zeros((h, w, 3), dtype=np.uint8)
        nav = navigable.astype(bool)
        obs = observed.astype(bool)
        base[nav] = (218, 222, 224)
        base[~nav] = (72, 74, 76)
        base[occupancy.astype(bool)] = (24, 24, 24)
        base[~obs] = (base[~obs].astype(np.float32) * 0.45 + np.array([20, 24, 34], dtype=np.float32)).astype(np.uint8)
        active_room_masks = self._active_room_masks(occupancy.shape)
        proposal_room_masks = self._proposal_room_masks(occupancy.shape)
        proposal_room_cell_count = 0
        if self.show_room_proposals and proposal_room_masks:
            base, proposal_room_cell_count = self._apply_boolean_mask_overlay(
                base,
                proposal_room_masks,
                alpha=0.16,
                boundary_only=True,
            )
        room_mask_cell_count = 0
        room_boundary_cell_count = 0
        if self.show_room_masks and active_room_masks:
            base, room_mask_cell_count, room_boundary_cell_count = self._apply_room_mask_overlay(base, active_room_masks)

        r0, r1, c0, c1 = self._map_crop_bounds(
            occupancy=occupancy,
            navigable=navigable,
            observed=observed,
            current_grid=current_grid,
            frontiers=frontiers,
            nav_decision=nav_decision,
            current_path=current_path,
            full_path=full_path,
            object_memory=object_memory,
            goal_category=goal_category,
        )
        crop = base[r0:r1, c0:c1]
        crop_h, crop_w = crop.shape[:2]
        margin = 12
        scale = min((width - 2 * margin) / max(crop_w, 1), (map_h_available - 2 * margin) / max(crop_h, 1))
        map_w, map_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
        ox, oy = (width - map_w) // 2, (map_h_available - map_h) // 2
        image = Image.new("RGB", (width, map_h_available), (18, 20, 24))
        map_img = Image.fromarray(crop).resize((map_w, map_h), Image.NEAREST)
        image.paste(map_img, (ox, oy))
        draw = ImageDraw.Draw(image)
        layers: List[dict] = []

        def record(name: str, enabled: bool, color: Tuple[int, int, int], count: int, note: str = "", **extra) -> None:
            item = {
                "name": name,
                "enabled": bool(enabled),
                "color": [int(color[0]), int(color[1]), int(color[2])],
                "primitive_count": int(count),
                "green_like": bool(_is_green_like(color)),
                "note": note,
            }
            item.update(extra)
            layers.append(item)

        def xy(cell: GridCell) -> Tuple[int, int]:
            r, c = int(cell[0]), int(cell[1])
            return int(ox + (c - c0 + 0.5) * scale), int(oy + (r - r0 + 0.5) * scale)

        room_color = (145, 110, 255)
        record("proposal_room_masks", self.show_room_proposals, (150, 150, 160), proposal_room_cell_count, "watershed/proposal basins before doorway-constrained merge")
        record("room_masks", self.show_room_masks, room_color, room_mask_cell_count, "online geometry room mask fill")
        record("final_room_masks", self.show_room_masks, room_color, room_mask_cell_count, "post-merge room masks used by SG-Nav")
        record("room_boundaries", self.show_room_masks, room_color, room_boundary_cell_count, "online geometry room mask boundary")
        room_label_count = self._draw_room_labels(
            draw,
            active_room_masks,
            xy,
            crop_bounds=(r0, r1, c0, c1),
        ) if self.show_room_labels else 0
        record("room_labels", self.show_room_labels, (250, 250, 255), room_label_count, "VLM room category and reliability")
        merged_count, doorway_count, merge_reasons = self._draw_room_adjacency_debug_lines(draw, xy, (r0, r1, c0, c1))
        record(
            "room_merged_boundaries",
            merged_count > 0,
            (150, 150, 155),
            merged_count,
            "dashed proposal boundaries removed by doorway-constrained merge",
            adjacency_merge_reasons=merge_reasons,
        )
        record(
            "room_doorway_cuts",
            doorway_count > 0,
            (255, 170, 40),
            doorway_count,
            "bold verified doorway/gateway cuts preserved as room splits",
            adjacency_merge_reasons=[item for item in merge_reasons if item.get("verified_doorway")],
        )

        goal_color = (30, 220, 80)
        goal_count = self._draw_cells(draw, goal_cells, xy, goal_color, radius=2, max_cells=500) if self.show_gt_goal_cells else 0
        record("gt_goal_cells", self.show_gt_goal_cells, goal_color, goal_count, "disabled by default; oracle GT overlay")
        full_path_color = (80, 130, 255)
        record("full_path", True, full_path_color, self._draw_cells(draw, full_path, xy, full_path_color, radius=1, max_cells=1200))
        current_path_color = (245, 245, 245)
        record("current_path", True, current_path_color, self._draw_cells(draw, current_path, xy, current_path_color, radius=2, max_cells=500))
        frontier_raw_cells: List[GridCell] = []
        for frontier in frontiers[:64]:
            frontier_raw_cells.extend(frontier.members)
        frontier_cell_color = (0, 180, 220)
        frontier_cell_count = (
            self._draw_cells(draw, frontier_raw_cells, xy, frontier_cell_color, radius=1, max_cells=500)
            if self.show_frontier_member_cells
            else 0
        )
        record("frontier_member_cells", self.show_frontier_member_cells, frontier_cell_color, frontier_cell_count)
        frontier_center_count = 0
        for frontier in frontiers[:64]:
            self._triangle(draw, xy(frontier.center_grid), (0, 225, 255), radius=5)
            frontier_center_count += 1
        record("frontier_centers", True, (0, 225, 255), frontier_center_count)

        selected_frontier = None
        if nav_decision and nav_decision.frontier_decision:
            selected_frontier = nav_decision.frontier_decision.selected_frontier
        if selected_frontier is not None:
            self._star(draw, xy(selected_frontier.center_grid), (255, 225, 40), radius=8)
        record("selected_frontier", selected_frontier is not None, (255, 225, 40), 1 if selected_frontier is not None else 0)
        candidate_marker_color = (220, 70, 255)
        candidate_marker_count = 0
        if self.show_candidate_markers and nav_decision and nav_decision.target_cells and nav_decision.mode == "candidate":
            candidate_marker_count = self._draw_crosses(draw, nav_decision.target_cells, xy, candidate_marker_color, radius=5, max_cells=16)
        record("candidate_standoff_markers", self.show_candidate_markers, candidate_marker_color, candidate_marker_count)
        planner_target = None
        if current_path:
            planner_target = current_path[-1]
        elif nav_decision and nav_decision.target_cells:
            planner_target = nav_decision.target_cells[0]
        if planner_target is not None:
            self._star(draw, xy(planner_target), (255, 150, 40), radius=7)
        record("planner_target", planner_target is not None, (255, 150, 40), 1 if planner_target is not None else 0)
        selected_id = self._selected_candidate_id(nav_decision)
        object_node_count = 0
        accepted_candidate_count = 0
        if self.show_object_nodes:
            for node in self._visible_map_nodes(object_memory, goal_category, nav_decision)[:300]:
                radius = 8 if selected_id is not None and int(node.node_id) == selected_id else 5
                color = self._candidate_node_color(node, selected_id, nav_decision)
                if color == (40, 220, 90):
                    accepted_candidate_count += 1
                self._dot(draw, xy(node.center_grid), color, radius=radius)
                object_node_count += 1
        record("object_nodes", self.show_object_nodes, (255, 150, 40), object_node_count)
        record("accepted_candidate", self.show_object_nodes, (40, 220, 90), accepted_candidate_count)

        self._draw_agent(draw, xy(current_grid), float(pose[3]) if len(pose) > 3 else 0.0, scale)
        record("agent", True, (255, 60, 60), 1)
        zoom = max(1.0, min(w / max(crop_w, 1), h / max(crop_h, 1)))
        target_count = len(nav_decision.target_cells) if nav_decision is not None else 0
        self._label(draw, (10, 8), "Map / frontiers / A* / goal candidates  zoom %.1fx target_cells=%d" % (zoom, target_count), (255, 255, 255))
        self._legend(draw, (10, max(32, map_h_available - 120)))
        if rose_panel_enabled and rose_h > 0:
            rose_panel, rose_layers = self._render_rose_occupancy_panel(
                occupancy=occupancy,
                navigable=navigable,
                observed=observed,
                size=(width, rose_h),
                crop_bounds=(r0, r1, c0, c1),
            )
            final = Image.new("RGB", (width, height), (18, 20, 24))
            final.paste(image, (0, 0))
            final_draw = ImageDraw.Draw(final)
            final_draw.line([(0, map_h_available), (width, map_h_available)], fill=(70, 74, 80), width=divider_h)
            final.paste(rose_panel, (0, map_h_available + divider_h))
            layers.extend(rose_layers)
            self._last_overlay_layers = layers if self.debug_overlay_layers else []
            return final
        self._last_overlay_layers = layers if self.debug_overlay_layers else []
        return image

    def _render_rose_occupancy_panel(
        self,
        *,
        occupancy: np.ndarray,
        navigable: np.ndarray,
        observed: np.ndarray,
        size: Tuple[int, int],
        crop_bounds: Tuple[int, int, int, int],
    ) -> Tuple[Image.Image, List[dict]]:
        width, height = size
        shape = tuple(np.asarray(occupancy).shape[:2])
        occ = np.asarray(occupancy, dtype=bool)
        nav = np.asarray(navigable, dtype=bool)
        obs = np.asarray(observed, dtype=bool)
        roomseg_free = self._room_debug_array("initial_roomseg_free", shape, bool)
        roomseg_occupied = self._room_debug_array("initial_roomseg_occupied", shape, bool)
        initial_unknown_after_fusion = self._room_debug_array("initial_roomseg_unknown_after_fusion", shape, bool)
        vertical_free_room_domain = self._room_debug_array("vertical_free_room_domain", shape, bool)
        vertical_occupied_0p2_2p0 = self._room_debug_array("vertical_occupied_0p2_2p0", shape, bool)
        vertical_observed = self._room_debug_array("vertical_observed_map", shape, bool)
        vertical_observed_0p2_2p0 = self._room_debug_array("vertical_observed_0p2_2p0", shape, bool)
        vertical_unknown_before_overlay = self._room_debug_array("vertical_unknown_before_overlay", shape, bool)
        roomseg_ray_covered_count = self._room_debug_array("roomseg_ray_covered_count", shape, np.uint16)
        roomseg_terminal_wall_count = self._room_debug_array("roomseg_terminal_wall_count", shape, np.uint16)
        roomseg_terminal_wall_splat = self._room_debug_array("roomseg_terminal_wall_splat", shape, bool)
        ray_valid_wall = self._room_debug_array("ray_valid_wall_inference", shape, bool)
        unknown_removed_by_ray_wall = self._room_debug_array("unknown_removed_by_ray_wall", shape, bool)
        nav_raw_obstacle = self._room_debug_array("nav_raw_obstacle", shape, bool)
        static_structural = self._room_debug_array("roomseg_static_structural_occupied", shape, bool)
        nav_obstacle_overlay_accepted = self._room_debug_array("nav_obstacle_overlay_accepted", shape, bool)
        walls_rescued_from_unknown = self._room_debug_array("walls_rescued_from_unknown", shape, bool)
        vertical_free_over_nav_obstacle = self._room_debug_array("vertical_free_over_nav_obstacle", shape, bool)
        repaired_free = self._room_debug_array("repaired_roomseg_free", shape, bool)
        repaired_occupied = self._room_debug_array("repaired_roomseg_occupied", shape, bool)
        boundary_map = self._room_debug_array("boundary_map", shape, bool)
        virtual_boundary = self._room_debug_array("virtual_boundary_map", shape, bool)
        accepted_closure = self._room_debug_array("accepted_closure_map", shape, bool)
        vertical_or_free = self._room_debug_array("vertical_or_free_map", shape, bool)
        vertical_carved = self._room_debug_array("vertical_carved_map", shape, bool)
        structural = self._room_debug_array("structural_wall_mask", shape, bool)
        clean_structure = self._room_debug_array("clean_structure_map", shape, bool)
        rejected_structure = self._room_debug_array("structural_component_rejected_mask", shape, bool)
        interior_clutter = self._room_debug_array("interior_clutter_suppression_mask", shape, bool)
        furniture_suppressed = self._room_debug_array("furniture_suppression_mask", shape, bool)
        suppressed_clutter = rejected_structure | interior_clutter | furniture_suppressed
        wall_conf = self._room_debug_array("wall_confidence_map", shape, np.float32)
        context_labels = self._room_debug_array("context_room_label_map", shape, np.int32)
        final_labels = self._room_debug_array("final_room_label_map", shape, np.int32)
        context_absorbed = (context_labels > 0) & (final_labels <= 0)
        threshold = float(self._room_segmentation_debug.get("wall_confidence_threshold", 0.55) or 0.55)
        wall_conf_hot = wall_conf >= threshold if wall_conf.shape == shape else np.zeros(shape, dtype=bool)
        debug_only = bool(self._room_segmentation_debug.get("roomseg_debug_only", False))
        vertical_debug_free = (
            vertical_free_room_domain
            if np.any(vertical_free_room_domain)
            else (repaired_free if np.any(repaired_free) else roomseg_free)
        )

        has_rose_input = bool(
            np.any(vertical_debug_free)
            or np.any(roomseg_free)
            or np.any(roomseg_occupied)
            or np.any(vertical_observed)
            or np.any(vertical_observed_0p2_2p0)
            or np.any(ray_valid_wall)
            or np.any(roomseg_terminal_wall_splat)
            or np.any(nav_raw_obstacle)
            or np.any(static_structural)
            or np.any(nav_obstacle_overlay_accepted)
            or np.any(vertical_or_free)
            or np.any(vertical_carved)
            or np.any(structural)
            or np.any(clean_structure)
            or np.any(suppressed_clutter)
            or np.any(wall_conf > 0.0)
        )
        rose_occupied = (
            repaired_occupied
            if debug_only and np.any(repaired_occupied)
            else (roomseg_occupied if np.any(roomseg_occupied) else (structural if np.any(structural) else (clean_structure | wall_conf_hot)))
        )
        vertical_free_overridden_occupied = occ & vertical_or_free & ~roomseg_occupied
        canvas = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        if debug_only:
            canvas[:, :] = (12, 14, 18)
            canvas[obs] = (28, 31, 36)
            canvas[initial_unknown_after_fusion] = (8, 10, 14)
            canvas[vertical_debug_free] = (166, 170, 174)
            canvas[rose_occupied | boundary_map] = (0, 0, 0)
            label_palette = [
                (125, 104, 235),
                (100, 185, 245),
                (95, 210, 155),
                (235, 185, 80),
                (235, 120, 145),
                (180, 140, 240),
            ]
            for label in np.unique(final_labels):
                if int(label) <= 0:
                    continue
                mask = final_labels == int(label)
                color = np.asarray(label_palette[(int(label) - 1) % len(label_palette)], dtype=np.float32)
                base = canvas[mask].astype(np.float32)
                canvas[mask] = np.clip(base * 0.45 + color * 0.55, 0, 255).astype(np.uint8)
            canvas[accepted_closure] = (255, 205, 45)
            canvas[virtual_boundary] = (255, 65, 90)
            canvas[ray_valid_wall] = (255, 80, 40)
            canvas[roomseg_terminal_wall_splat] = (255, 135, 25)
            canvas[nav_obstacle_overlay_accepted] = (230, 40, 230)
            canvas[walls_rescued_from_unknown] = (255, 35, 35)
            canvas[vertical_free_over_nav_obstacle] = (45, 135, 255)
        else:
            canvas[:, :] = (36, 40, 48)
            canvas[nav] = (86, 92, 96)
            canvas[obs & nav] = (118, 126, 130)
            canvas[obs & ~nav] = (54, 56, 60)
            if not has_rose_input:
                canvas[occ] = (8, 8, 8)
            canvas[roomseg_free] = (150, 156, 160)
            canvas[suppressed_clutter] = (58, 82, 132)
            canvas[vertical_or_free] = (116, 96, 74)
            canvas[vertical_carved] = (126, 104, 75)
            canvas[vertical_free_overridden_occupied] = (225, 132, 45)
            canvas[ray_valid_wall] = (255, 80, 40)
            canvas[roomseg_terminal_wall_splat] = (255, 135, 25)
            canvas[unknown_removed_by_ray_wall] = (255, 35, 35)
            canvas[wall_conf_hot] = (255, 105, 75)
            canvas[rose_occupied] = (0, 0, 0)
            canvas[clean_structure] = (255, 190, 70)
            canvas[context_absorbed] = (120, 210, 255)
            canvas[nav_obstacle_overlay_accepted] = (230, 40, 230)
            canvas[walls_rescued_from_unknown] = (255, 35, 35)
            canvas[vertical_free_over_nav_obstacle] = (45, 135, 255)

        r0, r1, c0, c1 = crop_bounds
        crop = canvas[r0:r1, c0:c1]
        crop_h, crop_w = crop.shape[:2]
        label_h = 22
        margin = 8
        available_h = max(1, int(height) - label_h - margin)
        scale = min((width - 2 * margin) / max(crop_w, 1), available_h / max(crop_h, 1))
        map_w, map_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
        ox = (width - map_w) // 2
        oy = label_h + max(0, (available_h - map_h) // 2)
        image = Image.new("RGB", (width, height), (15, 17, 21))
        image.paste(Image.fromarray(crop).resize((map_w, map_h), Image.NEAREST), (ox, oy))
        draw = ImageDraw.Draw(image)

        def xy(cell: GridCell) -> Tuple[int, int]:
            r, c = int(cell[0]), int(cell[1])
            return int(ox + (c - c0 + 0.5) * scale), int(oy + (r - r0 + 0.5) * scale)

        window_gap_count = self._draw_rose_gap_markers(
            draw,
            list(self._room_segmentation_debug.get("repaired_window_gaps") or []),
            xy,
            crop_bounds,
            (255, 80, 130),
        )
        doorway_gap_count = self._draw_rose_gap_markers(
            draw,
            list(self._room_segmentation_debug.get("verified_doorway_gaps") or []),
            xy,
            crop_bounds,
            (80, 255, 130),
        )
        title = "vertical-free roomseg debug" if debug_only else "ROSE roomseg input after vertical-free operation"
        if has_rose_input:
            title += " | occupied=%d vfree=%d ray_wall=%d room_pixels=%d closures=%d win=%d door=%d" % (
                int(np.count_nonzero(rose_occupied)),
                int(np.count_nonzero(vertical_debug_free)),
                int(np.count_nonzero(ray_valid_wall)),
                int(np.count_nonzero(final_labels > 0)),
                int(np.count_nonzero(accepted_closure | virtual_boundary)),
                int(window_gap_count),
                int(doorway_gap_count),
            )
        else:
            title += " | waiting for ROSE debug; showing current map underlay"
        self._label(draw, (8, 5), title[:120], (255, 255, 255))
        self._rose_legend(draw, (8, max(label_h + 4, height - 50)))

        layers = [
            self._overlay_record(
                "rose_occupancy_map",
                True,
                (0, 0, 0),
                int(np.count_nonzero(rose_occupied)),
                "room segmentation occupancy after vertical-free operation, shown below the runtime occupancy map",
                has_rose_input=has_rose_input,
                current_occupancy_underlay_cells=int(np.count_nonzero(occ)),
                current_navigable_underlay_cells=int(np.count_nonzero(nav)),
                roomseg_free_cells=int(np.count_nonzero(roomseg_free)),
                vertical_free_overridden_occupied_cells=int(np.count_nonzero(vertical_free_overridden_occupied)),
            ),
            self._overlay_record(
                "roomseg_ray_valid_wall_inference",
                True,
                (255, 80, 40),
                int(np.count_nonzero(ray_valid_wall)),
                "roomseg occupied cells inferred only from vertical occupied evidence or valid depth-ray terminal wall evidence",
                terminal_wall_cells=int(np.count_nonzero(roomseg_terminal_wall_count)),
                terminal_wall_splat_cells=int(np.count_nonzero(roomseg_terminal_wall_splat)),
                ray_covered_cells=int(np.count_nonzero(roomseg_ray_covered_count)),
                unknown_removed_by_ray_wall_cells=int(np.count_nonzero(unknown_removed_by_ray_wall)),
            ),
            self._overlay_record(
                "roomseg_nav_obstacle_overlay_accepted",
                True,
                (230, 40, 230),
                int(np.count_nonzero(nav_obstacle_overlay_accepted)),
                "debug-only audit; strict ray-valid roomseg must keep this at zero",
                nav_raw_obstacle_cells=int(np.count_nonzero(nav_raw_obstacle)),
                roomseg_static_structural_occupied_cells=int(np.count_nonzero(static_structural)),
            ),
            self._overlay_record(
                "roomseg_walls_rescued_from_unknown",
                True,
                (255, 35, 35),
                int(np.count_nonzero(walls_rescued_from_unknown)),
                "wall cells rescued from roomseg unknown before ROSE2 receives the structural map",
            ),
            self._overlay_record(
                "roomseg_vertical_free_over_nav_obstacle",
                True,
                (45, 135, 255),
                int(np.count_nonzero(vertical_free_over_nav_obstacle)),
                "vertical-free cells that override raw/static navigation obstacle evidence",
            ),
            self._overlay_record(
                "vertical_free_roomseg_input",
                True,
                (150, 156, 160),
                int(np.count_nonzero(vertical_debug_free)),
                "vertical-free room segmentation input used by the current latest room segmenter",
                vertical_observed_cells=int(np.count_nonzero(vertical_observed)),
                vertical_observed_0p2_2p0_cells=int(np.count_nonzero(vertical_observed_0p2_2p0)),
                vertical_occupied_0p2_2p0_cells=int(np.count_nonzero(vertical_occupied_0p2_2p0)),
                vertical_unknown_before_overlay_cells=int(np.count_nonzero(vertical_unknown_before_overlay)),
                initial_roomseg_unknown_after_fusion_cells=int(np.count_nonzero(initial_unknown_after_fusion)),
            ),
            self._overlay_record(
                "vertical_free_room_labels",
                bool(np.any(final_labels > 0)),
                (125, 104, 235),
                int(np.count_nonzero(final_labels > 0)),
                "final room labels overlaid on the vertical-free roomseg input",
            ),
            self._overlay_record(
                "vertical_free_gap_closure_boundaries",
                bool(np.any(accepted_closure | virtual_boundary)),
                (255, 65, 90),
                int(np.count_nonzero(accepted_closure | virtual_boundary)),
                "accepted or virtual gap-closure boundaries from the latest room segmentation method",
            ),
            self._overlay_record(
                "rose_vertical_free_overrides",
                True,
                (225, 132, 45),
                int(np.count_nonzero(vertical_free_overridden_occupied)),
                "runtime occupied cells changed to free in the ROSE-only room segmentation input",
            ),
            self._overlay_record(
                "rose_vertical_carved_map",
                True,
                (126, 104, 75),
                int(np.count_nonzero(vertical_carved)),
                "furniture-suppressed vertical carved map passed into ROSE preprocessing",
            ),
            self._overlay_record(
                "rose_wall_confidence_map",
                True,
                (255, 105, 75),
                int(np.count_nonzero(wall_conf_hot)),
                "wall-confidence cells above room segmentation threshold",
                threshold=threshold,
            ),
            self._overlay_record(
                "rose_structural_rejected_clutter",
                True,
                (58, 82, 132),
                int(np.count_nonzero(suppressed_clutter)),
                "occupied clutter/furniture components rejected before ROSE structural occupancy",
            ),
            self._overlay_record(
                "rose_repaired_window_gaps",
                True,
                (255, 80, 130),
                int(window_gap_count),
                "window/non-traversable gaps closed as walls for room segmentation",
            ),
            self._overlay_record(
                "rose_verified_doorway_gaps",
                True,
                (80, 255, 130),
                int(doorway_gap_count),
                "floor-traversable doorway gaps recorded as portals",
            ),
            self._overlay_record(
                "roomseg_context_absorbed_nav_free",
                bool(np.any(context_absorbed)),
                (120, 210, 255),
                int(np.count_nonzero(context_absorbed)),
                "navigation-free cells absorbed only into room context overlay",
            ),
        ]
        return image, layers

    def _room_debug_array(self, key: str, shape: Tuple[int, int], dtype) -> np.ndarray:
        raw = self._room_segmentation_debug.get(key)
        if raw is None:
            return np.zeros(shape, dtype=dtype)
        try:
            arr = np.asarray(raw, dtype=dtype)
        except Exception:
            return np.zeros(shape, dtype=dtype)
        if arr.shape != tuple(shape):
            return np.zeros(shape, dtype=dtype)
        return arr

    def _overlay_record(
        self,
        name: str,
        enabled: bool,
        color: Tuple[int, int, int],
        count: int,
        note: str = "",
        **extra,
    ) -> dict:
        item = {
            "name": name,
            "enabled": bool(enabled),
            "color": [int(color[0]), int(color[1]), int(color[2])],
            "primitive_count": int(count),
            "green_like": bool(_is_green_like(color)),
            "note": note,
        }
        item.update(extra)
        return item

    def _draw_rose_gap_markers(
        self,
        draw: ImageDraw.ImageDraw,
        gaps: Sequence[object],
        xy_func,
        crop_bounds: Tuple[int, int, int, int],
        color: Tuple[int, int, int],
    ) -> int:
        r0, r1, c0, c1 = crop_bounds
        count = 0
        for gap in gaps[:128]:
            if not isinstance(gap, Mapping):
                continue
            axis = str(gap.get("axis", "vertical"))
            index = int(gap.get("index", 0) or 0)
            start = int(gap.get("start", 0) or 0)
            end = int(gap.get("end", start) or start)
            p0 = (start, index) if axis == "vertical" else (index, start)
            p1 = (end, index) if axis == "vertical" else (index, end)
            if not (_cell_in_crop(p0, r0, r1, c0, c1) or _cell_in_crop(p1, r0, r1, c0, c1)):
                continue
            draw.line([xy_func(p0), xy_func(p1)], fill=color, width=3)
            count += 1
        return count

    def _rose_legend(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int]) -> None:
        x, y = xy
        items = [
            ((0, 0, 0), "ROSE roomseg occupied"),
            ((150, 156, 160), "ROSE roomseg free"),
            ((225, 132, 45), "occupied -> free"),
            ((230, 40, 230), "nav obstacle overlay"),
            ((255, 35, 35), "rescued wall"),
            ((45, 135, 255), "vfree wins warning"),
            ((58, 82, 132), "rejected clutter"),
            ((126, 104, 75), "vertical-carved"),
            ((255, 190, 70), "ROSE line"),
            ((255, 80, 130), "closed window"),
            ((80, 255, 130), "doorway portal"),
        ]
        for color, label in items:
            self._dot(draw, (x + 5, y + 7), color, radius=4)
            draw.text((x + 14, y), label, fill=(230, 232, 235), font=self._font)
            y += 10

    def _map_crop_bounds(
        self,
        *,
        occupancy: np.ndarray,
        navigable: np.ndarray,
        observed: np.ndarray,
        current_grid: GridCell,
        frontiers: Sequence[FrontierCluster],
        nav_decision: Optional[NavigationDecision],
        current_path: Sequence[GridCell],
        full_path: Sequence[GridCell],
        object_memory: ObjectMemory,
        goal_category: str,
    ) -> Tuple[int, int, int, int]:
        h, w = occupancy.shape
        rows: List[int] = []
        cols: List[int] = []

        active = observed.astype(bool) | occupancy.astype(bool) | navigable.astype(bool)
        rr, cc = np.nonzero(active)
        if rr.size:
            rows.extend(int(v) for v in rr)
            cols.extend(int(v) for v in cc)

        def add_cell(cell: GridCell) -> None:
            r, c = int(cell[0]), int(cell[1])
            if 0 <= r < h and 0 <= c < w:
                rows.append(r)
                cols.append(c)

        add_cell(current_grid)
        for cell in current_path:
            add_cell(cell)
        for cell in full_path:
            add_cell(cell)
        for frontier in frontiers[:128]:
            add_cell(frontier.center_grid)
        if nav_decision is not None:
            target_cells = nav_decision.target_cells
            if nav_decision.mode != "candidate":
                target_cells = target_cells[:1]
            for cell in target_cells[:32]:
                add_cell(cell)
            if nav_decision.frontier_decision is not None and nav_decision.frontier_decision.selected_frontier is not None:
                add_cell(nav_decision.frontier_decision.selected_frontier.center_grid)
            if nav_decision.selected_candidate is not None:
                add_cell(nav_decision.selected_candidate.center_grid)
        for node in self._visible_map_nodes(object_memory, goal_category, nav_decision)[:500]:
            add_cell(node.center_grid)
        for room in self._active_room_masks((h, w))[:64]:
            mask = np.asarray(getattr(room, "mask", None), dtype=bool)
            rr, cc = np.nonzero(mask)
            if rr.size:
                rows.extend(int(v) for v in rr[:: max(1, rr.size // 128)])
                cols.extend(int(v) for v in cc[:: max(1, cc.size // 128)])

        if not rows or not cols:
            return 0, h, 0, w

        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        padding = max(24, int(round(min(h, w) * 0.04)))
        min_r = max(0, min_r - padding)
        max_r = min(h - 1, max_r + padding)
        min_c = max(0, min_c - padding)
        max_c = min(w - 1, max_c + padding)

        crop_h = max_r - min_r + 1
        crop_w = max_c - min_c + 1
        min_crop = min(max(h, w), 160)
        if crop_h < min_crop:
            extra = min_crop - crop_h
            min_r = max(0, min_r - extra // 2)
            max_r = min(h - 1, max_r + extra - extra // 2)
        if crop_w < min_crop:
            extra = min_crop - crop_w
            min_c = max(0, min_c - extra // 2)
            max_c = min(w - 1, max_c + extra - extra // 2)

        return min_r, max_r + 1, min_c, max_c + 1

    def _render_text(
        self,
        *,
        step: int,
        detections_2d: Sequence[Detection2D],
        frontiers: Sequence[FrontierCluster],
        nav_decision: Optional[NavigationDecision],
        object_memory: ObjectMemory,
        goal_category: str,
        distance_to_goal: float,
        path_length: float,
        scenegraph_backend: str,
        score_debug: dict,
        failure_reason: Optional[str],
        size: Tuple[int, int],
    ) -> Image.Image:
        width, height = size
        image = Image.new("RGB", (width, height), (24, 26, 30))
        draw = ImageDraw.Draw(image)
        goal_norm = normalize_category(goal_category)
        goal_nodes = [
            node
            for node in object_memory.nodes
            if normalize_category(node.category) == goal_norm
            or (goal_norm and (goal_norm in normalize_category(node.category) or normalize_category(node.category) in goal_norm))
        ]
        mode = nav_decision.mode if nav_decision else "init"
        reason = nav_decision.reason if nav_decision else ""
        lines = [
            "SG-Nav decision panel",
            "step=%d goal=%s mode=%s reason=%s" % (int(step), goal_category, mode, reason),
            "dtg=%.2fm path=%.2fm objects=%d goal_candidates=%d detections=%d frontiers=%d backend=%s"
            % (
                float(distance_to_goal),
                float(path_length),
                len(object_memory.nodes),
                len(goal_nodes),
                len(detections_2d),
                len(frontiers),
                scenegraph_backend,
            ),
        ]
        if failure_reason:
            lines.append("failure=%s" % failure_reason)
        if nav_decision and nav_decision.selected_candidate is not None:
            node = nav_decision.selected_candidate
            lines.append(
                "candidate id=%d cat=%s conf=%.2f hits=%d grid=%s"
                % (node.node_id, node.category, node.confidence, node.observed_count, tuple(node.center_grid))
            )
        if nav_decision:
            meta = nav_decision.metadata or {}
            lines.append(
                "selected_frontier=%s selected_candidate=%s target_cells=%d"
                % (
                    nav_decision.frontier_decision.selected_index
                    if nav_decision.frontier_decision is not None
                    else None,
                    self._selected_candidate_id(nav_decision),
                    len(nav_decision.target_cells),
                )
            )
            if meta:
                commit = meta.get("frontier_commitment", {}) if isinstance(meta, dict) else {}
                if isinstance(commit, dict) and commit:
                    lines.append(
                        "frontier_commit id=%s age=%s dist=%s reason=%s"
                        % (
                            self._short(commit.get("active_frontier_id", "n/a")),
                            self._short(commit.get("active_frontier_age", "n/a")),
                            self._short(commit.get("active_frontier_distance_m", "n/a")),
                            self._short(commit.get("frontier_commitment_reason", "n/a")),
                        )
                    )
                lines.append(
                    "candidate credibility=%s track_obs=%s rep_steps=%s accepted=%s rejected=%s"
                    % (
                        self._short(meta.get("candidate_credibility", "n/a")),
                        self._short(meta.get("candidate_track_observation_count", "n/a")),
                        self._short(meta.get("candidate_reperception_steps", "n/a")),
                        self._short(meta.get("candidate_accepted", "n/a")),
                        self._short(meta.get("candidate_rejected", "n/a")),
                    )
                )
        if nav_decision and nav_decision.frontier_decision is not None:
            fd = nav_decision.frontier_decision
            lines.append("frontier selected=%s" % (fd.selected_index,))
            top = sorted(
                enumerate(fd.total_scores),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
            for idx, total in top:
                sg = fd.scenegraph_scores[idx] if idx < len(fd.scenegraph_scores) else 0.0
                dist = fd.distance_scores[idx] if idx < len(fd.distance_scores) else 0.0
                lines.append("  #%d sg=%.3f dist=%.3f total=%.3f" % (idx, sg, dist, total))
        if score_debug:
            compact = ", ".join("%s=%s" % (k, self._short(v)) for k, v in list(score_debug.items())[:5])
            lines.append("score_debug: %s" % compact)
        if detections_2d:
            det_line = ", ".join("%s %.2f" % (d.category, d.confidence) for d in detections_2d[:5])
            lines.append("detections: %s" % det_line)

        y = 8
        for i, line in enumerate(lines[:11]):
            fill = (255, 255, 255) if i == 0 else (215, 220, 225)
            draw.text((10, y), line[:110], fill=fill, font=self._font)
            y += 18
        return image

    def _active_room_masks(self, shape: Tuple[int, int]) -> List[object]:
        out = []
        for room in self._room_masks:
            if bool(getattr(room, "stale", False)):
                continue
            mask = getattr(room, "mask", None)
            if mask is None:
                continue
            arr = np.asarray(mask, dtype=bool)
            if arr.shape != tuple(shape) or not np.any(arr):
                continue
            out.append(room)
        return out

    def _proposal_room_masks(self, shape: Tuple[int, int]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for item in list(self._room_segmentation_debug.get("proposal_room_masks") or []):
            if not isinstance(item, Mapping):
                continue
            raw_mask = item.get("mask")
            if raw_mask is None:
                continue
            arr = np.asarray(raw_mask, dtype=bool)
            if arr.shape != tuple(shape) or not np.any(arr):
                continue
            out.append(arr)
        return out

    def _apply_boolean_mask_overlay(
        self,
        base: np.ndarray,
        masks: Sequence[np.ndarray],
        alpha: float,
        boundary_only: bool = False,
    ) -> Tuple[np.ndarray, int]:
        out = np.asarray(base, dtype=np.uint8).copy()
        total_cells = 0
        for idx, mask in enumerate(masks[:96]):
            arr = np.asarray(mask, dtype=bool)
            if arr.shape != out.shape[:2] or not np.any(arr):
                continue
            draw_mask = self._mask_boundary(arr) if boundary_only else arr
            total_cells += int(np.count_nonzero(draw_mask))
            color = np.asarray(self._room_color(idx), dtype=np.float32)
            blended = out[draw_mask].astype(np.float32) * (1.0 - float(alpha)) + color[None, :] * float(alpha)
            out[draw_mask] = np.clip(blended, 0, 255).astype(np.uint8)
        return out, total_cells

    def _apply_room_mask_overlay(self, base: np.ndarray, room_masks: Sequence[object]) -> Tuple[np.ndarray, int, int]:
        out = np.asarray(base, dtype=np.uint8).copy()
        owner = np.zeros(out.shape[:2], dtype=np.int32)
        total_mask_cells = 0
        total_boundary_cells = 0
        for idx, room in enumerate(room_masks[:64]):
            mask = np.asarray(getattr(room, "mask", None), dtype=bool)
            if mask.shape != out.shape[:2] or not np.any(mask):
                continue
            owner[(owner <= 0) & mask] = int(idx) + 1
            color = np.asarray(self._room_color(idx), dtype=np.float32)
            total_mask_cells += int(np.count_nonzero(mask))
            blended = out[mask].astype(np.float32) * 0.62 + color[None, :] * 0.38
            out[mask] = np.clip(blended, 0, 255).astype(np.uint8)
            boundary = self._mask_boundary(mask)
            total_boundary_cells += int(np.count_nonzero(boundary))
            out[boundary] = np.asarray(np.clip(color * 1.08, 0, 255), dtype=np.uint8)
        adjacency_boundary = self._room_label_adjacency_boundary(owner)
        if np.any(adjacency_boundary):
            total_boundary_cells += int(np.count_nonzero(adjacency_boundary))
            out[adjacency_boundary] = np.asarray((245, 250, 255), dtype=np.uint8)
        return out, total_mask_cells, total_boundary_cells

    @staticmethod
    def _mask_boundary(mask: np.ndarray) -> np.ndarray:
        arr = np.asarray(mask, dtype=bool)
        padded = np.pad(arr, 1, mode="constant", constant_values=False)
        neighbors = (
            padded[1:-1, :-2]
            & padded[1:-1, 2:]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
        )
        return arr & ~neighbors

    @staticmethod
    def _room_label_adjacency_boundary(owner: np.ndarray) -> np.ndarray:
        labels = np.asarray(owner, dtype=np.int32)
        if labels.size == 0:
            return np.zeros_like(labels, dtype=bool)
        pos = labels > 0
        boundary = np.zeros_like(pos, dtype=bool)
        boundary[:, 1:] |= pos[:, 1:] & pos[:, :-1] & (labels[:, 1:] != labels[:, :-1])
        boundary[:, :-1] |= pos[:, :-1] & pos[:, 1:] & (labels[:, :-1] != labels[:, 1:])
        boundary[1:, :] |= pos[1:, :] & pos[:-1, :] & (labels[1:, :] != labels[:-1, :])
        boundary[:-1, :] |= pos[:-1, :] & pos[1:, :] & (labels[:-1, :] != labels[1:, :])
        return boundary

    def _draw_room_labels(
        self,
        draw: ImageDraw.ImageDraw,
        room_masks: Sequence[object],
        xy_func,
        crop_bounds: Tuple[int, int, int, int],
    ) -> int:
        r0, r1, c0, c1 = crop_bounds
        count = 0
        for idx, room in enumerate(room_masks[:64]):
            center = self._room_center_cell(room)
            if center is None:
                continue
            r, c = center
            if not (r0 <= r < r1 and c0 <= c < c1):
                continue
            label = self._room_label_text(room)
            if not label:
                continue
            x, y = xy_func(center)
            color = self._room_color(idx)
            self._label(draw, (x + 5, y - 9), label[:36], color)
            self._dot(draw, (x, y), color, radius=4)
            count += 1
        return count

    def _draw_room_adjacency_debug_lines(self, draw: ImageDraw.ImageDraw, xy_func, crop_bounds: Tuple[int, int, int, int]) -> Tuple[int, int, List[dict]]:
        r0, r1, c0, c1 = crop_bounds
        merged_count = 0
        doorway_count = 0
        reasons: List[dict] = []
        for item in list(self._room_segmentation_debug.get("adjacency_evidence") or []):
            if not isinstance(item, Mapping):
                continue
            cells = []
            for raw_cell in list(item.get("boundary_cells_sample") or []):
                try:
                    row, col = int(raw_cell[0]), int(raw_cell[1])
                except Exception:
                    continue
                if r0 <= row < r1 and c0 <= col < c1:
                    cells.append((row, col))
            if not cells:
                continue
            verified = bool(item.get("verified_doorway", False))
            reason = str(item.get("merge_reason", ""))
            reasons.append(
                {
                    "room_a_label": item.get("room_a_label"),
                    "room_b_label": item.get("room_b_label"),
                    "verified_doorway": verified,
                    "merge_reason": reason,
                }
            )
            if verified:
                for cell in cells:
                    self._dot(draw, xy_func(cell), (255, 170, 40), radius=3)
                    doorway_count += 1
            else:
                for idx, cell in enumerate(cells):
                    if idx % 2 == 0:
                        self._dot(draw, xy_func(cell), (150, 150, 155), radius=2)
                        merged_count += 1
        return merged_count, doorway_count, reasons

    def _room_center_cell(self, room: object) -> Optional[GridCell]:
        metadata = getattr(room, "metadata", {}) or {}
        if isinstance(metadata, Mapping) and metadata.get("centroid_grid") is not None:
            try:
                row, col = metadata["centroid_grid"][:2]
                return int(round(float(row))), int(round(float(col)))
            except Exception:
                pass
        mask = getattr(room, "mask", None)
        if mask is None:
            return None
        rr, cc = np.nonzero(np.asarray(mask, dtype=bool))
        if rr.size == 0:
            return None
        return int(round(float(np.mean(rr)))), int(round(float(np.mean(cc))))

    def _room_label_text(self, room: object) -> str:
        room_id = str(getattr(room, "room_id", "room"))
        label = self._room_semantic_labels.get(room_id)
        if label is None:
            return room_id
        if isinstance(label, Mapping):
            category = str(label.get("category", "unknown"))
            reliability = label.get("label_reliability", label.get("confidence"))
        else:
            category = str(getattr(label, "category", "unknown"))
            reliability = getattr(label, "label_reliability", getattr(label, "confidence", None))
        if reliability is None:
            return "%s | %s" % (room_id, category)
        try:
            return "%s | %s | reliability=%.2f" % (room_id, category, float(reliability))
        except Exception:
            return "%s | %s" % (room_id, category)

    @staticmethod
    def _room_color(idx: int) -> Tuple[int, int, int]:
        palette = [
            (145, 110, 255),
            (255, 120, 120),
            (80, 190, 255),
            (255, 190, 80),
            (180, 130, 255),
            (90, 210, 170),
            (255, 145, 210),
            (210, 210, 90),
        ]
        return palette[int(idx) % len(palette)]

    def _draw_agent(self, draw: ImageDraw.ImageDraw, center: Tuple[int, int], yaw: float, scale: float) -> None:
        x, y = center
        radius = max(5, int(3 * scale))
        self._dot(draw, center, (255, 60, 60), radius=radius)
        end = (int(x + np.cos(yaw) * radius * 2.2), int(y - np.sin(yaw) * radius * 2.2))
        draw.line([center, end], fill=(255, 255, 255), width=2)

    def _draw_cells(
        self,
        draw: ImageDraw.ImageDraw,
        cells: Iterable[GridCell],
        xy_func,
        color: Tuple[int, int, int],
        radius: int,
        max_cells: int,
    ) -> int:
        cells_list = list(cells)
        if not cells_list:
            return 0
        stride = max(1, len(cells_list) // max(1, int(max_cells)))
        drawn = 0
        for cell in cells_list[::stride][:max_cells]:
            self._dot(draw, xy_func(cell), color, radius=radius)
            drawn += 1
        return drawn

    def _dot(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], color: Tuple[int, int, int], radius: int = 3) -> None:
        x, y = int(xy[0]), int(xy[1])
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    def _triangle(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], color: Tuple[int, int, int], radius: int = 4) -> None:
        x, y = int(xy[0]), int(xy[1])
        pts = [(x, y - radius), (x - radius, y + radius), (x + radius, y + radius)]
        draw.polygon(pts, fill=color)

    def _cross(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], color: Tuple[int, int, int], radius: int = 4) -> None:
        x, y = int(xy[0]), int(xy[1])
        draw.line([(x - radius, y - radius), (x + radius, y + radius)], fill=color, width=2)
        draw.line([(x - radius, y + radius), (x + radius, y - radius)], fill=color, width=2)

    def _star(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], color: Tuple[int, int, int], radius: int = 6) -> None:
        x, y = int(xy[0]), int(xy[1])
        draw.line([(x - radius, y), (x + radius, y)], fill=color, width=2)
        draw.line([(x, y - radius), (x, y + radius)], fill=color, width=2)
        draw.line([(x - radius, y - radius), (x + radius, y + radius)], fill=color, width=1)
        draw.line([(x - radius, y + radius), (x + radius, y - radius)], fill=color, width=1)

    def _draw_crosses(
        self,
        draw: ImageDraw.ImageDraw,
        cells: Iterable[GridCell],
        xy_func,
        color: Tuple[int, int, int],
        radius: int,
        max_cells: int,
    ) -> int:
        cells_list = list(cells)
        if not cells_list:
            return 0
        drawn = 0
        for cell in cells_list[: max(1, int(max_cells))]:
            self._cross(draw, xy_func(cell), color, radius=radius)
            drawn += 1
        return drawn

    def _label(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, color: Tuple[int, int, int]) -> None:
        x, y = int(xy[0]), int(xy[1])
        bbox = draw.textbbox((x, y), text, font=self._font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(10, 12, 16))
        draw.text((x, y), text, fill=color, font=self._font)

    def _legend(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int]) -> None:
        x, y = xy
        items = [
            ((255, 60, 60), "agent"),
            ((145, 110, 255), "online room"),
            ((245, 245, 245), "A*"),
            ((0, 225, 255), "frontier center"),
            ((255, 225, 40), "chosen frontier"),
            ((255, 150, 40), "goal cand"),
            ((255, 50, 50), "selected cand"),
            ((40, 220, 90), "accepted goal"),
            ((220, 70, 255), "standoff"),
            ((255, 150, 40), "planner target"),
        ]
        for color, label in items:
            self._dot(draw, (x + 6, y + 8), color, radius=5)
            draw.text((x + 16, y), label, fill=(230, 232, 235), font=self._font)
            y += 12

    @classmethod
    def _bbox_color(
        cls,
        det: Detection2D,
        goal_category: str,
        nav_decision: Optional[NavigationDecision],
    ) -> Tuple[int, int, int]:
        if cls._category_matches_goal(det.category, goal_category):
            return (255, 60, 60)
        candidate = nav_decision.selected_candidate if nav_decision and nav_decision.selected_candidate is not None else None
        if candidate is not None and cls._category_matches_goal(det.category, candidate.category):
            return (255, 60, 60)
        return (80, 230, 120)

    @classmethod
    def _visible_map_nodes(
        cls,
        object_memory: ObjectMemory,
        goal_category: str,
        nav_decision: Optional[NavigationDecision],
    ) -> List[object]:
        selected_id = cls._selected_candidate_id(nav_decision)
        visible: List[object] = []
        seen_ids = set()
        for node in object_memory.nodes:
            node_id = int(node.node_id)
            if node_id == selected_id or cls._category_matches_goal(node.category, goal_category):
                visible.append(node)
                seen_ids.add(node_id)
        candidate = nav_decision.selected_candidate if nav_decision and nav_decision.selected_candidate is not None else None
        if candidate is not None and int(candidate.node_id) not in seen_ids:
            visible.append(candidate)
        return visible

    @staticmethod
    def _selected_candidate_id(nav_decision: Optional[NavigationDecision]) -> Optional[int]:
        if nav_decision is None or nav_decision.selected_candidate is None:
            return None
        return int(nav_decision.selected_candidate.node_id)

    @classmethod
    def _candidate_node_color(
        cls,
        node,
        selected_id: Optional[int],
        nav_decision: Optional[NavigationDecision],
    ) -> Tuple[int, int, int]:
        if selected_id is None or int(node.node_id) != int(selected_id):
            return (255, 150, 40)
        meta = dict(getattr(nav_decision, "metadata", {}) or {}) if nav_decision is not None else {}
        if bool(meta.get("candidate_rejected", False)):
            return (135, 135, 135)
        if bool(meta.get("candidate_accepted", False)) or (nav_decision is not None and nav_decision.mode == "stop"):
            return (40, 220, 90)
        return (255, 50, 50)

    @staticmethod
    def _category_matches_goal(category: str, goal_category: str) -> bool:
        cat = normalize_category(category)
        goal = normalize_category(goal_category)
        return bool(goal and cat and (cat == goal or goal in cat or cat in goal))

    @staticmethod
    def _short(value) -> str:
        if isinstance(value, float):
            return "%.3f" % value
        return str(value)[:32]
