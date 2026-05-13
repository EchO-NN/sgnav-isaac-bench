from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from isaac_bench.config import repo_root
from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.graph.decision import NavigationDecision
from isaac_bench.mapping.frontier import FrontierCluster
from isaac_bench.perception.detection_types import Detection2D
from isaac_bench.perception.object_memory import ObjectMemory


GridCell = Tuple[int, int]


class SGNavPopupVisualizer:
    def __init__(
        self,
        enabled: bool = True,
        window_name: str = "SG-Nav Isaac Debug",
        save_dir: Optional[str] = None,
        panel_size: Tuple[int, int] = (960, 540),
        save_every_steps: int = 10,
        ipc_jpeg_quality: int = 75,
    ) -> None:
        self.enabled = bool(enabled)
        self.window_name = window_name
        self.save_dir = Path(save_dir) if save_dir else None
        self.panel_size = (int(panel_size[0]), int(panel_size[1]))
        self.save_every_steps = max(1, int(save_every_steps))
        self.ipc_jpeg_quality = max(30, min(95, int(ipc_jpeg_quality)))
        self._proc: Optional[subprocess.Popen[str]] = None
        self._ipc_dir = Path(tempfile.gettempdir()) / ("sgnav_viz_%d" % os.getpid())
        self._frame_path = self._ipc_dir / "latest.jpg"
        self._font = ImageFont.load_default()
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

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
        if self.enabled:
            self._send_frame(panel)
        return panel

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
        left_w = panel_w // 2
        right_w = panel_w - left_w
        rgb_h = int(panel_h * 0.67)
        text_h = panel_h - rgb_h

        panel = Image.new("RGB", (panel_w, panel_h), (18, 20, 24))
        rgb_panel = self._render_rgb(rgb, detections_2d, (left_w, rgb_h))
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
            detections_2d=detections_2d,
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

    def _render_rgb(self, rgb: np.ndarray, detections: Sequence[Detection2D], size: Tuple[int, int]) -> Image.Image:
        width, height = size
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim != 3 or arr.shape[2] < 3:
            arr = np.zeros((height, width, 3), dtype=np.uint8)
        src_h, src_w = arr.shape[:2]
        image = Image.fromarray(arr[:, :, :3], mode="RGB").resize((width, height), Image.BILINEAR)
        draw = ImageDraw.Draw(image)
        sx, sy = width / max(src_w, 1), height / max(src_h, 1)
        for det in detections[:30]:
            x1, y1, x2, y2 = det.bbox_xyxy
            box = [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]
            color = (80, 230, 120)
            draw.rectangle(box, outline=color, width=2)
            label = "%s %.2f" % (det.category, float(det.confidence))
            self._label(draw, (box[0], max(0, box[1] - 14)), label, color)
        self._label(draw, (8, 8), "RGB / YOLO detections: %d" % len(detections), (255, 255, 255))
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
        h, w = occupancy.shape
        base = np.zeros((h, w, 3), dtype=np.uint8)
        nav = navigable.astype(bool)
        obs = observed.astype(bool)
        base[nav] = (218, 222, 224)
        base[~nav] = (72, 74, 76)
        base[occupancy.astype(bool)] = (24, 24, 24)
        base[~obs] = (base[~obs].astype(np.float32) * 0.45 + np.array([20, 24, 34], dtype=np.float32)).astype(np.uint8)

        margin = 12
        scale = min((width - 2 * margin) / max(w, 1), (height - 2 * margin) / max(h, 1))
        map_w, map_h = max(1, int(w * scale)), max(1, int(h * scale))
        ox, oy = (width - map_w) // 2, (height - map_h) // 2
        image = Image.new("RGB", (width, height), (18, 20, 24))
        map_img = Image.fromarray(base, mode="RGB").resize((map_w, map_h), Image.NEAREST)
        image.paste(map_img, (ox, oy))
        draw = ImageDraw.Draw(image)

        def xy(cell: GridCell) -> Tuple[int, int]:
            r, c = int(cell[0]), int(cell[1])
            return int(ox + (c + 0.5) * scale), int(oy + (r + 0.5) * scale)

        self._draw_cells(draw, goal_cells, xy, (30, 220, 80), radius=2, max_cells=500)
        self._draw_cells(draw, full_path, xy, (80, 130, 255), radius=1, max_cells=1200)
        self._draw_cells(draw, current_path, xy, (40, 190, 255), radius=2, max_cells=500)
        for frontier in frontiers[:64]:
            self._dot(draw, xy(frontier.center_grid), (0, 225, 255), radius=3)

        selected_frontier = None
        if nav_decision and nav_decision.frontier_decision:
            selected_frontier = nav_decision.frontier_decision.selected_frontier
        if selected_frontier is not None:
            self._dot(draw, xy(selected_frontier.center_grid), (255, 225, 40), radius=7)
        if nav_decision and nav_decision.target_cells:
            self._draw_cells(draw, nav_decision.target_cells, xy, (220, 70, 255), radius=2, max_cells=400)
        if nav_decision and nav_decision.selected_candidate is not None:
            self._dot(draw, xy(nav_decision.selected_candidate.center_grid), (255, 80, 220), radius=8)

        goal_norm = normalize_category(goal_category)
        for node in object_memory.nodes[:300]:
            cat = normalize_category(node.category)
            color = (255, 178, 50)
            if cat == goal_norm or (goal_norm and (goal_norm in cat or cat in goal_norm)):
                color = (45, 245, 95)
            self._dot(draw, xy(node.center_grid), color, radius=3)

        self._draw_agent(draw, xy(current_grid), float(pose[3]) if len(pose) > 3 else 0.0, scale)
        self._label(draw, (10, 8), "Map / frontiers / A* / object memory", (255, 255, 255))
        self._legend(draw, (10, height - 76))
        return image

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
    ) -> None:
        cells_list = list(cells)
        if not cells_list:
            return
        stride = max(1, len(cells_list) // max(1, int(max_cells)))
        for cell in cells_list[::stride][:max_cells]:
            self._dot(draw, xy_func(cell), color, radius=radius)

    def _dot(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], color: Tuple[int, int, int], radius: int = 3) -> None:
        x, y = int(xy[0]), int(xy[1])
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    def _label(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, color: Tuple[int, int, int]) -> None:
        x, y = int(xy[0]), int(xy[1])
        bbox = draw.textbbox((x, y), text, font=self._font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(10, 12, 16))
        draw.text((x, y), text, fill=color, font=self._font)

    def _legend(self, draw: ImageDraw.ImageDraw, xy: Tuple[int, int]) -> None:
        x, y = xy
        items = [
            ((255, 60, 60), "agent"),
            ((40, 190, 255), "A*"),
            ((0, 225, 255), "frontier"),
            ((255, 225, 40), "chosen"),
            ((45, 245, 95), "goal/object"),
            ((220, 70, 255), "target"),
        ]
        for color, label in items:
            self._dot(draw, (x + 6, y + 8), color, radius=5)
            draw.text((x + 16, y), label, fill=(230, 232, 235), font=self._font)
            y += 12

    @staticmethod
    def _short(value) -> str:
        if isinstance(value, float):
            return "%.3f" % value
        return str(value)[:32]
