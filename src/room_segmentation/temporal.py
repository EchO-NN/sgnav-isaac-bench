from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .config import TemporalConfig
from .data_types import CutCandidate, GridSpec, RoomInstance, RoomSegmentationOutput
from .utils import centroid_xy


class TemporalSmoother:
    def __init__(self, config: TemporalConfig | Mapping[str, object] | None = None, grid_spec: GridSpec | None = None):
        self.config = config if isinstance(config, TemporalConfig) else TemporalConfig.from_mapping(config or {})
        self.grid_spec = grid_spec
        self._cut_history: list[dict[str, object]] = []
        self._rooms: dict[int, dict[str, object]] = {}
        self._next_room_id = 1

    def reset(self) -> None:
        self._cut_history.clear()
        self._rooms.clear()
        self._next_room_id = 1

    def get_cut_history(self) -> Mapping[tuple[int, int], Mapping[str, float]]:
        out: dict[tuple[int, int], dict[str, float]] = {}
        for item in self._cut_history:
            center = item.get("center_uv")
            if not isinstance(center, tuple):
                continue
            out[(int(center[0]), int(center[1]))] = {
                "score": float(item.get("score", 0.0)),
                "observations": int(item.get("observations", 0)),
            }
        return out

    def update_cuts(self, cut_candidates: list[CutCandidate]) -> None:
        if not bool(self.config.enabled):
            return
        alpha = float(self.config.score_ema_alpha)
        updated: list[dict[str, object]] = []
        used: set[int] = set()
        for cand in cut_candidates:
            best_idx = None
            best_dist = float("inf")
            for idx, hist in enumerate(self._cut_history):
                if idx in used:
                    continue
                center = hist.get("center_uv")
                normal = hist.get("normal_xy")
                if not isinstance(center, tuple) or not isinstance(normal, tuple):
                    continue
                dist = math.hypot(int(center[0]) - int(cand.center_uv[0]), int(center[1]) - int(cand.center_uv[1]))
                angle = _normal_angle_diff(normal, cand.normal_xy)
                if dist <= max(1.0, 0.30 / max(float(self.grid_spec.resolution_m) if self.grid_spec else 0.05, 1e-6)) and angle <= math.radians(25.0):
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
            if best_idx is None:
                score = float(cand.final_score)
                observations = 1 if score >= float(self.config.soft_split_threshold) else 0
            else:
                hist = self._cut_history[best_idx]
                used.add(best_idx)
                score = alpha * float(cand.final_score) + (1.0 - alpha) * float(hist.get("score", cand.final_score))
                if score >= float(self.config.soft_split_threshold):
                    observations = int(hist.get("observations", 0)) + 1
                else:
                    observations = max(0, int(hist.get("observations", 0)) - 1)
            cand.temporal_score = float(score)
            cand.temporal_observations = int(observations)
            quality_ok = bool(cand.debug.get("quality_ok", cand.is_soft_separator))
            cand.is_soft_separator = bool(score >= float(self.config.soft_split_threshold) and quality_ok)
            cand.is_hard_separator = bool(score >= float(self.config.hard_split_threshold) and observations >= int(self.config.split_persistence_observations))
            updated.append(
                {
                    "center_uv": (int(cand.center_uv[0]), int(cand.center_uv[1])),
                    "normal_xy": (float(cand.normal_xy[0]), float(cand.normal_xy[1])),
                    "score": float(score),
                    "observations": int(observations),
                }
            )
        self._cut_history = updated[:256]

    def update(self, current_output: RoomSegmentationOutput, cut_candidates: list[CutCandidate], frame_id: int) -> RoomSegmentationOutput:
        if not bool(self.config.enabled):
            return current_output
        self.update_cuts(cut_candidates)
        labels = np.asarray(current_output.room_id_map, dtype=np.int32)
        confidence = np.asarray(current_output.room_confidence_map, dtype=np.float32).copy()
        current_labels = [int(v) for v in np.unique(labels) if int(v) > 0]
        matches: dict[int, int] = {}
        used_prev: set[int] = set()
        candidates = []
        for label in current_labels:
            mask = labels == label
            center = centroid_xy(mask, self.grid_spec) if self.grid_spec is not None else _centroid_idx(mask)
            for prev_id, prev in self._rooms.items():
                prev_mask = np.asarray(prev["mask"], dtype=bool)
                iou = _mask_iou(mask, prev_mask)
                dist = math.hypot(float(center[0]) - float(prev["centroid_xy"][0]), float(center[1]) - float(prev["centroid_xy"][1]))
                match_score = 0.70 * iou + 0.30 * math.exp(-dist / max(float(self.config.room_id_centroid_distance_threshold_m), 1e-6))
                if iou >= float(self.config.room_id_iou_match_threshold) or match_score >= 0.45:
                    candidates.append((match_score, iou, -dist, label, prev_id))
        candidates.sort(reverse=True)
        for _score, _iou, _neg_dist, label, prev_id in candidates:
            if label in matches or prev_id in used_prev:
                continue
            matches[label] = prev_id
            used_prev.add(prev_id)
        new_map = np.full(labels.shape, -1, dtype=np.int32)
        new_map[labels == 0] = 0
        instances_by_label = {int(inst.room_id): inst for inst in current_output.room_instances}
        new_instances: list[RoomInstance] = []
        new_rooms: dict[int, dict[str, object]] = {}
        label_to_room: dict[int, int] = {}
        for label in current_labels:
            room_id = matches.get(label)
            if room_id is None:
                room_id = self._next_room_id
                self._next_room_id += 1
            label_to_room[int(label)] = int(room_id)
            mask = labels == label
            new_map[mask] = int(room_id)
            inst = instances_by_label.get(label)
            if inst is not None:
                inst.room_id = int(room_id)
                inst.centroid_xy = centroid_xy(mask, self.grid_spec) if self.grid_spec is not None else _centroid_idx(mask)
                inst.confidence = float(np.mean(confidence[mask])) if np.any(mask) else float(inst.confidence)
                new_instances.append(inst)
            new_rooms[int(room_id)] = {
                "mask": mask.copy(),
                "centroid_xy": centroid_xy(mask, self.grid_spec) if self.grid_spec is not None else _centroid_idx(mask),
                "last_frame": int(frame_id),
                "confidence": float(np.mean(confidence[mask])) if np.any(mask) else 0.0,
            }
        for prev_id, prev in self._rooms.items():
            if prev_id in used_prev:
                continue
            age = int(frame_id) - int(prev.get("last_frame", frame_id))
            if age <= int(self.config.max_room_id_age_frames):
                mask = np.asarray(prev["mask"], dtype=bool)
                keep = mask & (new_map < 0)
                if np.any(keep):
                    new_map[keep] = int(prev_id)
                    confidence[keep] = np.asarray(confidence[keep], dtype=np.float32) * float(self.config.confidence_decay_unobserved) ** max(1, age)
                    new_rooms[int(prev_id)] = {
                        "mask": mask.copy(),
                        "centroid_xy": prev["centroid_xy"],
                        "last_frame": int(prev.get("last_frame", frame_id)),
                        "confidence": float(prev.get("confidence", 0.0)) * float(self.config.confidence_decay_unobserved) ** max(1, age),
                    }
        self._rooms = new_rooms
        remapped_soft = {}
        for label, soft in current_output.room_soft_masks.items():
            room_id = label_to_room.get(int(label), int(label))
            remapped_soft[int(room_id)] = np.asarray(soft, dtype=np.float32)
        edges = []
        for a, b, weight in current_output.room_graph_edges:
            ra = label_to_room.get(int(a), int(a))
            rb = label_to_room.get(int(b), int(b))
            if ra != rb:
                edges.append((int(ra), int(rb), float(weight)))
        connected: dict[int, set[int]] = {int(inst.room_id): set() for inst in new_instances}
        for a, b, _weight in edges:
            connected.setdefault(int(a), set()).add(int(b))
            connected.setdefault(int(b), set()).add(int(a))
        for inst in new_instances:
            inst.connected_room_ids = sorted(connected.get(int(inst.room_id), set()))
        debug_layers = dict(current_output.debug_layers)
        debug_layers["stable_room_id_map"] = new_map.astype(np.int32)
        return RoomSegmentationOutput(
            room_id_map=new_map.astype(np.int32),
            room_confidence_map=confidence.astype(np.float32),
            room_soft_masks=remapped_soft,
            corridor_mask=current_output.corridor_mask,
            open_space_mask=current_output.open_space_mask,
            functional_zone_map=current_output.functional_zone_map,
            room_instances=new_instances,
            room_graph_edges=edges,
            cut_candidates=cut_candidates,
            debug_layers=debug_layers,
        )


def _normal_angle_diff(a: tuple[float, float], b: tuple[float, float]) -> float:
    aa = math.atan2(float(a[1]), float(a[0]))
    bb = math.atan2(float(b[1]), float(b[0]))
    diff = abs((aa - bb + math.pi) % (2.0 * math.pi) - math.pi)
    return min(diff, abs(math.pi - diff))


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    inter = int(np.count_nonzero(np.asarray(a, dtype=bool) & np.asarray(b, dtype=bool)))
    union = int(np.count_nonzero(np.asarray(a, dtype=bool) | np.asarray(b, dtype=bool)))
    return float(inter) / float(max(1, union))


def _centroid_idx(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return 0.0, 0.0
    return float(np.mean(cols)), float(np.mean(rows))
