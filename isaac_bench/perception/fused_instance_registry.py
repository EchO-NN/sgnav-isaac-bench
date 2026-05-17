from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from isaac_bench.dataset.category_normalizer import normalize_category
from isaac_bench.perception.detection_types import (
    MIN_VALID_DETECTION_CONFIDENCE,
    Detection2D,
    Detection3D,
    FusedInstance,
    bbox_touches_image_edge,
    detection_confidence_is_valid,
)
from isaac_bench.sensors.camera_geometry import CameraIntrinsics
from isaac_bench.sensors.depth_backproject import detection_to_world_points, world_points_bbox


DEFAULT_ROOM_CATEGORIES = {
    "bedroom",
    "bathroom",
    "kitchen",
    "living_room",
    "living room",
    "dining_room",
    "dining room",
    "office",
    "hallway",
    "unknown_room",
    "unknown room",
}


class FusedInstanceRegistry:
    """Cross-frame registry for YOLO-World + SAM2 + RGB-D projected instances."""

    def __init__(
        self,
        merge_distance_m: float = 0.75,
        merge_iou_3d: float = 0.15,
        max_points_per_instance: int = 4096,
        room_categories: Optional[Iterable[str]] = None,
        min_valid_confidence: float = MIN_VALID_DETECTION_CONFIDENCE,
        reject_edge_touching_bboxes: bool = False,
        bbox_edge_margin_px: float = 2,
        bbox_edge_margin_ratio: float = 0.0,
    ):
        self.merge_distance_m = float(merge_distance_m)
        self.merge_iou_3d = float(merge_iou_3d)
        self.max_points_per_instance = max(1, int(max_points_per_instance))
        self.room_categories = {normalize_category(name) for name in (room_categories or DEFAULT_ROOM_CATEGORIES)}
        self.min_valid_confidence = float(min_valid_confidence)
        self.reject_edge_touching_bboxes = bool(reject_edge_touching_bboxes)
        self.bbox_edge_margin_px = float(bbox_edge_margin_px)
        self.bbox_edge_margin_ratio = float(bbox_edge_margin_ratio)
        self.instances: List[FusedInstance] = []
        self._next_id = 0
        self.raw_detection_log: List[dict] = []
        self.raw_rejected_detections_count = 0

    def reset(self) -> None:
        self.instances = []
        self._next_id = 0
        self.raw_detection_log = []
        self.raw_rejected_detections_count = 0

    def update(
        self,
        detections: Iterable[Detection2D],
        depth: np.ndarray,
        intr: CameraIntrinsics,
        camera_pose_world: Tuple[float, float, float, float],
        step_id: int,
        depth_max_m: float = 6.0,
        min_points: int = 20,
        stride: int = 4,
    ) -> List[FusedInstance]:
        for det in detections:
            raw_record = self._raw_detection_record(det, step_id, intr.width, intr.height)
            self.raw_detection_log.append(raw_record)
            if raw_record["bbox_touches_edge"] and self.reject_edge_touching_bboxes:
                self.raw_rejected_detections_count += 1
                if len(self.instances) == 1:
                    self.instances[0].edge_rejected_count = int(self.instances[0].edge_rejected_count) + 1
                continue
            if not detection_confidence_is_valid(float(det.confidence), self.min_valid_confidence):
                raw_record["used_for_object_track"] = False
                raw_record["reject_reason"] = "low_confidence"
                continue
            points = detection_to_world_points(
                det,
                depth,
                intr,
                camera_pose_world,
                depth_max_m=depth_max_m,
                min_points=min_points,
                stride=stride,
            )
            if points is None or len(points) == 0:
                raw_record["used_for_object_track"] = False
                raw_record["reject_reason"] = "insufficient_depth_points"
                continue
            category = normalize_category(det.category)
            node_type = self._node_type_for_category(category)
            bbox_world = world_points_bbox(points)
            center_world = np.median(np.asarray(points, dtype=np.float32), axis=0).astype(np.float32)
            match = self._find_match(category, node_type, center_world, bbox_world)
            if match is None:
                class_conf_sums = {category: float(det.confidence)}
                class_hits = {category: 1}
                self.instances.append(
                    FusedInstance(
                        instance_id="%s_%d" % (node_type, self._next_id),
                        category=category,
                        node_type=node_type,
                        confidence=float(det.confidence),
                        point_cloud_world=self._downsample_points(points),
                        bbox_world=bbox_world,
                        center_world=center_world,
                        last_mask=self._copy_mask(det.mask),
                        last_seen_step=int(step_id),
                        observed_count=1,
                        class_conf_sums=class_conf_sums,
                        class_hits=class_hits,
                        valid_detection_count=1,
                        total_conf_sum=float(det.confidence),
                    )
                )
                self._next_id += 1
                raw_record["used_for_object_track"] = True
                continue
            self._merge(match, points, bbox_world, center_world, det, step_id)
            raw_record["used_for_object_track"] = True
        return list(self.instances)

    def to_detections_3d(self, node_type: str = "object") -> List[Detection3D]:
        out: List[Detection3D] = []
        for instance in self.instances:
            if instance.node_type != node_type:
                continue
            out.append(
                Detection3D(
                    category=instance.category,
                    raw_label=instance.category,
                    confidence=float(instance.mean_confidence),
                    center_world=tuple(float(v) for v in instance.center_world),
                    bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                    point_cloud_world=instance.point_cloud_world,
                    bbox_world=instance.bbox_world,
                    mask=instance.last_mask,
                )
            )
        return out

    def _find_match(
        self,
        category: str,
        node_type: str,
        center_world: Sequence[float],
        bbox_world: np.ndarray,
    ) -> Optional[FusedInstance]:
        best: Optional[FusedInstance] = None
        best_dist = float("inf")
        for instance in self.instances:
            _ = category
            if instance.node_type != node_type:
                continue
            dist = float(np.linalg.norm(np.asarray(instance.center_world[:2], dtype=np.float32) - np.asarray(center_world[:2], dtype=np.float32)))
            if dist > self.merge_distance_m or dist >= best_dist:
                continue
            iou = bbox_iou_3d(instance.bbox_world, bbox_world)
            if iou < self.merge_iou_3d:
                continue
            best = instance
            best_dist = dist
        return best

    def _merge(
        self,
        instance: FusedInstance,
        points: np.ndarray,
        bbox_world: np.ndarray,
        center_world: np.ndarray,
        det: Detection2D,
        step_id: int,
    ) -> None:
        old_count = max(1, int(instance.observed_count))
        merged_points = np.concatenate([np.asarray(instance.point_cloud_world, dtype=np.float32), np.asarray(points, dtype=np.float32)], axis=0)
        instance.point_cloud_world = self._downsample_points(merged_points)
        instance.bbox_world = world_points_bbox(instance.point_cloud_world)
        weighted_center = (np.asarray(instance.center_world, dtype=np.float32) * float(old_count) + np.asarray(center_world, dtype=np.float32)) / float(old_count + 1)
        bbox_center = np.mean(np.asarray(bbox_world, dtype=np.float32), axis=0)
        instance.center_world = ((weighted_center + bbox_center) * 0.5).astype(np.float32)
        instance.confidence = max(float(instance.confidence), float(det.confidence))
        instance.last_mask = self._copy_mask(det.mask)
        instance.last_seen_step = int(step_id)
        instance.observed_count = old_count + 1
        category = normalize_category(det.category)
        instance.class_conf_sums[category] = float(instance.class_conf_sums.get(category, 0.0)) + float(det.confidence)
        instance.class_hits[category] = int(instance.class_hits.get(category, 0)) + 1
        instance.valid_detection_count = int(instance.valid_detection_count) + 1
        instance.total_conf_sum = float(instance.total_conf_sum) + float(det.confidence)
        instance.category = instance.stable_category

    def _node_type_for_category(self, category: str) -> str:
        return "room" if normalize_category(category) in self.room_categories else "object"

    def _downsample_points(self, points: np.ndarray) -> np.ndarray:
        arr = np.asarray(points, dtype=np.float32)
        if len(arr) <= self.max_points_per_instance:
            return arr.copy()
        indices = np.linspace(0, len(arr) - 1, self.max_points_per_instance).astype(np.int64)
        return arr[indices].copy()

    @staticmethod
    def _copy_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        return np.asarray(mask).astype(bool).copy()

    def _raw_detection_record(self, det: Detection2D, step_id: int, width: int, height: int) -> dict:
        touches = bbox_touches_image_edge(
            det.bbox_xyxy,
            width,
            height,
            margin_px=self.bbox_edge_margin_px,
            margin_ratio=self.bbox_edge_margin_ratio,
        )
        rejected = bool(touches and self.reject_edge_touching_bboxes)
        det.bbox_touches_edge = bool(touches)
        det.used_for_object_track = not rejected
        det.reject_reason = "bbox_touches_image_edge" if touches and self.reject_edge_touching_bboxes else None
        return {
            "raw_detection_id": "frame_%04d_det_%04d" % (int(step_id), len(self.raw_detection_log)),
            "step": int(step_id),
            "category": normalize_category(det.category),
            "raw_label": str(det.raw_label),
            "confidence": float(det.confidence),
            "bbox_xyxy": [float(v) for v in det.bbox_xyxy],
            "bbox_touches_edge": bool(touches),
            "used_for_object_track": not rejected,
            "reject_reason": "bbox_touches_image_edge" if touches and self.reject_edge_touching_bboxes else None,
        }


def bbox_iou_3d(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.shape != (2, 3) or bb.shape != (2, 3):
        return 0.0
    aa = _inflate_degenerate_bbox(aa)
    bb = _inflate_degenerate_bbox(bb)
    inter_min = np.maximum(aa[0], bb[0])
    inter_max = np.minimum(aa[1], bb[1])
    inter_extent = np.maximum(inter_max - inter_min, 0.0)
    extent_a = np.maximum(aa[1] - aa[0], 0.0)
    extent_b = np.maximum(bb[1] - bb[0], 0.0)
    if float(np.prod(extent_a)) <= 1e-9 or float(np.prod(extent_b)) <= 1e-9:
        inter_area = float(np.prod(inter_extent[:2]))
        area_a = float(np.prod(extent_a[:2]))
        area_b = float(np.prod(extent_b[:2]))
        denom_area = area_a + area_b - inter_area
        return 0.0 if denom_area <= 1e-9 else float(inter_area / denom_area)
    inter_vol = float(np.prod(inter_extent))
    vol_a = float(np.prod(extent_a))
    vol_b = float(np.prod(extent_b))
    denom = vol_a + vol_b - inter_vol
    if denom <= 1e-9:
        return 0.0
    return float(inter_vol / denom)


def _inflate_degenerate_bbox(bbox: np.ndarray, min_extent: float = 0.20) -> np.ndarray:
    out = np.asarray(bbox, dtype=np.float32).copy()
    center = np.mean(out, axis=0)
    extent = out[1] - out[0]
    half = np.maximum(extent, float(min_extent)) * 0.5
    out[0] = center - half
    out[1] = center + half
    return out
