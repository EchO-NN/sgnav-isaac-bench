from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from .bottleneck import BottleneckDetector, SeparatorScorer
from .config import OVBConfig
from .corridor import CorridorAnalyzer
from .data_types import GridSpec, MaskAssignmentResult, RoomSegmentationOutput
from .free_space import FreeSpaceExtractor
from .functional_zone import FunctionalZoneAssigner
from .graph_partition import GraphPartitioner
from .mask_assignment import MaskAssigner
from .place_graph import PlaceGraphBuilder
from .skeleton import SkeletonExtractor
from .structural_wall import StructuralWallEstimator
from .temporal import TemporalSmoother
from .vertical_map_builder import VerticalMapBuilder
from .visibility import VisibilityComputer


class OnlineRoomSegmenter:
    def __init__(self, config: OVBConfig | Mapping[str, object] | None = None, grid_spec: GridSpec | None = None):
        self.config = config if isinstance(config, OVBConfig) else OVBConfig.from_mapping(config or {})
        self.grid_spec = grid_spec or GridSpec(
            resolution_m=float(self.config.map.resolution_m),
            origin_xy=(0.0, 0.0),
            width=1,
            height=1,
        )
        self.vertical_map_builder = VerticalMapBuilder(self.config.map, self.config.depth, self.config.vertical_projection, self.grid_spec)
        self.structural_wall_estimator = StructuralWallEstimator(self.config.structural_wall, resolution_m=float(self.grid_spec.resolution_m))
        self.free_space_extractor = FreeSpaceExtractor(self.config.free_space, self.config.structural_wall, resolution_m=float(self.grid_spec.resolution_m))
        self.skeleton_extractor = SkeletonExtractor(self.config.skeleton, grid_spec=self.grid_spec)
        self.visibility = VisibilityComputer(self.config.visibility, self.config.structural_wall, resolution_m=float(self.grid_spec.resolution_m))
        self.bottleneck_detector = BottleneckDetector(self.config.bottleneck, self.config.structural_wall, grid_spec=self.grid_spec)
        self.separator_scorer = SeparatorScorer(
            self.config.separator_scoring,
            self.config.bottleneck,
            self.config.temporal,
            visibility=self.visibility,
            resolution_m=float(self.grid_spec.resolution_m),
        )
        self.place_graph_builder = PlaceGraphBuilder(self.config.place_graph, grid_spec=self.grid_spec, visibility=self.visibility)
        self.graph_partitioner = GraphPartitioner(self.config.partition)
        self.mask_assigner = MaskAssigner(self.grid_spec)
        self.corridor_analyzer = CorridorAnalyzer(self.config.corridor, resolution_m=float(self.grid_spec.resolution_m))
        self.functional_zone_assigner = FunctionalZoneAssigner(self.config.functional_zone, self.config.open_space, grid_spec=self.grid_spec)
        self.temporal_smoother = TemporalSmoother(self.config.temporal, grid_spec=self.grid_spec)
        self.last_output: RoomSegmentationOutput | None = None

    def reset(self) -> None:
        self.vertical_map_builder.reset()
        self.place_graph_builder.reset()
        self.temporal_smoother.reset()
        self.last_output = None

    def update(
        self,
        depth: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_pose_world: np.ndarray,
        frame_id: int,
        semantic_objects: Optional[list[dict]] = None,
        existing_2d_map: Optional[np.ndarray | Mapping[str, np.ndarray]] = None,
    ) -> RoomSegmentationOutput:
        self.visibility.set_frame_id(int(frame_id), int(self.config.runtime.full_recompute_interval_frames))
        if isinstance(existing_2d_map, Mapping) and "observed_free_mask" in existing_2d_map:
            return self.update_from_masks(
                observed_free_mask=np.asarray(existing_2d_map["observed_free_mask"]),
                obstacle_mask=np.asarray(existing_2d_map.get("obstacle_mask", existing_2d_map.get("wall_mask", np.zeros_like(existing_2d_map["observed_free_mask"])))),
                unknown_mask=np.asarray(existing_2d_map.get("unknown_mask", np.zeros_like(existing_2d_map["observed_free_mask"]))),
                frame_id=int(frame_id),
                semantic_objects=semantic_objects,
                existing_2d_map=None,
                camera_pose_world=camera_pose_world,
            )
        vertical_state = self.vertical_map_builder.update_from_depth(
            depth=depth,
            camera_intrinsics=camera_intrinsics,
            camera_pose_world=camera_pose_world,
            frame_id=int(frame_id),
        )
        structural_map = self.structural_wall_estimator.update(vertical_state, existing_2d_map if isinstance(existing_2d_map, np.ndarray) else None)
        return self._run_pipeline(structural_map, int(frame_id), semantic_objects or [], camera_pose_world=camera_pose_world)

    def update_from_masks(
        self,
        observed_free_mask: np.ndarray,
        obstacle_mask: np.ndarray | None = None,
        unknown_mask: np.ndarray | None = None,
        frame_id: int = 0,
        semantic_objects: Optional[list[dict]] = None,
        existing_2d_map: Optional[np.ndarray] = None,
        camera_pose_world: Optional[np.ndarray] = None,
    ) -> RoomSegmentationOutput:
        free = np.asarray(observed_free_mask, dtype=bool)
        self._ensure_grid_shape(free.shape)
        self.visibility.set_frame_id(int(frame_id), int(self.config.runtime.full_recompute_interval_frames))
        vertical_state = self.vertical_map_builder.from_masks(
            observed_free_mask=free,
            obstacle_mask=np.zeros_like(free, dtype=bool) if obstacle_mask is None else np.asarray(obstacle_mask, dtype=bool),
            unknown_mask=(~free if unknown_mask is None else np.asarray(unknown_mask, dtype=bool)),
            frame_id=int(frame_id),
        )
        structural_map = self.structural_wall_estimator.update(vertical_state, existing_2d_map)
        return self._run_pipeline(structural_map, int(frame_id), semantic_objects or [], camera_pose_world=camera_pose_world)

    def _run_pipeline(
        self,
        structural_map,
        frame_id: int,
        semantic_objects: list[dict],
        camera_pose_world: Optional[np.ndarray] = None,
    ) -> RoomSegmentationOutput:
        free_space = self.free_space_extractor.extract(structural_map)
        skeleton_graph = self.skeleton_extractor.extract(free_space, structural_map)
        cut_candidates = self.bottleneck_detector.generate(skeleton_graph, structural_map, free_space)
        cut_candidates = self.separator_scorer.score_all(
            cut_candidates,
            skeleton_graph,
            structural_map,
            free_space,
            temporal_state=self.temporal_smoother.get_cut_history(),
        )
        place_graph = self.place_graph_builder.build(
            skeleton_graph,
            cut_candidates,
            structural_map,
            semantic_objects=semantic_objects,
            frame_id=int(frame_id),
            camera_pose_world=camera_pose_world,
        )
        partition = self.graph_partitioner.partition(place_graph, cut_candidates, structural_map)
        mask_result = self.mask_assigner.assign(partition, place_graph, cut_candidates, structural_map, free_space)
        corridor_result = self.corridor_analyzer.refine(mask_result, skeleton_graph, structural_map, cut_candidates)
        zone_result = self.functional_zone_assigner.assign(corridor_result, semantic_objects, structural_map, place_graph)
        output = self._build_output(zone_result, structural_map, free_space, skeleton_graph, cut_candidates, place_graph)
        output = self.temporal_smoother.update(output, cut_candidates, int(frame_id))
        self.last_output = output
        return output

    def _build_output(
        self,
        result: MaskAssignmentResult,
        structural_map,
        free_space,
        skeleton_graph,
        cut_candidates,
        place_graph,
    ) -> RoomSegmentationOutput:
        debug_layers = dict(result.debug_layers)
        cut_score_map = debug_layers.get("cut_score_map", np.zeros(result.room_id_map.shape, dtype=np.float32))
        hard_separator = np.zeros(result.room_id_map.shape, dtype=bool)
        soft_separator = np.zeros(result.room_id_map.shape, dtype=bool)
        for cand in cut_candidates:
            for cell in cand.cut_cells:
                if 0 <= cell[0] < hard_separator.shape[0] and 0 <= cell[1] < hard_separator.shape[1]:
                    hard_separator[cell] = hard_separator[cell] or bool(cand.is_hard_separator)
                    soft_separator[cell] = soft_separator[cell] or bool(cand.is_soft_separator)
        place_node_map = np.zeros(result.room_id_map.shape, dtype=np.int32)
        for node in place_graph.nodes:
            if 0 <= node.uv[0] < place_node_map.shape[0] and 0 <= node.uv[1] < place_node_map.shape[1]:
                place_node_map[node.uv] = int(node.node_id) + 1
        debug_layers.update(
            {
                "p_free": np.asarray(structural_map.p_free, dtype=np.float32),
                "p_occupied": np.asarray(structural_map.debug.get("p_occupied", np.zeros_like(structural_map.p_free)), dtype=np.float32),
                "p_unknown": np.asarray(structural_map.p_unknown, dtype=np.float32),
                "p_wall": np.asarray(structural_map.p_wall, dtype=np.float32),
                "hard_wall_mask": np.asarray(structural_map.hard_wall_mask, dtype=np.uint8),
                "observed_free_mask": np.asarray(structural_map.observed_free_mask, dtype=np.uint8),
                "frontier_mask": np.asarray(structural_map.frontier_mask, dtype=np.uint8),
                "distance_transform": np.asarray(free_space.distance_transform_m, dtype=np.float32),
                "skeleton_mask": np.asarray(skeleton_graph.skeleton_mask, dtype=np.uint8),
                "cut_score_map": np.asarray(cut_score_map, dtype=np.float32),
                "hard_separator_map": hard_separator.astype(np.uint8),
                "soft_separator_map": soft_separator.astype(np.uint8),
                "place_graph_node_map": place_node_map.astype(np.int32),
                "stable_room_id_map": np.asarray(result.room_id_map, dtype=np.int32),
                "corridor_mask": np.asarray(result.corridor_mask, dtype=np.uint8),
                "open_space_mask": np.asarray(result.open_space_mask, dtype=np.uint8),
                "functional_zone_map": np.asarray(result.functional_zone_map, dtype=np.int32),
                "room_confidence_map": np.asarray(result.room_confidence_map, dtype=np.float32),
            }
        )
        return RoomSegmentationOutput(
            room_id_map=np.asarray(result.room_id_map, dtype=np.int32),
            room_confidence_map=np.asarray(result.room_confidence_map, dtype=np.float32),
            room_soft_masks={int(k): np.asarray(v, dtype=np.float32) for k, v in result.room_soft_masks.items()},
            corridor_mask=np.asarray(result.corridor_mask, dtype=bool),
            open_space_mask=np.asarray(result.open_space_mask, dtype=bool),
            functional_zone_map=np.asarray(result.functional_zone_map, dtype=np.int32),
            room_instances=list(result.room_instances),
            room_graph_edges=list(result.room_graph_edges),
            cut_candidates=list(cut_candidates),
            debug_layers=debug_layers,
        )

    def _ensure_grid_shape(self, shape: tuple[int, int]) -> None:
        h, w = int(shape[0]), int(shape[1])
        if h == int(self.grid_spec.height) and w == int(self.grid_spec.width):
            return
        self.grid_spec.height = h
        self.grid_spec.width = w
        self.vertical_map_builder = VerticalMapBuilder(self.config.map, self.config.depth, self.config.vertical_projection, self.grid_spec)
        self.skeleton_extractor.grid_spec = self.grid_spec
        self.bottleneck_detector.grid_spec = self.grid_spec
        self.place_graph_builder.grid_spec = self.grid_spec
        self.mask_assigner = MaskAssigner(self.grid_spec)
        self.functional_zone_assigner.grid_spec = self.grid_spec
        self.temporal_smoother.grid_spec = self.grid_spec

